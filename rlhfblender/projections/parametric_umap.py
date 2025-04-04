import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from umap import UMAP
from warnings import warn, catch_warnings, filterwarnings
import os
from umap.spectral import spectral_layout
from sklearn.utils import check_random_state, check_array
import pickle
from sklearn.neighbors import KDTree


class EdgeDataset(Dataset):
    """PyTorch Dataset for UMAP edge sampling."""
    
    def __init__(self, X, head, tail, weights=None, parametric_embedding=True, 
                 parametric_reconstruction=False):
        """
        Initialize the EdgeDataset for UMAP edges.
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The input data
        head : array
            Head indices of edges
        tail : array
            Tail indices of edges
        weights : array, optional
            Edge weights
        parametric_embedding : bool
            Whether to use parametric embedding
        parametric_reconstruction : bool
            Whether to use parametric reconstruction
        """
        self.X = X
        self.head = head
        self.tail = tail
        self.weights = weights
        self.parametric_embedding = parametric_embedding
        self.parametric_reconstruction = parametric_reconstruction
        
    def __len__(self):
        return len(self.head)
    
    def __getitem__(self, idx):
        head_idx = self.head[idx]
        tail_idx = self.tail[idx]
        
        head_sample = self.X[head_idx]
        tail_sample = self.X[tail_idx]
        
        # Create output dictionary based on what's needed
        outputs = {}
        
        if self.weights is not None:
            weight = self.weights[idx] if idx < len(self.weights) else 1.0
        else:
            weight = 1.0
            
        # Return appropriate data based on model type
        return (head_sample, tail_sample), weight


class ParametricUMAP(UMAP):
    def __init__(
        self,
        optimizer=None,
        batch_size=None,
        dims=None,
        encoder=None,
        decoder=None,
        parametric_embedding=True,
        parametric_reconstruction=False,
        parametric_reconstruction_loss_cls=nn.BCEWithLogitsLoss,
        parametric_reconstruction_loss_weight=1.0,
        autoencoder_loss=False,
        reconstruction_validation=None,
        loss_report_frequency=10,
        n_training_epochs=1,
        global_correlation_loss_weight=0,
        landmark_loss_fn=None,
        landmark_loss_weight=1.0,
        torch_fit_kwargs={},
        **kwargs
    ):
        """
        Parametric UMAP subclassing UMAP-learn, based on PyTorch.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer, optional
            The optimizer used for embedding, by default None
        batch_size : int, optional
            size of batch used for batch training, by default None
        dims : tuple, optional
            dimensionality of data, if not flat (e.g. (32x32x3) images for ConvNet), by default None
        encoder : torch.nn.Module, optional
            The encoder PyTorch network
        decoder : torch.nn.Module, optional
            The decoder PyTorch network
        parametric_embedding : bool, optional
            Whether the embedder is parametric or non-parametric, by default True
        parametric_reconstruction : bool, optional
            Whether the decoder is parametric or non-parametric, by default False
        parametric_reconstruction_loss_cls : callable, optional
            What loss function to use for parametric reconstruction, by default nn.BCEWithLogitsLoss
        parametric_reconstruction_loss_weight : float, optional
            How to weight the parametric reconstruction loss relative to umap loss, by default 1.0
        autoencoder_loss : bool, optional
            Whether to use autoencoder loss, by default False
        reconstruction_validation : array, optional
            validation X data for reconstruction loss, by default None
        loss_report_frequency : int, optional
            how many times per epoch to report loss, by default 10
        n_training_epochs : int, optional
            number of epochs to train for, by default 1
        global_correlation_loss_weight : float, optional
            Whether to additionally train on correlation of global pairwise relationships (>0), by default 0
        landmark_loss_fn : callable, optional
            The function to use for landmark loss, by default None
        landmark_loss_weight : float, optional
            How to weight the landmark loss relative to umap loss, by default 1.0
        torch_fit_kwargs : dict, optional
            additional arguments for model training (like callbacks), by default {}
        """
        super().__init__(**kwargs)
        
        # Add to network
        self.dims = dims  # if this is an image, we should reshape for network
        self.encoder = encoder  # neural network used for embedding
        self.decoder = decoder  # neural network used for decoding
        self.parametric_embedding = parametric_embedding
        self.parametric_reconstruction = parametric_reconstruction
        self.parametric_reconstruction_loss_cls = parametric_reconstruction_loss_cls
        self.parametric_reconstruction_loss_weight = parametric_reconstruction_loss_weight
        self.autoencoder_loss = autoencoder_loss
        self.batch_size = batch_size
        self.loss_report_frequency = loss_report_frequency
        self.global_correlation_loss_weight = global_correlation_loss_weight
        self.landmark_loss_fn = landmark_loss_fn
        self.landmark_loss_weight = landmark_loss_weight
        self.prev_epoch_X = None
        
        self.reconstruction_validation = reconstruction_validation  # holdout data for reconstruction acc
        self.torch_fit_kwargs = torch_fit_kwargs  # arguments for training
        self.parametric_model = None
        
        # How many epochs to train for
        self.n_training_epochs = n_training_epochs
        
        # Set optimizer
        self.optimizer_class = None
        self.lr = None
        
        if optimizer is None:
            if parametric_embedding:
                # Adam is better for parametric_embedding
                self.optimizer_class = torch.optim.Adam
                self.lr = 1e-3
            else:
                # Larger learning rate can be used for embedding
                self.optimizer_class = torch.optim.Adam
                self.lr = 1e-1
        else:
            self.optimizer = optimizer
        
        if parametric_reconstruction and not parametric_embedding:
            warn(
                "Parametric decoding is not implemented with nonparametric "
                "embedding. Turning off parametric decoding"
            )
            self.parametric_reconstruction = False
        
        # Device configuration - use CUDA if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.encoder is not None:
            if hasattr(self.encoder, 'output_shape') and self.encoder.output_shape[-1] != self.n_components:
                raise ValueError(
                    f"Dimensionality of embedder network output ({self.encoder.output_shape[-1]}) does "
                    f"not match n_components ({self.n_components})"
                )

    def fit(self, X, y=None, precomputed_distances=None, landmark_positions=None):
        """Fit X into an embedded space.

        Optionally use a precomputed distance matrix, y for supervised
        dimension reduction, or landmarked positions.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Contains a sample per row.
            
        y : array, shape (n_samples)
            A target array for supervised dimension reduction. How this is
            handled is determined by parameters UMAP was instantiated with.
            
        precomputed_distances : array, shape (n_samples, n_samples), optional
            A precomputed distance matrix.
            
        landmark_positions : array, shape (n_samples, n_components), optional
            The desired position in low-dimensional space of each sample in X.
            Points that are not landmarks should have nan coordinates.
        """
        if (self.prev_epoch_X is not None) & (landmark_positions is None):
            # Add the landmark points for training, then make a landmark vector.
            landmark_positions = np.stack(
                [np.array([np.nan, np.nan])]*X.shape[0] + list(
                    self.transform(
                        self.prev_epoch_X
                    )
                )
            )
            X = np.concatenate((X, self.prev_epoch_X))

        if landmark_positions is not None:
            len_X = len(X)
            len_land = len(landmark_positions)
            if len_X != len_land:
                raise ValueError(
                    f"Length of X = {len_X}, length of landmark_positions "
                    f"= {len_land}, while it must be equal."
                )

        if self.metric == "precomputed":
            if precomputed_distances is None:
                raise ValueError(
                    "Precomputed distances must be supplied if metric "
                    "is precomputed."
                )
            # prepare X for training the network
            self._X = X
            # generate the graph on precomputed distances
            return super().fit(
                precomputed_distances, y, landmark_positions=landmark_positions
            )
        else:
            return super().fit(X, y, landmark_positions=landmark_positions)

    def fit_transform(self, X, y=None, precomputed_distances=None, landmark_positions=None):
        """Fit X into an embedded space and return that transformed output.

        Optionally use a precomputed distance matrix, y for supervised
        dimension reduction, or landmarked positions.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Contains a sample per row.
            
        y : array, shape (n_samples)
            A target array for supervised dimension reduction. How this is
            handled is determined by parameters UMAP was instantiated with.
            
        precomputed_distances : array, shape (n_samples, n_samples), optional
            A precomputed distance matrix.
            
        landmark_positions : array, shape (n_samples, n_components), optional
            The desired position in low-dimensional space of each sample in X.
            Points that are not landmarks should have nan coordinates.
        """
        if (self.prev_epoch_X is not None) & (landmark_positions is None):
            # Add the landmark points for training, then make a landmark vector.
            landmark_positions = np.stack(
                [np.array([np.nan, np.nan])]*X.shape[0] + list(
                    self.transform(
                        self.prev_epoch_X
                    )
                )
            )
            X = np.concatenate((X, self.prev_epoch_X))

        if landmark_positions is not None:
            len_X = len(X)
            len_land = len(landmark_positions)
            if len_X != len_land:
                raise ValueError(
                    f"Length of X = {len_X}, length of landmark_positions "
                    f"= {len_land}, while it must be equal."
                )

        if self.metric == "precomputed":
            if precomputed_distances is None:
                raise ValueError(
                    "Precomputed distances must be supplied if metric "
                    "is precomputed."
                )
            # prepare X for training the network
            self._X = X
            # generate the graph on precomputed distances
            return super().fit_transform(
                precomputed_distances, y, landmark_positions=landmark_positions
            )
        else:
            return super().fit_transform(X, y, landmark_positions=landmark_positions)

    def transform(self, X, batch_size=None):
        """Transform X into the existing embedded space and return that
        transformed output.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            New data to be transformed.
        batch_size : int, optional
            Batch size for transformation, defaults to the self.batch_size used in training.

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the new data in low-dimensional space.
        """
        if self.parametric_embedding:
            if batch_size is None:
                batch_size = self.batch_size
            
            if self.dims is not None and len(self.dims) > 1:
                X = np.reshape(X, [len(X)] + list(self.dims))
            
            # Convert to PyTorch tensor and move to appropriate device
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            
            # Process in batches to avoid memory issues
            embeddings = []
            with torch.no_grad():
                self.encoder.eval()
                for i in range(0, len(X), batch_size):
                    batch = X_tensor[i:i+batch_size]
                    embedding = self.encoder(batch)
                    embeddings.append(embedding.cpu().numpy())
            
            return np.vstack(embeddings)
        else:
            warn(
                "Embedding new data is not supported by ParametricUMAP "
                "in non-parametric mode. Using original UMAP transform."
            )
            return super().transform(X)

    def inverse_transform(self, X, batch_size=None):
        """Transform X in the existing embedded space back into the input
        data space and return that transformed output.

        Parameters
        ----------
        X : array, shape (n_samples, n_components)
            New points to be inverse transformed.
        batch_size : int, optional
            Batch size for transformation, defaults to the self.batch_size used in training.

        Returns
        -------
        X_new : array, shape (n_samples, n_features)
            Generated data points new data in data space.
        """
        if self.parametric_reconstruction:
            if batch_size is None:
                batch_size = self.batch_size
            
            # Convert to PyTorch tensor and move to appropriate device
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            
            # Process in batches to avoid memory issues
            reconstructions = []
            with torch.no_grad():
                self.decoder.eval()
                for i in range(0, len(X), batch_size):
                    batch = X_tensor[i:i+batch_size]
                    reconstruction = self.decoder(batch)
                    reconstructions.append(reconstruction.cpu().numpy())
            
            result = np.vstack(reconstructions)
            
            # Reshape if necessary
            if self.dims is not None and len(self.dims) > 1:
                result = np.reshape(result, [len(result)] + list(self.dims))
                
            return result
        else:
            return super().inverse_transform(X)

    def _define_model(self):
        """Define the model in PyTorch"""
        self.parametric_model = UMAPModel(
            a=self._a,
            b=self._b,
            negative_sample_rate=self.negative_sample_rate,
            encoder=self.encoder,
            decoder=self.decoder,
            parametric_embedding=self.parametric_embedding,
            parametric_reconstruction=self.parametric_reconstruction,
            parametric_reconstruction_loss_weight=self.parametric_reconstruction_loss_weight,
            parametric_reconstruction_loss_cls=self.parametric_reconstruction_loss_cls,
            global_correlation_loss_weight=self.global_correlation_loss_weight,
            autoencoder_loss=self.autoencoder_loss,
            landmark_loss_fn=self.landmark_loss_fn,
            landmark_loss_weight=self.landmark_loss_weight,
        ).to(self.device)
        
        # Set up optimizer
        if self.optimizer_class:
            self.optimizer = self.optimizer_class(
                self.parametric_model.parameters(), lr=self.lr
            )
        elif not hasattr(self, 'optimizer'):
            self.optimizer = torch.optim.Adam(
                self.parametric_model.parameters(), lr=1e-3
            )

    def _fit_embed_data(self, X, n_epochs, init, random_state, landmark_positions=None):
        """
        Fit the embedding model using provided data.
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data to be embedded
        n_epochs : int
            Number of epochs for each edge
        init : str or array
            Initial embedding strategy
        random_state : numpy.random.RandomState
            Random state for reproducibility
        landmark_positions : array, optional
            Target positions for landmarks
            
        Returns
        -------
        embedding : array
            The embedding of X
        """
        if self.metric == "precomputed":
            X = self._X

        # get dimensionality of dataset
        if self.dims is None:
            self.dims = [np.shape(X)[-1]]
        else:
            # reshape data for network
            if len(self.dims) > 1:
                X = np.reshape(X, [len(X)] + list(self.dims))

        if self.parametric_reconstruction and (np.max(X) > 1.0 or np.min(X) < 0.0):
            warn(
                "Data should be scaled to the range 0-1 for cross-entropy reconstruction loss."
            )

        # Make sure landmark_positions is float32.
        if landmark_positions is not None:
            landmark_positions = check_array(
                landmark_positions, dtype=np.float32, ensure_all_finite="allow-nan"
            )

        # get dataset of edges
        (
            dataloader,
            n_edges,
            head,
            tail,
            self.edge_weight
        ) = self._construct_edge_dataset(
            X,
            self.graph_,
            self.n_epochs,
            self.batch_size,
            landmark_positions=landmark_positions
        )
        
        self.head = torch.from_numpy(head.astype(np.int64)).unsqueeze(0).to(self.device)
        self.tail = torch.from_numpy(tail.astype(np.int64)).unsqueeze(0).to(self.device)

        if not self.parametric_embedding:
            init_embedding = init_embedding_from_graph(
                X,
                self.graph_,
                self.n_components,
                random_state,
                self.metric,
                self._metric_kwds,
                init="spectral",
            )
        else:
            init_embedding = None

        # create encoder and decoder model
        n_data = len(X)
        self.encoder, self.decoder = prepare_networks(
            self.encoder,
            self.decoder,
            self.n_components,
            self.dims,
            n_data,
            self.parametric_embedding,
            self.parametric_reconstruction,
            init_embedding,
        )

        # Define the model
        self._define_model()

        # Validation dataset for reconstruction
        if (
            self.parametric_reconstruction
            and self.reconstruction_validation is not None
        ):
            # reshape data for network
            if len(self.dims) > 1:
                validation_data = np.reshape(
                    self.reconstruction_validation,
                    [len(self.reconstruction_validation)] + list(self.dims)
                )
            else:
                validation_data = self.reconstruction_validation.copy()
                
            validation_tensor = torch.tensor(validation_data, dtype=torch.float32).to(self.device)
        else:
            validation_tensor = None

        # Training loop
        best_loss = float("inf")
        steps_per_epoch = max(1, n_edges // (self.batch_size * self.loss_report_frequency))
        total_steps = steps_per_epoch * self.n_training_epochs
        
        self.parametric_model.train()
        for step, (data, weight) in enumerate(dataloader):
            if step >= total_steps:
                break
                
            # Unpack data
            head_sample, tail_sample = data
            
            # Move tensors to the right device
            head_sample = head_sample.to(self.device)
            tail_sample = tail_sample.to(self.device)
            weight = weight.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Check if we need landmark loss
            landmark_data = None
            if landmark_positions is not None:
                # Get indices of the current batch
                batch_indices = torch.arange(step * self.batch_size, 
                                            min((step + 1) * self.batch_size, len(head)))
                landmark_data = torch.tensor(
                    landmark_positions[batch_indices], dtype=torch.float32
                ).to(self.device)
            
            # Compute loss
            loss = self.parametric_model(
                head_sample, 
                tail_sample, 
                weight=weight,
                landmark_data=landmark_data
            )
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parametric_model.parameters(), 4.0)
            self.optimizer.step()
            
            # Report progress
            if step % steps_per_epoch == 0:
                # Validation step
                if validation_tensor is not None:
                    with torch.no_grad():
                        embeddings = self.encoder(validation_tensor)
                        reconstructions = self.decoder(embeddings)
                        val_loss = F.binary_cross_entropy_with_logits(
                            reconstructions, validation_tensor
                        )
                        
                        # Early stopping check
                        if val_loss < best_loss:
                            best_loss = val_loss
                        else:
                            if step > steps_per_epoch * 2:  # Allow at least 2 epochs
                                print(f"Early stopping at step {step}/{total_steps}")
                                break
                
                # Print progress
                if self.verbose:
                    print(f"Step {step}/{total_steps}, Loss: {loss.item():.4f}")

        # Get the final embedding
        if self.parametric_embedding:
            self.encoder.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                embeddings = []
                for i in range(0, len(X), self.batch_size):
                    batch = X_tensor[i:i+self.batch_size]
                    embedding = self.encoder(batch)
                    embeddings.append(embedding.cpu().numpy())
                embedding = np.vstack(embeddings)
        else:
            embedding = self.parametric_model.embedding.cpu().detach().numpy()

        return embedding, {}

    def _construct_edge_dataset(self, X, graph_, n_epochs, batch_size, landmark_positions=None):
        """
        Construct a DataLoader of edges, sampled by edge weight.
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data to be embedded
        graph_ : scipy.sparse.csr.csr_matrix
            UMAP graph
        n_epochs : int
            Number of epochs for each edge
        batch_size : int
            Batch size for training
        landmark_positions : array, optional
            Target positions for landmarks
            
        Returns
        -------
        dataloader : torch.utils.data.DataLoader
            DataLoader for edges
        n_edges : int
            Number of edges
        head : array
            Edge heads
        tail : array
            Edge tails
        weight : array
            Edge weights
        """
        # Get graph elements
        graph, epochs_per_sample, head, tail, weight, n_vertices = get_graph_elements(
            graph_, n_epochs
        )
        
        # Default batch size if not provided
        if batch_size is None:
            batch_size = min(n_vertices, 1000)
            
        # Repeat edges based on importance (epochs_per_sample)
        edges_to_exp, edges_from_exp = (
            np.repeat(head, epochs_per_sample.astype("int")),
            np.repeat(tail, epochs_per_sample.astype("int")),
        )
        
        # Shuffle edges
        shuffle_mask = np.random.permutation(range(len(edges_to_exp)))
        edges_to_exp = edges_to_exp[shuffle_mask].astype(np.int64)
        edges_from_exp = edges_from_exp[shuffle_mask].astype(np.int64)
        
        # Create PyTorch Dataset
        edge_dataset = EdgeDataset(
            X, 
            edges_to_exp, 
            edges_from_exp, 
            weights=weight,
            parametric_embedding=self.parametric_embedding,
            parametric_reconstruction=self.parametric_reconstruction
        )
        
        # Create DataLoader
        dataloader = DataLoader(
            edge_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0
        )
        
        return dataloader, len(edges_to_exp), head, tail, weight

    def add_landmarks(
        self,
        X,
        sample_pct=0.01,
        sample_mode="uniform",
        landmark_loss_weight=0.01,
        idx=None,
    ):
        """Add some points from a dataset X as "landmarks."

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Old data to be retained.
        sample_pct : float, optional
            Percentage of old data to use as landmarks.
        sample_mode : str, optional
            Method for sampling points. Allows "uniform" and "predefined."
        landmark_loss_weight : float, optional
            Multiplier for landmark loss function.
        idx : array, optional
            Indices of samples to use as landmarks if sample_mode="predetermined".
        """
        self.sample_pct = sample_pct
        self.sample_mode = sample_mode
        self.landmark_loss_weight = landmark_loss_weight

        if self.sample_mode == "uniform":
            self.prev_epoch_idx = list(
                np.random.choice(
                    range(X.shape[0]), int(X.shape[0]*sample_pct), replace=False
                )
            )
            self.prev_epoch_X = X[self.prev_epoch_idx]
        elif self.sample_mode == "predetermined":
            if idx is None:
                raise ValueError(
                    "Choice of sample_mode is not supported."
                )
            else:
                self.prev_epoch_idx = idx
                self.prev_epoch_X = X[self.prev_epoch_idx]
        else:
            raise ValueError(
                "Choice of sample_mode is not supported."
            )

    def remove_landmarks(self):
        """Remove landmarks from model."""
        self.prev_epoch_X = None

    def save(self, save_location, verbose=True):
        """
        Save the model to disk.
        
        Parameters
        ----------
        save_location : str
            Directory to save the model
        verbose : bool, optional
            Whether to print progress, by default True
        """
        os.makedirs(save_location, exist_ok=True)
        
        # Save encoder if available
        if self.encoder is not None:
            encoder_output = os.path.join(save_location, "encoder.pt")
            torch.save(self.encoder.state_dict(), encoder_output)
            if verbose:
                print(f"PyTorch encoder model saved to {encoder_output}")
        
        # Save decoder if available
        if self.decoder is not None:
            decoder_output = os.path.join(save_location, "decoder.pt")
            torch.save(self.decoder.state_dict(), decoder_output)
            if verbose:
                print(f"PyTorch decoder model saved to {decoder_output}")
        
        # Save full model if available
        if self.parametric_model is not None:
            model_output = os.path.join(save_location, "parametric_model.pt")
            torch.save(self.parametric_model.state_dict(), model_output)
            if verbose:
                print(f"PyTorch full model saved to {model_output}")
        
        # Save pickle of the model object (ignoring unpickleable warnings)
        with catch_warnings():
            filterwarnings("ignore")
            model_output = os.path.join(save_location, "model.pkl")
            with open(model_output, "wb") as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
            if verbose:
                print(f"Pickle of ParametricUMAP model saved to {model_output}")
    
    def to_ONNX(self, save_location):
        """
        Export trained parametric UMAP encoder as ONNX.
        
        Parameters
        ----------
        save_location : str
            Path to save the ONNX model
        """
        if self.encoder is None:
            raise ValueError("Encoder must be trained before exporting to ONNX")
            
        # Create dummy input
        if len(self.dims) > 1:
            dummy_input = torch.randn(1, *self.dims).to(self.device)
        else:
            dummy_input = torch.randn(1, self.dims[0]).to(self.device)
            
        # Export to ONNX
        torch.onnx.export(
            self.encoder,
            dummy_input,
            save_location,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        return save_location
        

def load_ParametricUMAP(save_location, verbose=True):
    """
    Load a parametric UMAP model and corresponding PyTorch models.
    
    Parameters
    ----------
    save_location : str
        Directory containing the saved model
    verbose : bool, optional
        Whether to print progress, by default True
        
    Returns
    -------
    ParametricUMAP
        Loaded model
    """
    # Load pickle of the model
    model_output = os.path.join(save_location, "model.pkl")
    model = pickle.load(open(model_output, "rb"))
    if verbose:
        print(f"Pickle of ParametricUMAP model loaded from {model_output}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load encoder
    encoder_output = os.path.join(save_location, "encoder.pt")
    if os.path.exists(encoder_output):
        if model.encoder is not None:
            model.encoder.load_state_dict(torch.load(encoder_output, map_location=device))
        if verbose:
            print(f"PyTorch encoder model loaded from {encoder_output}")
    
    # Load decoder
    decoder_output = os.path.join(save_location, "decoder.pt")
    if os.path.exists(decoder_output):
        if model.decoder is not None:
            model.decoder.load_state_dict(torch.load(decoder_output, map_location=device))
        if verbose:
            print(f"PyTorch decoder model loaded from {decoder_output}")
    
    # Load full model
    parametric_model_output = os.path.join(save_location, "parametric_model.pt")
    if os.path.exists(parametric_model_output):
        # Recreate the model first
        model._define_model()
        # Then load the state dict
        model.parametric_model.load_state_dict(
            torch.load(parametric_model_output, map_location=device)
        )
        if verbose:
            print(f"PyTorch full model loaded from {parametric_model_output}")
    
    return model


class UMAPModel(nn.Module):
    """PyTorch model for UMAP dimensionality reduction."""
    
    def __init__(
        self,
        a,
        b,
        negative_sample_rate,
        encoder=None,
        decoder=None,
        parametric_embedding=True,
        parametric_reconstruction=False,
        parametric_reconstruction_loss_cls=nn.BCEWithLogitsLoss,
        parametric_reconstruction_loss_weight=1.0,
        global_correlation_loss_weight=0.0,
        autoencoder_loss=False,
        landmark_loss_fn=None,
        landmark_loss_weight=1.0
    ):
        """
        Initialize the UMAP model.
        
        Parameters
        ----------
        a : float
            Parameter in the UMAP loss function
        b : float
            Parameter in the UMAP loss function
        negative_sample_rate : int
            Number of negative samples per positive sample
        encoder : torch.nn.Module, optional
            The encoder network
        decoder : torch.nn.Module, optional
            The decoder network
        parametric_embedding : bool, optional
            Whether to use parametric embedding
        parametric_reconstruction : bool, optional
            Whether to use parametric reconstruction
        parametric_reconstruction_loss_cls : callable, optional
            Loss function for reconstruction
        parametric_reconstruction_loss_weight : float, optional
            Weight for reconstruction loss
        global_correlation_loss_weight : float, optional
            Weight for global correlation loss
        autoencoder_loss : bool, optional
            Whether to use autoencoder loss
        landmark_loss_fn : callable, optional
            Function for landmark loss
        landmark_loss_weight : float, optional
            Weight for landmark loss
        """
        super().__init__()
        
        self.a = a
        self.b = b
        self.negative_sample_rate = negative_sample_rate
        self.encoder = encoder
        self.decoder = decoder
        self.parametric_embedding = parametric_embedding
        self.parametric_reconstruction = parametric_reconstruction
        self.parametric_reconstruction_loss_weight = parametric_reconstruction_loss_weight
        self.global_correlation_loss_weight = global_correlation_loss_weight
        self.autoencoder_loss = autoencoder_loss
        self.landmark_loss_weight = landmark_loss_weight
        
        # Initialize loss functions
        self.parametric_reconstruction_loss_fn = parametric_reconstruction_loss_cls()
        
        if landmark_loss_fn is None:
            self.landmark_loss_fn = self._default_landmark_loss
        else:
            self.landmark_loss_fn = landmark_loss_fn
            
        # For non-parametric embedding
        if not parametric_embedding:
            self.embedding = nn.Parameter(torch.randn(1, 1))
    
    def forward(self, head_input, tail_input, weight=None, landmark_data=None):
        """
        Forward pass for UMAP model.
        
        Parameters
        ----------
        head_input : torch.Tensor
            Input for head of edge
        tail_input : torch.Tensor
            Input for tail of edge
        weight : torch.Tensor, optional
            Edge weights
        landmark_data : torch.Tensor, optional
            Target positions for landmarks
            
        Returns
        -------
        loss : torch.Tensor
            The computed loss
        """
        # Compute embeddings
        embedding_to = self.encoder(head_input)
        embedding_from = self.encoder(tail_input)
        
        # Initialize total loss
        total_loss = 0.0
        
        # UMAP loss
        umap_loss = self._umap_loss(embedding_to, embedding_from)
        total_loss += umap_loss
        
        # Global correlation loss
        if self.global_correlation_loss_weight > 0:
            global_corr_loss = self._global_correlation_loss(head_input, embedding_to)
            total_loss += self.global_correlation_loss_weight * global_corr_loss
        
        # Reconstruction loss
        if self.parametric_reconstruction:
            if self.autoencoder_loss:
                # Allow gradient flow from reconstruction to encoder
                embedding_to_recon = self.decoder(embedding_to)
            else:
                # Stop gradient to encoder
                embedding_to_recon = self.decoder(embedding_to.detach())
                
            recon_loss = self.parametric_reconstruction_loss_fn(
                embedding_to_recon, head_input
            )
            total_loss += self.parametric_reconstruction_loss_weight * recon_loss
        
        # Landmark loss
        if landmark_data is not None and self.landmark_loss_weight > 0:
            landmark_loss = self._compute_landmark_loss(embedding_to, landmark_data)
            total_loss += self.landmark_loss_weight * landmark_loss
        
        return total_loss
        
    def _umap_loss(self, embedding_to, embedding_from, repulsion_strength=1.0):
        """
        Compute UMAP loss between embeddings.
        
        Parameters
        ----------
        embedding_to : torch.Tensor
            Embeddings for head points
        embedding_from : torch.Tensor
            Embeddings for tail points
        repulsion_strength : float, optional
            Strength of repulsion, by default 1.0
            
        Returns
        -------
        ce_loss : torch.Tensor
            Cross-entropy loss
        """
        batch_size = embedding_to.shape[0]
        
        # Get negative samples
        embedding_neg_to = embedding_to.repeat(self.negative_sample_rate, 1)
        repeat_neg = embedding_from.repeat(self.negative_sample_rate, 1)
        
        # Shuffle the negative samples
        permutation = torch.randperm(repeat_neg.shape[0])
        embedding_neg_from = repeat_neg[permutation]
        
        # Compute distances
        distance_embedding = torch.cat([
            torch.norm(embedding_to - embedding_from, dim=1),
            torch.norm(embedding_neg_to - embedding_neg_from, dim=1)
        ])
        
        # Convert distances to probabilities
        probabilities_distance = self._convert_distance_to_probability(
            distance_embedding, self.a, self.b
        )
        
        # Set true probabilities based on negative sampling
        probabilities_graph = torch.cat([
            torch.ones(batch_size, device=embedding_to.device),
            torch.zeros(batch_size * self.negative_sample_rate, device=embedding_to.device)
        ])
        
        # Compute cross entropy
        attraction_loss, repellant_loss, ce_loss = self._compute_cross_entropy(
            probabilities_graph,
            probabilities_distance,
            repulsion_strength=repulsion_strength
        )
        
        return torch.mean(ce_loss)
    
    def _global_correlation_loss(self, x, z_x):
        """
        Compute correlation loss between inputs and their embeddings.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data
        z_x : torch.Tensor
            Embeddings
            
        Returns
        -------
        corr_d : torch.Tensor
            Correlation loss
        """
        # Flatten data
        x = torch.flatten(x, start_dim=1)
        z_x = torch.flatten(z_x, start_dim=1)
        
        # Z-score data
        def z_score(tensor):
            return (tensor - torch.mean(tensor)) / (torch.std(tensor) + 1e-10)
        
        x = z_score(x)
        z_x = z_score(z_x)
        
        # Clip to avoid outliers
        x = torch.clamp(x, -10, 10)
        z_x = torch.clamp(z_x, -10, 10)
        
        # Compute pairwise distances
        if x.shape[0] > 1:  # Need at least 2 samples to compute correlation
            dx = torch.norm(x[1:] - x[:-1], dim=1)
            dz = torch.norm(z_x[1:] - z_x[:-1], dim=1)
            
            # Add small noise to prevent mode collapse
            dz = dz + torch.rand_like(dz) * 1e-10
            
            # Compute correlation coefficient
            corr_d = self._correlation(dx, dz)
            
            # Negative correlation is better (we want similar inputs to have similar embeddings)
            return -corr_d
        else:
            return torch.tensor(0.0, device=x.device)
    
    def _compute_landmark_loss(self, embeddings, landmark_positions):
        """
        Compute loss for landmark positions.
        
        Parameters
        ----------
        embeddings : torch.Tensor
            Computed embeddings
        landmark_positions : torch.Tensor
            Target landmark positions
            
        Returns
        -------
        loss : torch.Tensor
            Landmark loss
        """
        # Replace NaN positions with zeros and create mask
        mask = ~torch.isnan(landmark_positions).any(dim=1)
        
        if torch.sum(mask) == 0:
            # No landmarks in this batch
            return torch.tensor(0.0, device=embeddings.device)
            
        # Extract valid landmark positions and corresponding embeddings
        valid_landmarks = landmark_positions[mask]
        valid_embeddings = embeddings[mask]
        
        # Compute landmark loss
        return self.landmark_loss_fn(valid_embeddings, valid_landmarks)
    
    def _default_landmark_loss(self, y_pred, y):
        """
        Default loss function for landmarks.
        
        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted embeddings
        y : torch.Tensor
            Target landmark positions
            
        Returns
        -------
        loss : torch.Tensor
            Euclidean distance between points
        """
        # ReLU smooths gradients
        return F.relu(torch.mean(torch.norm(y_pred - y, dim=1)))
    
    def _convert_distance_to_probability(self, distances, a=1.0, b=1.0):
        """
        Convert distance to probability using UMAP formula.
        
        Parameters
        ----------
        distances : torch.Tensor
            Distances between points
        a : float, optional
            Parameter a, by default 1.0
        b : float, optional
            Parameter b, by default 1.0
            
        Returns
        -------
        probabilities : torch.Tensor
            Probabilities
        """
        return 1.0 / (1.0 + a * torch.pow(distances, 2 * b))
    
    def _compute_cross_entropy(self, probabilities_graph, probabilities_distance, 
                              eps=1e-4, repulsion_strength=1.0):
        """
        Compute cross entropy loss.
        
        Parameters
        ----------
        probabilities_graph : torch.Tensor
            High-dimensional probabilities
        probabilities_distance : torch.Tensor
            Low-dimensional probabilities
        eps : float, optional
            Small constant for numerical stability, by default 1e-4
        repulsion_strength : float, optional
            Strength of repulsion, by default 1.0
            
        Returns
        -------
        attraction_term : torch.Tensor
            Attraction term
        repellant_term : torch.Tensor
            Repellant term
        ce_loss : torch.Tensor
            Cross-entropy loss
        """
        # Attraction term (positive samples)
        attraction_term = -probabilities_graph * torch.log(
            torch.clamp(probabilities_distance, min=eps, max=1.0)
        )
        
        # Repulsion term (negative samples)
        repellant_term = -(1.0 - probabilities_graph) * torch.log(
            torch.clamp(1.0 - probabilities_distance, min=eps, max=1.0)
        ) * repulsion_strength
        
        # Total cross-entropy loss
        ce_loss = attraction_term + repellant_term
        
        return attraction_term, repellant_term, ce_loss
    
    def _correlation(self, x, y):
        """
        Compute correlation coefficient between x and y.
        
        Parameters
        ----------
        x : torch.Tensor
            First tensor
        y : torch.Tensor
            Second tensor
            
        Returns
        -------
        correlation : torch.Tensor
            Pearson correlation coefficient
        """
        # Mean
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)
        
        # Covariance
        xm = x - mean_x
        ym = y - mean_y
        
        # Pearson correlation coefficient
        r_num = torch.sum(xm * ym)
        r_den = torch.sqrt(torch.sum(xm**2) * torch.sum(ym**2))
        
        r = r_num / (r_den + 1e-8)  # Add small epsilon to avoid division by zero
        
        return r


def get_graph_elements(graph_, n_epochs):
    """
    Extract elements from UMAP graph.
    
    Parameters
    ----------
    graph_ : scipy.sparse.csr.csr_matrix
        UMAP graph of probabilities
    n_epochs : int
        Maximum number of epochs per edge
        
    Returns
    -------
    graph : scipy.sparse.csr.csr_matrix
        UMAP graph
    epochs_per_sample : array
        Number of epochs to train each sample
    head : array
        Edge head indices
    tail : array
        Edge tail indices
    weight : array
        Edge weights
    n_vertices : int
        Number of vertices in graph
    """
    graph = graph_.tocoo()
    
    # Eliminate duplicate entries by summing them
    graph.sum_duplicates()
    
    # Number of vertices in dataset
    n_vertices = graph.shape[1]
    
    # Get the number of epochs based on dataset size
    if n_epochs is None:
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200
    
    # Remove elements with very low probability
    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()
    
    # Get epochs per sample based on edge probability
    epochs_per_sample = n_epochs * graph.data
    
    head = graph.row
    tail = graph.col
    weight = graph.data
    
    return graph, epochs_per_sample, head, tail, weight, n_vertices


def init_embedding_from_graph(
    _raw_data, graph, n_components, random_state, metric, _metric_kwds, init="spectral"
):
    """Initialize embedding using graph for non-parametric embedding.

    Parameters
    ----------
    _raw_data : array
        The raw data
    graph : scipy.sparse.csr.csr_matrix
        UMAP graph
    n_components : int
        Dimension of the embedding
    random_state : numpy.random.RandomState
        Random state for reproducibility
    metric : str
        Distance metric
    _metric_kwds : dict
        Keyword arguments for the metric
    init : str, optional
        Initialization method, by default "spectral"

    Returns
    -------
    embedding : array
        Initialized embedding
    """
    if random_state is None:
        random_state = check_random_state(None)

    if isinstance(init, str) and init == "random":
        embedding = random_state.uniform(
            low=-10.0, high=10.0, size=(graph.shape[0], n_components)
        ).astype(np.float32)
    elif isinstance(init, str) and init == "spectral":
        # We add a little noise to avoid local minima for optimization to come
        initialisation = spectral_layout(
            _raw_data,
            graph,
            n_components,
            random_state,
            metric=metric,
            metric_kwds=_metric_kwds,
        )
        expansion = 10.0 / np.abs(initialisation).max()
        embedding = (initialisation * expansion).astype(
            np.float32
        ) + random_state.normal(
            scale=0.0001, size=[graph.shape[0], n_components]
        ).astype(
            np.float32
        )
    else:
        init_data = np.array(init)
        if len(init_data.shape) == 2:
            if np.unique(init_data, axis=0).shape[0] < init_data.shape[0]:
                tree = KDTree(init_data)
                dist, ind = tree.query(init_data, k=2)
                nndist = np.mean(dist[:, 1])
                embedding = init_data + random_state.normal(
                    scale=0.001 * nndist, size=init_data.shape
                ).astype(np.float32)
            else:
                embedding = init_data

    return embedding


def prepare_networks(
    encoder,
    decoder,
    n_components,
    dims,
    n_data,
    parametric_embedding,
    parametric_reconstruction,
    init_embedding=None,
):
    """
    Generate PyTorch networks for encoder and decoder.
    
    Parameters
    ----------
    encoder : torch.nn.Module
        Encoder network
    decoder : torch.nn.Module
        Decoder network
    n_components : int
        Dimension of the embedding
    dims : list
        Dimensions of the input data
    n_data : int
        Number of data points
    parametric_embedding : bool
        Whether to use parametric embedding
    parametric_reconstruction : bool
        Whether to use parametric reconstruction
    init_embedding : array, optional
        Initial embedding for non-parametric UMAP
        
    Returns
    -------
    encoder : torch.nn.Module
        Encoder network
    decoder : torch.nn.Module
        Decoder network
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flat_dim = np.prod(dims)
    
    if encoder is None:
        if parametric_embedding:
            if len(dims) > 1:
                # For multi-dimensional inputs (e.g., images)
                encoder_layers = [
                    nn.Flatten(),
                    nn.Linear(flat_dim, 100),
                    nn.ReLU(),
                    nn.Linear(100, 100),
                    nn.ReLU(),
                    nn.Linear(100, 100),
                    nn.ReLU(),
                    nn.Linear(100, n_components),
                ]
            else:
                # For flat inputs
                encoder_layers = [
                    nn.Linear(dims[0], 100),
                    nn.ReLU(),
                    nn.Linear(100, 100),
                    nn.ReLU(),
                    nn.Linear(100, 100),
                    nn.ReLU(),
                    nn.Linear(100, n_components),
                ]
            
            encoder = nn.Sequential(*encoder_layers).to(device)
        else:
            # For non-parametric embedding, create an embedding layer
            if init_embedding is not None:
                # Use provided initial embedding
                embedding_layer = nn.Embedding(n_data, n_components)
                embedding_layer.weight.data = torch.tensor(
                    init_embedding, dtype=torch.float32
                )
                embedding_layer = embedding_layer.to(device)
                
                # Wrap in a module that returns the weights directly
                class NonParametricEncoder(nn.Module):
                    def __init__(self, embedding_layer):
                        super().__init__()
                        self.embedding = embedding_layer
                        
                    def forward(self, indices):
                        if isinstance(indices, np.ndarray):
                            indices = torch.from_numpy(indices).to(device)
                        return self.embedding(indices.long())
                
                encoder = NonParametricEncoder(embedding_layer).to(device)
            else:
                # Create random initialization
                embedding_layer = nn.Embedding(n_data, n_components)
                embedding_layer = embedding_layer.to(device)
                
                class NonParametricEncoder(nn.Module):
                    def __init__(self, embedding_layer):
                        super().__init__()
                        self.embedding = embedding_layer
                        
                    def forward(self, indices):
                        if isinstance(indices, np.ndarray):
                            indices = torch.from_numpy(indices).to(device)
                        return self.embedding(indices.long())
                
                encoder = NonParametricEncoder(embedding_layer).to(device)
    
    if decoder is None and parametric_reconstruction:
        if len(dims) > 1:
            # For multi-dimensional reconstruction (e.g., images)
            decoder_layers = [
                nn.Linear(n_components, 100),
                nn.ReLU(),
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100, flat_dim),
            ]
            
            if len(dims) > 1:
                # Add reshape layer for multi-dimensional output
                final_shape = [-1] + list(dims)
                
                class ReshapeLayer(nn.Module):
                    def __init__(self, shape):
                        super().__init__()
                        self.shape = shape
                        
                    def forward(self, x):
                        return x.view(*self.shape)
                
                decoder_layers.append(ReshapeLayer(final_shape))
        else:
            # For flat reconstruction
            decoder_layers = [
                nn.Linear(n_components, 100),
                nn.ReLU(),
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100, dims[0]),
            ]
        
        decoder = nn.Sequential(*decoder_layers).to(device)
    
    return encoder, decoder


# Main class with direct PyTorch implementation
class PumapNet(nn.Module):
    """PyTorch network for ParametricUMAP for ONNX export."""
    
    def __init__(self, indim, outdim):
        """
        Initialize the network.
        
        Parameters
        ----------
        indim : int
            Input dimension
        outdim : int
            Output dimension
        """
        super().__init__()
        self.dense1 = nn.Linear(indim, 100)
        self.dense2 = nn.Linear(100, 100)
        self.dense3 = nn.Linear(100, 100)
        self.dense4 = nn.Linear(100, outdim)
        
    def forward(self, x):
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
            
        Returns
        -------
        x : torch.Tensor
            Output tensor
        """
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = self.dense4(x)
        return x