import os
import pickle
from warnings import warn

import numpy as np
import torch as th
import torch.nn
from numba import TypingError
from sklearn.neighbors import KDTree
from sklearn.utils import check_random_state
from umap import UMAP
from umap.spectral import spectral_layout


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
        parametric_reconstruction_loss_cls=torch.nn.BCELoss,
        parametric_reconstruction_loss_weight=1.0,
        autoencoder_loss=False,
        reconstruction_validation=None,
        loss_report_frequency=10,
        n_training_epochs=1,
        global_correlation_loss_weight=0,
        torch_fit_kwargs={},
        **kwargs
    ):
        """
        Parametric UMAP subclassing UMAP-learn, based on torch.
        There is also a non-parametric implementation contained within to compare
        with the base non-parametric implementation.

        Parameters
        ----------
        optimizer : th.optim.Ooptimizer, optional
            The tensorflow optimizer used for embedding, by default None
        batch_size : int, optional
            size of batch used for batch training, by default None
        dims :  tuple, optional
            dimensionality of data, if not flat (e.g. (32x32x3 images for ConvNet), by default None
        encoder : th.nn.Module, optional
            The encoder Torch network
        decoder : th.nn.Module, optional
            The decoder Torch network
        parametric_embedding : bool, optional
            Whether the embedder is parametric or non-parametric, by default True
        parametric_reconstruction : bool, optional
            Whether the decoder is parametric or non-parametric, by default False
        parametric_reconstruction_loss_fcn : bool, optional
            What loss function to use for parametric reconstruction, by default th.nn.BCELoss
        parametric_reconstruction_loss_weight : float, optional
            How to weight the parametric reconstruction loss relative to umap loss, by default 1.0
        autoencoder_loss : bool, optional
            [description], by default False
        reconstruction_validation : array, optional
            validation X data for reconstruction loss, by default None
        loss_report_frequency : int, optional
            how many times per epoch to report loss, by default 1
        n_training_epochs : int, optional
            number of epochs to train for, by default 1
        global_correlation_loss_weight : float, optional
            Whether to additionally train on correlation of global pairwise relationships (>0), by default 0
        torch_fit_kwargs : dict, optional
            additional arguments for model.fit (like callbacks), by default {}
        """
        super().__init__(**kwargs)

        # add to network
        self.dims = dims  # if this is an image, we should reshape for network
        self.encoder = encoder  # neural network used for embedding
        self.decoder = decoder  # neural network used for decoding
        self.parametric_embedding = (
            parametric_embedding  # nonparametric vs parametric embedding
        )
        self.parametric_reconstruction = parametric_reconstruction
        self.parametric_reconstruction_loss_cls = parametric_reconstruction_loss_cls
        self.parametric_reconstruction_loss_weight = (
            parametric_reconstruction_loss_weight
        )
        self.autoencoder_loss = autoencoder_loss
        self.batch_size = batch_size
        self.loss_report_frequency = (
            loss_report_frequency  # how many times per epoch to report loss in torch
        )
        self.global_correlation_loss_weight = global_correlation_loss_weight

        self.reconstruction_validation = (
            reconstruction_validation  # holdout data for reconstruction acc
        )
        self.torch_fit_kwargs = torch_fit_kwargs  # arguments for model.fit
        self.parametric_model = None

        # how many epochs to train for (different than n_epochs which is specific to each sample)
        self.n_training_epochs = n_training_epochs
        # set optimizer
        if optimizer is None:
            if parametric_embedding:
                # Adam is better for parametric_embedding
                self.optimizer_class = th.optim.Adam
                self.lr = 1e-3
            else:
                # Larger learning rate can be used for embedding
                self.optimizer_class = th.optim.Adam
                self.lr = 1e-1
        else:
            self.optimizer = optimizer
        if parametric_reconstruction and not parametric_embedding:
            warn(
                "Parametric decoding is not implemented with nonparametric \
            embedding. Turning off parametric decoding"
            )
            self.parametric_reconstruction = False

        if self.encoder is not None:
            if encoder.outputs[0].shape[-1] != self.n_components:
                raise ValueError(
                    (
                        "Dimensionality of embedder network output ({}) does"
                        "not match n_components ({})".format(
                            encoder.outputs[0].shape[-1], self.n_components
                        )
                    )
                )

    def fit(self, X, y=None, precomputed_distances=None):
        if self.metric == "precomputed":
            if precomputed_distances is None:
                raise ValueError(
                    "Precomputed distances must be supplied if metric \
                    is precomputed."
                )
            # prepare X for training the network
            self._X = X
            # geneate the graph on precomputed distances
            return super().fit(precomputed_distances, y)
        else:
            return super().fit(X, y)

    def fit_transform(self, X, y=None, precomputed_distances=None):

        if self.metric == "precomputed":
            if precomputed_distances is None:
                raise ValueError(
                    "Precomputed distances must be supplied if metric \
                    is precomputed."
                )
            # prepare X for training the network
            self._X = X
            # geneate the graph on precomputed distances
            return super().fit_transform(precomputed_distances, y)
        else:
            return super().fit_transform(X, y)

    def transform(self, X):
        """Transform X into the existing embedded space and return that
        transformed output.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            New data to be transformed.
        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the new data in low-dimensional space.
        """
        if self.parametric_embedding:
            return self.encoder.predict(
                np.asanyarray(X), batch_size=self.batch_size, verbose=self.verbose
            )
        else:
            warn(
                "Embedding new data is not supported by ParametricUMAP. \
                Using original embedder."
            )
            return super().transform(X)

    def inverse_transform(self, X):
        """ "Transform X in the existing embedded space back into the input
        data space and return that transformed output.
        Parameters
        ----------
        X : array, shape (n_samples, n_components)
            New points to be inverse transformed.
        Returns
        -------
        X_new : array, shape (n_samples, n_features)
            Generated data points new data in data space.
        """
        if self.parametric_reconstruction:
            return self.decoder.predict(
                np.asanyarray(X), batch_size=self.batch_size, verbose=self.verbose
            )
        else:
            return super().inverse_transform(X)

    def _fit_embed_data(self, X, n_epochs, init, random_state):

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

        # get dataset of edges
        (
            edge_dataset,
            self.batch_size,
            n_edges,
            head,
            tail,
            self.edge_weight,
        ) = construct_edge_dataset(
            X,
            self.graph_,
            self.n_epochs,
            self.batch_size,
            self.parametric_embedding,
            self.parametric_reconstruction,
            self.global_correlation_loss_weight,
        )
        self.head = th.unsqueeze(head.astype(np.int64), 0)
        self.tail = th.unsqueeze(tail.astype(np.int64), 0)

        if self.parametric_embedding:
            init_embedding = None
        else:
            init_embedding = init_embedding_from_graph(
                X,
                self.graph_,
                self.n_components,
                self.random_state,
                self.metric,
                self._metric_kwds,
                init="spectral",
            )

        # create encoder and decoder model
        n_data = len(X)

        umap_params = {
            "batch_size": self.batch_size,
            "negative_sample_rate": self.negative_sample_rate,
            "a": self._a,
            "b": self._b,
            "edge_weight": self.edge_weight,
            "parametric_embedding": self.parametric_embedding,
        }

        self.parametric_model = GradientClippedModel(
            self.n_components,
            self.dims,
            n_data,
            self.parametric_embedding,
            self.parametric_reconstruction,
            init_embedding,
            umap_params,
        )

        # report every loss_report_frequency subdivision of an epochs
        if self.parametric_embedding:
            steps_per_epoch = int(
                n_edges / self.batch_size / self.loss_report_frequency
            )
        else:
            # all edges are trained simultaneously with nonparametric, so this is arbitrary
            steps_per_epoch = 100

        # Validation dataset for reconstruction
        if (
            self.parametric_reconstruction
            and self.reconstruction_validation is not None
        ):

            # reshape data for network
            if len(self.dims) > 1:
                self.reconstruction_validation = np.reshape(
                    self.reconstruction_validation,
                    [len(self.reconstruction_validation)] + list(self.dims),
                )

            validation_data = (
                (
                    self.reconstruction_validation,
                    th.zeros_like(self.reconstruction_validation),
                ),
                {"reconstruction": self.reconstruction_validation},
            )
        else:
            validation_data = None

        last_val_loss = np.inf

        # Training loop ported to Pytorch, edge_dataset is a generator
        self.parametric_model.train()
        for edge_index, edge_weight in edge_dataset:
            self.parametric_model.optimizer.zero_grad()

            x, y = edge_weight

            y_pred = self.parametric_model(x, training=True)  # Forward pass

            # Compute the loss value based on the given losses and weights
            loss = sum(
                loss_fn(y, y_pred) * weight
                for loss_fn, weight in zip(
                    self.parametric_model.losses.values(),
                    self.parametric_model.loss_weights.values(),
                )
            )

            loss.backward()

            th.nn.utils.clip_grad_norm_(self.parameters(), 4.0)
            self.parametric_model.optimizer.step()

            # max epochs
            if (
                self.parametric_model.optimizer.n_iter
                >= self.n_training_epochs * steps_per_epoch
            ):
                break

            # Early stopping on validation data if provided
            if validation_data is not None:
                # Call reconstruct and compute reconstruction loss
                for val_idx, (val_x, val_y) in enumerate(validation_data):
                    rec_pred = self.parametric_model.reconstruct(val_x)
                    rec_loss = self.parametric_model.losses["reconstruction"](
                        val_y, rec_pred
                    )
                    if val_idx == 0:
                        rec_loss_sum = rec_loss
                    else:
                        rec_loss_sum += rec_loss
                    current = rec_loss_sum / (val_idx + 1)
                if last_val_loss < current:
                    break

        y_pred = self(x, training=True)  # Forward pass

        # get the final embedding
        if self.parametric_embedding:
            embedding = self.parametric_model.encoder(X)
        else:
            # embedding = self.parametric_model.encoder.trainable_variables[0].numpy()
            embedding = self.parametric_model.encoder.weight[-1].cpu().detach().numpy()

        return embedding, {}

    def save(self, save_location, verbose=True):

        # save parametric_model
        if self.parametric_model is not None:
            parametric_model_output = os.path.join(save_location, "parametric_model")
            th.save(self.parametric_model.state_dict(), parametric_model_output)
            if verbose:
                print("Torch full model saved to {}".format(parametric_model_output))


def get_graph_elements(graph_, n_epochs):
    """
    gets elements of graphs, weights, and number of epochs per edge

    Parameters
    ----------
    graph_ : scipy.sparse.csr.csr_matrix
        umap graph of probabilities
    n_epochs : int
        maximum number of epochs per edge

    Returns
    -------
    graph scipy.sparse.csr.csr_matrix
        umap graph
    epochs_per_sample np.array
        number of epochs to train each sample for
    head np.array
        edge head
    tail np.array
        edge tail
    weight np.array
        edge weight
    n_vertices int
        number of verticies in graph
    """
    ### should we remove redundancies () here??
    # graph_ = remove_redundant_edges(graph_)

    graph = graph_.tocoo()
    # eliminate duplicate entries by summing them together
    graph.sum_duplicates()
    # number of vertices in dataset
    n_vertices = graph.shape[1]
    # get the number of epochs based on the size of the dataset
    if n_epochs is None:
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200
    # remove elements with very low probability
    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()
    # get epochs per sample based upon edge probability
    epochs_per_sample = n_epochs * graph.data

    head = graph.row
    tail = graph.col
    weight = graph.data

    return graph, epochs_per_sample, head, tail, weight, n_vertices


def init_embedding_from_graph(
    _raw_data, graph, n_components, random_state, metric, _metric_kwds, init="spectral"
):
    """Initialize embedding using graph. This is for direct embeddings.

    Parameters
    ----------
    init : str, optional
        Type of initialization to use. Either random, or spectral, by default "spectral"

    Returns
    -------
    embedding : np.array
        the initialized embedding
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


def convert_distance_to_probability(distances, a=1.0, b=1.0):
    """
     convert distance representation into probability,
        as a function of a, b params

    Parameters
    ----------
    distances : array
        euclidean distance between two points in embedding
    a : float, optional
        parameter based on min_dist, by default 1.0
    b : float, optional
        parameter based on min_dist, by default 1.0

    Returns
    -------
    float
        probability in embedding space
    """
    return 1.0 / (1.0 + a * distances ** (2 * b))


def compute_cross_entropy(
    probabilities_graph, probabilities_distance, EPS=1e-4, repulsion_strength=1.0
):
    """
    Compute cross entropy between low and high probability

    Parameters
    ----------
    probabilities_graph : array
        high dimensional probabilities
    probabilities_distance : array
        low dimensional probabilities
    EPS : float, optional
        offset to to ensure log is taken of a positive number, by default 1e-4
    repulsion_strength : float, optional
        strength of repulsion between negative samples, by default 1.0

    Returns
    -------
    attraction_term: th.float32
        attraction term for cross entropy loss
    repellant_term: th.float32
        repellant term for cross entropy loss
    cross_entropy: th.float32
        cross entropy umap loss

    """
    # cross entropy
    attraction_term = -probabilities_graph * th.log(
        th.clip(probabilities_distance, EPS, 1.0)
    )
    repellant_term = (
        -(1.0 - probabilities_graph)
        * th.log(th.clip(1.0 - probabilities_distance, EPS, 1.0))
        * repulsion_strength
    )

    # balance the expected losses between atrraction and repel
    CE = attraction_term + repellant_term
    return attraction_term, repellant_term, CE


def umap_loss(
    batch_size,
    negative_sample_rate,
    _a,
    _b,
    edge_weights,
    parametric_embedding,
    repulsion_strength=1.0,
):
    """
    Generate a torch-ccompatible loss function for UMAP loss

    Parameters
    ----------
    batch_size : int
        size of mini-batches
    negative_sample_rate : int
        number of negative samples per positive samples to train on
    _a : float
        distance parameter in embedding space
    _b : float float
        distance parameter in embedding space
    edge_weights : array
        weights of all edges from sparse UMAP graph
    parametric_embedding : bool
        whether the embeddding is parametric or nonparametric
    repulsion_strength : float, optional
        strength of repulsion vs attraction for cross-entropy, by default 1.0

    Returns
    -------
    loss : function
        loss function that takes in a placeholder (0) and the output of the torch network
    """

    if not parametric_embedding:
        # multiply loss by weights for nonparametric
        weights_tiled = np.tile(edge_weights, negative_sample_rate + 1)

    def loss(placeholder_y, embed_to_from):
        # split out to/from
        embedding_to, embedding_from = th.split(
            embed_to_from, split_size_or_sections=2, dim=1
        )

        # get negative samples
        embedding_neg_to = embedding_to.repeat(negative_sample_rate)
        repeat_neg = embedding_from.repeat(negative_sample_rate)
        embedding_neg_from = repeat_neg[th.randperm(repeat_neg.shape[0])]

        #  distances between samples (and negative samples)
        distance_embedding = th.cat(
            [
                th.norm(embedding_to - embedding_from, dim=1),
                th.norm(embedding_neg_to - embedding_neg_from, dim=1),
            ],
            dim=0,
        )

        # convert probabilities to distances
        probabilities_distance = convert_distance_to_probability(
            distance_embedding, _a, _b
        )

        # set true probabilities based on negative sampling
        probabilities_graph = th.cat(
            [th.ones(batch_size), th.zeros(batch_size * negative_sample_rate)], dim=0
        )

        # compute cross entropy
        (attraction_loss, repellant_loss, ce_loss) = compute_cross_entropy(
            probabilities_graph,
            probabilities_distance,
            repulsion_strength=repulsion_strength,
        )

        if not parametric_embedding:
            ce_loss = ce_loss * weights_tiled

        return th.mean(ce_loss)

    return loss


def distance_loss_corr(x, z_x):
    """Loss based on the distance between elements in a batch"""

    # flatten data
    x = th.flatten(x)
    z_x = th.flatten(z_x)

    ## z score data
    def z_score(x):
        return (x - th.mean(x)) / th.std(x)

    x = z_score(x)
    z_x = z_score(z_x)

    # clip distances to 10 standard deviations for stability
    x = th.clip(x, -10, 10)
    z_x = th.clip(z_x, -10, 10)

    dx = th.norm(x[1:] - x[:-1], dim=1, p=2)
    dz = th.norm(z_x[1:] - z_x[:-1], dim=1, p=2)

    # jitter dz to prevent mode collapse
    dz = dz + th.rand(dz.shape) * 1e-10

    # compute correlation
    corr_d = th.squeeze(th.corrcoef(x=th.unsqueeze(dx, -1), y=th.unsqueeze(dz, -1)))
    if th.isnan(corr_d):
        raise ValueError("NaN values found in correlation loss.")

    return -corr_d


def construct_edge_dataset(
    X,
    graph_,
    n_epochs,
    batch_size,
    parametric_embedding,
    parametric_reconstruction,
    global_correlation_loss_weight,
):
    """
    Construct a Dataset of edges, sampled by edge weight.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        New data to be transformed.
    graph_ : scipy.sparse.csr.csr_matrix
        Generated UMAP graph
    n_epochs : int
        # of epochs to train each edge
    batch_size : int
        batch size
    parametric_embedding : bool
        Whether the embedder is parametric or non-parametric
    parametric_reconstruction : bool
        Whether the decoder is parametric or non-parametric
    """

    # if X is > 512Mb in size, we need to use a different, slower method for
    #    batching data.
    True if X.nbytes * 1e-9 > 0.5 else False

    def gather_X(edge_to, edge_from):
        return X[edge_to], X[edge_from]

    def get_outputs(edge_to_batch, edge_from_batch):
        outputs = {"umap": th.IntTensor(0).repeat(batch_size)}
        if global_correlation_loss_weight > 0:
            outputs["global_correlation"] = edge_to_batch
        if parametric_reconstruction:
            # add reconstruction to iterator output
            outputs["reconstruction"] = edge_to_batch
        return (edge_to_batch, edge_from_batch), outputs

    def make_sham_generator():
        """
        The sham generator is a placeholder when all data is already intrinsic to
        the model, but kertorchas wants some input data. Used for non-parametric
        embedding.
        """

        def sham_generator():
            while True:
                yield th.zeros(1, dtype=th.int32), th.zeros(1, dtype=th.int32)

        return sham_generator

    # get data from graph
    graph, epochs_per_sample, head, tail, weight, n_vertices = get_graph_elements(
        graph_, n_epochs
    )

    # number of elements per batch for embedding
    if batch_size is None:
        # batch size can be larger if its just over embeddings
        if parametric_embedding:
            batch_size = np.min([n_vertices, 1000])
        else:
            batch_size = len(head)

    edges_to_exp, edges_from_exp = (
        np.repeat(head, epochs_per_sample.astype("int")),
        np.repeat(tail, epochs_per_sample.astype("int")),
    )

    # shuffle edges
    shuffle_mask = np.random.permutation(range(len(edges_to_exp)))
    edges_to_exp = edges_to_exp[shuffle_mask].astype(np.int64)
    edges_from_exp = edges_from_exp[shuffle_mask].astype(np.int64)

    # create edge iterator
    if parametric_embedding:
        edge_dataset = tf.data.Dataset.from_tensor_slices(
            (edges_to_exp, edges_from_exp)
        )
        edge_dataset = edge_dataset.repeat()
        edge_dataset = edge_dataset.shuffle(10000)
        edge_dataset = edge_dataset.batch(batch_size, drop_remainder=True)
        edge_dataset = edge_dataset.map(
            gather_X, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        edge_dataset = edge_dataset.map(
            get_outputs, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        edge_dataset = edge_dataset.prefetch(10)
    else:
        # nonparametric embedding uses a sham dataset
        gen = make_sham_generator()
        edge_dataset = tf.data.Dataset.from_generator(
            gen,
            (th.int32, th.int32),
            output_shapes=(tf.TensorShape(1), tf.TensorShape((1,))),
        )
    return edge_dataset, batch_size, len(edges_to_exp), head, tail, weight


def load_ParametricUMAP(save_location, verbose=True):
    """
    Load a parametric UMAP model consisting of a umap-learn UMAP object
    and corresponding torch models.

    Parameters
    ----------
    save_location : str
        the folder that the model was saved in
    verbose : bool, optional
        Whether to print the loading steps, by default True

    Returns
    -------
    parametric_umap.ParametricUMAP
        Parametric UMAP objects
    """

    ## Loads a ParametricUMAP model and its related torch models

    model_output = os.path.join(save_location, "model.pkl")
    model = pickle.load((open(model_output, "rb")))
    if verbose:
        print("Pickle of ParametricUMAP model loaded from {}".format(model_output))

    # Work around optimizer not pickling anymore (since tf 2.4)
    class_name = model._optimizer_dict["name"]
    OptimizerClass = getattr(th.optim.Optimizer, class_name)
    model.optimizer = OptimizerClass.from_config(model._optimizer_dict)

    # load encoder
    encoder_output = os.path.join(save_location, "encoder")
    if os.path.exists(encoder_output):
        model.encoder = th.load(encoder_output)
        if verbose:
            print("Torch encoder model loaded from {}".format(encoder_output))

    # save decoder
    decoder_output = os.path.join(save_location, "decoder")
    if os.path.exists(decoder_output):
        model.decoder = th.load(decoder_output)
        print("Torch decoder model loaded from {}".format(decoder_output))

    # get the custom loss function
    umap_loss_fn = umap_loss(
        model.batch_size,
        model.negative_sample_rate,
        model._a,
        model._b,
        model.edge_weight,
        model.parametric_embedding,
    )

    # save parametric_model
    parametric_model_output = os.path.join(save_location, "parametric_model")
    if os.path.exists(parametric_model_output):
        model.parametric_model = th.load(
            parametric_model_output, map_location={"loss": umap_loss_fn}
        )
        print("Torch full model loaded from {}".format(parametric_model_output))

    return model


class GradientClippedModel(th.nn.Module):
    """
    We need to define a custom torch model here for gradient clipping,
    to stabilize training.
    """

    def __init__(
        self,
        encoder,
        decoder,
        n_components,
        dims,
        n_data,
        parametric_embedding,
        parametric_reconstruction,
        init_embedding,
        umap_params,
    ):
        super().__init__()

        losses = {}
        loss_weights = {}

        umap_loss_fn = umap_loss(**umap_params)
        losses["umap"] = umap_loss_fn
        loss_weights["umap"] = 1.0

        if self.global_correlation_loss_weight > 0:
            losses["global_correlation"] = distance_loss_corr
            loss_weights["global_correlation"] = self.global_correlation_loss_weight

        if self.parametric_reconstruction:
            losses["reconstruction"] = self.parametric_reconstruction_loss_fcn
            loss_weights["reconstruction"] = self.parametric_reconstruction_loss_weight

        self.losses = losses
        self.loss_weights = loss_weights

        # Define networks
        flat_shape = th.prod(dims)

        if parametric_embedding:
            if encoder is None:
                self.encoder = th.nn.Sequential(
                    th.nn.Flatten(),
                    th.nn.Linear(in_features=flat_shape.item(), out_features=100),
                    th.nn.ReLU(),
                    th.nn.Linear(in_features=100, out_features=100),
                    th.nn.ReLU(),
                    th.nn.Linear(in_features=100, out_features=100),
                    th.nn.ReLU(),
                    th.nn.Linear(in_features=100, out_features=n_components),
                )
            else:
                self.encoder = encoder
        else:
            embedding_layer = th.nn.Embedding(n_data, n_components)
            self.encoder = th.nn.Sequential(embedding_layer)

        if decoder is None:
            if parametric_reconstruction:
                self.decoder = th.nn.Sequential(
                    th.nn.Flatten(),
                    th.nn.Linear(in_features=n_components, out_features=100),
                    th.nn.ReLU(),
                    th.nn.Linear(in_features=100, out_features=100),
                    th.nn.ReLU(),
                    th.nn.Linear(in_features=100, out_features=100),
                    th.nn.ReLU(),
                    th.nn.Linear(in_features=100, out_features=np.product(dims)),
                )

        # Define an optimizer and a loss function for training
        self.optimizer = th.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        outputs = {}

        if self.parametric_embedding:
            # parametric embedding
            embedding_to = self.encoder(x)
            embedding_from = self.encoder(x)

            if self.parametric_reconstruction:
                # parametric reconstruction
                if self.autoencoder_loss:
                    embedding_to_recon = self.decoder(embedding_to)
                else:
                    with th.no_grad():
                        embedding_to_recon = self.decoder(embedding_to)

                outputs["reconstruction"] = embedding_to_recon

        else:
            # this is the sham input (its just a 0) to make torch think there is input data
            batch_sample = th.tensor([0])

            # gather all of the edges (so torch model is happy)
            to_x = th.squeeze(self.head[batch_sample[0]])
            from_x = th.squeeze(self.tail[batch_sample[0]])

            # grab relevant embeddings
            embedding_to = self.encoder(to_x)[:, -1, :]
            embedding_from = self.encoder(from_x)[:, -1, :]

            [batch_sample]

        # concatenate to/from projections for loss computation
        embedding_to_from = th.cat([embedding_to, embedding_from], axis=1)
        outputs["umap"] = embedding_to_from

        if self.global_correlation_loss_weight > 0:
            outputs["global_correlation"] = embedding_to

        return outputs
