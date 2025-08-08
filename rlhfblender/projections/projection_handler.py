import inspect
import json
import os
import pickle

import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding

from rlhfblender.projections.parametric_umap import ParametricUMAP, load_ParametricUMAP


class ProjectionHandler:
    """
    Embedding Visualization Helper with support for joint projections
    """

    def __init__(self, projection_method: str = "UMAP", projection_props: dict = None, **kwargs):
        self.in_fitting = None
        self.is_fitted = False
        self.joint_projection_path = kwargs.get("joint_projection_path", None)
        self.projection_results = None  # Store projection results consistently
        self.state_model_path = None  # Path to inverse state projection model for joint obs-state projections

        # If a joint projection path is provided, load it
        if self.joint_projection_path:
            self.load_joint_projection(self.joint_projection_path)
        else:
            self.set_embedding_method(projection_method, projection_props)

        self.save_embedding = False
        self.save_embedding_path = None

    def set_embedding_props(self, projection_props: dict, **kwargs):
        if "save_embedding" in kwargs:
            self.save_embedding = kwargs["save_embedding"]
            kwargs.pop("save_embedding")
        if "save_embedding_path" in kwargs:
            # Overwrite the default settings with the stored model
            self.save_embedding_path = kwargs.pop("save_embedding_path")
            # If path is not empty, load the embedding from the path
            if self.save_embedding_path != "" and (
                self.embedding_method.__class__.__name__ == "ParametricUMAP"
                or self.embedding_method.__class__.__name__ == "ParametricAngleUMAP"
            ):
                self.embedding_method = load_ParametricUMAP(
                    os.path.join("data", "saved_embeddings", "parametric_embedding", self.save_embedding_path)
                )

    def set_embedding_method(self, embedding_method: str, projection_props: dict = None):
        projection_props = projection_props if projection_props else {}
        embedding_class = None

        if embedding_method == "UMAP":
            embedding_class = umap.UMAP
        elif embedding_method == "ParametricUMAP":
            embedding_class = ParametricUMAP
        elif embedding_method == "t-SNE":
            embedding_class = TSNE
        elif embedding_method == "LLE":
            embedding_class = LocallyLinearEmbedding
        elif embedding_method == "PCA":
            embedding_class = PCA
        else:
            raise ValueError(f"Unknown embedding method: {embedding_method}")

        # Get valid parameters by inspecting the class signature
        signature = inspect.signature(embedding_class.__init__)
        valid_params = set(signature.parameters.keys()) - {"self"}

        # Filter projection_props to only include valid parameters
        filtered_props = {k: v for k, v in projection_props.items() if k in valid_params}

        # Set default n_components if not specified and it's a valid parameter
        if "n_components" in valid_params and "n_components" not in filtered_props:
            filtered_props["n_components"] = 2

        # Initialize the embedding method with the filtered properties
        self.embedding_method = embedding_class(**filtered_props)

    def load_joint_projection(self, joint_projection_path: str):
        """
        Load a pre-fitted joint projection.

        Args:
            joint_projection_path: Path to the joint projection metadata file or handler pickle file
        """
        print(f"Loading joint projection from: {joint_projection_path}")

        if joint_projection_path.endswith(".json"):
            # Load from metadata file
            with open(joint_projection_path, "r") as f:
                metadata = json.load(f)

            # Handle joint observation-state projection format
            if "state_model_path" in metadata:
                # Load the underlying observation projection first
                obs_proj_metadata_path = metadata["observation_projection_metadata"]
                if not os.path.exists(obs_proj_metadata_path):
                    raise FileNotFoundError(f"Observation projection metadata not found: {obs_proj_metadata_path}")

                # Recursively load the observation projection
                self.load_joint_projection(obs_proj_metadata_path)

                # Store reference to state model for later use if needed
                self.state_model_path = metadata["state_model_path"]
                return

            # Handle old observation projection format
            elif "handler_path" in metadata:
                handler_path = metadata["handler_path"]
                if not os.path.exists(handler_path):
                    raise FileNotFoundError(f"Joint projection handler not found: {handler_path}")

                # Load the fitted handler data
                with open(handler_path, "rb") as f:
                    save_data = pickle.load(f)

                # Handle both old format (direct handler) and new format (save_data dict)
                if isinstance(save_data, dict):
                    # New format
                    self.embedding_method = save_data["embedding_method"]
                    self.projection_results = save_data["projection_results"]
                    self.is_fitted = save_data.get("is_fitted", True)
                else:
                    # Old format - save_data is the handler itself
                    fitted_handler = save_data
                    self.embedding_method = fitted_handler.embedding_method
                    self.projection_results = getattr(fitted_handler, "projection_results", None)
                    # Try to get results from method-specific storage if not available
                    if self.projection_results is None and hasattr(fitted_handler.embedding_method, "embedding_"):
                        self.projection_results = fitted_handler.embedding_method.embedding_
                    self.is_fitted = True

                print(f"Loaded observation projection: {metadata['projection_method']}")
                print(f"Experiment: {metadata['experiment_name']}")
                print(f"Checkpoints: {metadata['checkpoints']}")
                return

            else:
                raise ValueError(
                    "Unsupported joint projection format - missing required metadata keys (need either 'state_model_path' or 'handler_path')"
                )

    def fit(self, data: np.ndarray, sequence_length: int, step_range=None, episode_indices=None, actions=None, suffix=""):
        """
        Fit the embedding method to the data.

        If a joint projection has been loaded, this will transform the data using the pre-fitted model
        instead of fitting a new one.
        """
        if step_range:
            data = data[step_range[0] : step_range[1]]

        if len(data.shape) <= 2:
            # stack multiple sequence steps before t-SNE
            data = np.vstack(
                np.split(
                    data,
                    np.array([[i, i + sequence_length] for i in range(data.shape[0] - data.shape[1])]).reshape(-1),
                )
            ).reshape(-1, data.shape[1] * sequence_length)

        # If we have high dimensional data, first apply PCA before the UMAP embedding
        if np.prod(data.shape[1:]) > 100 and self.embedding_method.__class__.__name__ != "PCA":
            pca = PCA(n_components=50)
            data = pca.fit_transform(data.reshape(data.shape[0], np.prod(data.shape[1:])))

        self.in_fitting = True

        if episode_indices is not None:
            data = np.concatenate((data, np.expand_dims(episode_indices, -1)), axis=1)

        if actions is not None:
            pass
            # data = np.concatenate((data, np.expand_dims(actions, -1)), axis=1)

        # Check if we're using a pre-fitted joint projection
        if self.is_fitted:
            print("Using pre-fitted joint projection for transformation...")

            # For pre-fitted models, use transform instead of fit_transform
            if hasattr(self.embedding_method, "transform"):
                projected_data = self.embedding_method.transform(np.squeeze(data))
            else:
                # Some methods don't have transform, need to use a different approach
                print("Warning: Pre-fitted model doesn't support transform. Results may be inconsistent.")
                projected_data = self.embedding_method.fit_transform(data)

            # Store the projection result consistently
            self.projection_results = projected_data

        else:
            # Normal fitting process
            if self.save_embedding_path != "" and (self.embedding_method.__class__.__name__ == "ParametricUMAP"):
                # If a pre-trained model exists, use transform instead of fit
                projected_data = self.embedding_method.transform(np.squeeze(data))
            else:
                # Fit the embedding method to the data
                projected_data = self.embedding_method.fit_transform(data)

            # Store results consistently across all methods
            self.projection_results = projected_data

            # Also store in embedding_ if the method supports it (for backwards compatibility)
            if hasattr(self.embedding_method, "embedding_"):
                self.embedding_method.embedding_ = projected_data

            if self.save_embedding and (
                self.embedding_method.__class__.__name__ == "ParametricUMAP"
                or self.embedding_method.__class__.__name__ == "ParametricAngleUMAP"
            ):
                self.embedding_method.save(
                    os.path.join(
                        "data", "saved_embeddings", "parametric_embedding", "overwrite_" + self.save_embedding_path + suffix
                    )
                )

        self.in_fitting = False
        return self.projection_results

    def transform(self, data: np.array):
        """
        Transform new data using the fitted embedding method.

        This is useful when you have a fitted joint projection and want to project
        new data into the same space.
        """
        if not self.is_fitted and not hasattr(self.embedding_method, "transform"):
            raise ValueError("Embedding method is not fitted or doesn't support transformation")

        if hasattr(self.embedding_method, "transform"):
            return self.embedding_method.transform(data)
        else:
            raise ValueError(f"Transform not supported for {self.embedding_method.__class__.__name__}")

    def get_state(self):
        """
        Return the current state of the embedding method.
        """
        # Try to get results from our consistent storage first
        if self.projection_results is not None:
            return self.projection_results

        # Fall back to method-specific storage for backwards compatibility
        if hasattr(self.embedding_method, "embedding_"):
            return self.embedding_method.embedding_

        # If neither exists, return None
        return None

    def is_fitting(self):
        """
        Return whether the embedding method is currently fitting.
        """
        return self.in_fitting

    @staticmethod
    def load_joint_projection_results(joint_projection_path: str) -> dict:
        """
        Load the results of a joint projection (coordinates for each checkpoint).

        Args:
            joint_projection_path: Path to the joint projection metadata file

        Returns:
            Dictionary with checkpoint coordinates and metadata
        """
        if joint_projection_path.endswith(".json"):
            with open(joint_projection_path, "r") as f:
                metadata = json.load(f)

            results_path = metadata["results_path"]
            if not os.path.exists(results_path):
                raise FileNotFoundError(f"Joint projection results not found: {results_path}")

            # Load the projection results
            results_data = np.load(results_path)

            # Extract per-checkpoint coordinates
            checkpoint_coords = {}
            for checkpoint in metadata["checkpoints"]:
                key = f"checkpoint_{checkpoint}"
                if key in results_data:
                    checkpoint_coords[checkpoint] = results_data[key]

            return {"metadata": metadata, "checkpoint_coordinates": checkpoint_coords, "checkpoints": metadata["checkpoints"]}
        else:
            raise ValueError("Joint projection path must be a .json metadata file")

    @staticmethod
    def get_available_embedding_methods() -> list[str]:
        """
        Return a list of all available embedding methods.
        """
        return ["UMAP", "ParametricUMAP", "ParametricAngleUMAP", "t-SNE", "LLE"]

    @staticmethod
    def get_embedding_method_params(embedding_method: str) -> dict:
        """
        Return the parameters of the embedding method.
        """
        param_dict = {}
        if embedding_method == "UMAP":
            module = umap.UMAP
        elif embedding_method == "ParametricUMAP":
            module = ParametricUMAP
        elif embedding_method == "t-SNE":
            module = TSNE
        elif embedding_method == "LLE":
            module = LocallyLinearEmbedding
        elif embedding_method == "PCA":
            module = PCA
        else:
            raise ValueError(f"Unknown embedding method: {embedding_method}")
        for param in inspect.signature(module.__init__).parameters.values():
            if not param.default == inspect.Signature.empty and ProjectionHandler.is_jsonable(param.default):
                param_dict[param.name] = param.default
        return param_dict

    @staticmethod
    def is_jsonable(x) -> bool:
        """
        Check if a value is JSONable.
        """
        try:
            json.dumps(x)
            return True
        except (TypeError, OverflowError):
            return False
