import inspect
import json
import os

import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding

from rlhfblender.projections.parametric_umap import ParametricUMAP, load_ParametricUMAP


class ProjectionHandler:
    """
    Embeddding Visualization Helper
    """

    def __init__(self, projection_method: str = "UMAP", projection_props: dict = None, **kwargs):

        self.in_fitting = None
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

    def fit(self, data: np.array, sequence_length: int, step_range=None, episode_indices=None, actions=None, suffix=""):
        """
        Fit the embedding method to the data.
        :param data:
        :param sequence_length:
        :param step_range:
        :param episode_indices:
        :param actions:
        :return:
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
            # data = data.reshape(data.shape[0], np.prod(data.shape[1:]))
        self.in_fitting = True
        if episode_indices is not None:
            data = np.concatenate((data, np.expand_dims(episode_indices, -1)), axis=1)
        if actions is not None:
            pass
            # data = np.concatenate((data, np.expand_dims(actions, -1)), axis=1)
        if self.save_embedding_path != "" and (self.embedding_method.__class__.__name__ == "ParametricUMAP"):
            # If a pre-trained, we do not need to fit the model again, just call the transform
            self.embedding_method.embedding_ = self.embedding_method.transform(np.squeeze(data))
        else:
            # Fit the embedding method to the data
            return self.embedding_method.fit_transform(data)

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

        return self.embedding_method.embedding_

    def get_state(self):
        """
        Return the current state of the embedding method.
        :return:
        """
        return self.embedding_method.embedding_

    def is_fitting(self):
        """
        Return whether the embedding method is currently fitting.
        :return:
        """
        return self.in_fitting

    @staticmethod
    def get_available_embedding_methods() -> list[str]:
        """
        Return a list of all available embedding methods.
        :return:
        """
        return ["UMAP", "ParametricUMAP", "ParametricAngleUMAP", "t-SNE", "LLE"]

    @staticmethod
    def get_embedding_method_params(embedding_method: str) -> dict:
        """
        Return the parameters of the embedding method.
        :param embedding_method:
        :return:
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
        :param x:
        :return:
        """
        try:
            json.dumps(x)
            return True
        except (TypeError, OverflowError):
            return False
