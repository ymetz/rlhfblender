import os

os.environ["RLHFBLENDER_DB_HOST"] = "sqlite:///test_api.db"

import pytest
from fastapi.testclient import TestClient

from rlhfblender.app import app
from rlhfblender.generate_data import generate_data


class DataGerationTests:

    # wait for app startup
    @pytest.fixture(scope="session")
    def client():
        # remove test db if it exists
        try:
            os.remove("test_api.db")
        except FileNotFoundError:
            pass

        with TestClient(app) as c:
            yield c

    @pytest.mark.dependency(depends=["test_register_env"])
    def test_generate_data_without_experiment(self):
        """
        Test generate_data without experiment, call async
        """
        benchmark_run = {
            "benchmark_type": "random",
            "n_episodes": 1,
            "env_id": "CartPole-v1",
            "benchmark_id": None,
            "force_overwrite": False,
        }
        generate_data([benchmark_run])
