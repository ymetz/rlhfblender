"""

Test the DATA API endpoints

"""

import os
os.environ["RLHFBLENDER_DB_HOST"] = "sqlite:///test_api.db"

from fastapi.testclient import TestClient

from rlhfblender.app import app

client = TestClient(app)


def test_available_frameworks():
    response = client.get("/data/get_available_frameworks")
    assert response.status_code == 200


def test_get_algorithms():
    response = client.get("/data/get_algorithms", params={"selected_framework": "StableBaselines3"})
    assert response.status_code == 200
