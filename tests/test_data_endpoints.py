"""

    Test the DATA API endpoints

"""
import os

from rlhfblender.data_models.global_models import (
    Dataset,
    Environment,
    Experiment,
    Project,
    TrackingItem,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Set DB environment variable
os.environ["RLHFBLENDER_DB_HOST"] = "sqlite:///test_api.db"

from rlhfblender.app import app
from rlhfblender.routes.data import *

client = TestClient(app)


def test_available_frameworks():
    response = client.get("/data/get_available_frameworks")
    assert response.status_code == 200


def test_get_algorithms():
    response = client.get("/data/get_algorithms")
    assert response.status_code == 200
