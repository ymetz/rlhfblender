"""

    Test the DATA API endpoints

"""
import os
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.data_models.global_models import Environment, Project, Experiment, Dataset, TrackingItem

# Set DB environment variable
os.environ["RLHFBLENDER_DB_HOST"] = "sqlite:///test_api.db"

from backend.app.app import app
from backend.routes.data import *

client = TestClient(app)

def test_available_frameworks():
    response = client.get("/data/get_available_frameworks")
    assert response.status_code == 200

def test_get_algorithms():
    response = client.get("/data/get_algorithms")
    assert response.status_code == 200

