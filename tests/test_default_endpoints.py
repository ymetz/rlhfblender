"""

    Test the API endpoints

"""

import json
import os

from backend.app.data_models.global_models import (
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

from backend.app.app import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200


def test_get_all():
    response = client.get("/get_all?model_name=Environment")
    assert response.status_code == 200
    assert response.json() == []


def test_create_environment():
    test_environment = Environment(
        name="test_environment", description="test_description"
    )
    response = client.post(
        "/create?model_name=Environment", json=test_environment.dict()
    )
    assert response.status_code == 200


def test_create_project():
    test_project = Project(name="test_project", description="test_description")
    response = client.post("/create?model_name=Project", json=test_project.dict())
    assert response.status_code == 200


def test_create_experiment():
    test_experiment = Experiment(name="test_experiment", description="test_description")
    response = client.post("/create?model_name=Experiment", json=test_experiment.dict())
    assert response.status_code == 200


def test_create_dataset():
    test_dataset = Dataset(name="test_dataset", description="test_description")
    response = client.post("/create?model_name=Dataset", json=test_dataset.dict())
    assert response.status_code == 200


def test_create_tracking_item():
    test_tracking_item = TrackingItem(
        name="test_tracking_item", description="test_description"
    )
    response = client.post(
        "/create?model_name=TrackingItem", json=test_tracking_item.dict()
    )
    assert response.status_code == 200


def test_get_all():

    test_environment = Environment(
        name="test_environment2", description="test_description2"
    )
    response = client.post(
        "/create?model_name=Environment", json=test_environment.dict()
    )
    assert response.status_code == 200

    response = client.get("/get_all?model_name=Environment")
    assert response.status_code == 200
    assert response.json() == [
        Environment(name="test_environment", description="test_description").dict(),
        Environment(name="test_environment2", description="test_description2").dict(),
    ]


def test_get_data_by_id():
    response = client.get("/get_data_by_id?model_name=Environment&item_id=0")
    assert response.status_code == 200
    assert (
        response.json()
        == Environment(name="test_environment", description="test_description").dict()
    )


def test_update_data():
    test_environment = Environment(
        name="test_environment_updated", description="test_description_updated"
    )
    response = client.post(
        "/update_data?model_name=Environment&item_id=0", json=test_environment.dict()
    )
    assert response.status_code == 200

    response = client.get("/get_data_by_id?model_name=Environment&item_id=0")
    assert response.status_code == 200
    assert (
        response.json()
        == Environment(
            name="test_environment_updated", description="test_description_updated"
        ).dict()
    )


def test_delete_data():
    response = client.delete("/delete_data?model_name=Environment&item_id=0")
    assert response.status_code == 200

    response = client.get("/get_all?model_name=Environment")
    assert response.status_code == 200
    assert response.json() == [
        Environment(name="test_environment2", description="test_description2").dict()
    ]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"message": "OK"}


def test_ui_configs():
    response = client.get("/ui_configs")
    assert response.status_code == 200
    assert response.json() == []


def test_save_ui_config():
    test_ui_config = {"test": "test"}
    response = client.post(
        "/save_ui_config?ui_config_name=test_ui_config", json=test_ui_config
    )
    assert response.status_code == 200

    response = client.get("/ui_configs")
    assert response.status_code == 200
    assert response.json() == ["test_ui_config"]


def test_retreive_logs():
    response = client.get("/retrieve_logs")
    assert response.status_code == 200
    assert response.json() == []


def test_retreive_demos():
    response = client.get("/retrieve_demos")
    assert response.status_code == 200
    assert response.json() == []


def test_retreive_feature_feedback():
    response = client.get("/retreive_feature_feedback")
    assert response.status_code == 200
    assert response.json() == []
