"""
    Test the API endpoints.
"""

import json
import os
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from unittest import TestCase

from rlhfblender.app import app
from rlhfblender.data_models.global_models import (
    Dataset,
    Environment,
    Experiment,
    Project,
    TrackingItem,
)

# Set DB environment variable
os.environ['RLHFBLENDER_DB_HOST'] = 'sqlite:///test_api.db'

# wait for app startup
@pytest.fixture(scope='session')
def client():

    # remove test db if it exists
    try:
        os.remove('test_api.db')
    except FileNotFoundError:
        pass

    with TestClient(app) as c:
        yield c


def test_read_main(client):
    response = client.get('/')
    assert response.status_code == 200


def test_get_all(client):
    response = client.get('/get_all?model_name=environment')
    assert response.status_code == 200
    assert response.json() == []


def test_create_environment(client):
    test_environment = Environment(env_name='test_environment', description='test_description')
    response = client.post('/add_data', json={'model_name': 'environment', 'data': test_environment.model_dump()})
    assert response.status_code == 200


def test_create_project(client):
    test_project = Project(project_name='test_project', project_description='test_description')
    response = client.post('/add_data', json={'model_name': 'project', 'data': test_project.model_dump()})
    assert response.status_code == 200


def test_create_experiment(client):
    test_experiment = Experiment(exp_name='test_experiment', exp_comment='test_description')
    response = client.post('/add_data', json={'model_name': 'experiment', 'data': test_experiment.model_dump()})
    assert response.status_code == 200


def test_create_dataset(client):
    test_dataset = Dataset(dataset_name='test_dataset', dataset_description='test_description')
    #response = client.post('/add_data?model_name=dataset', json=test_dataset.model_dump())
    response = client.post('/add_data', json={'model_name': 'dataset', 'data': test_dataset.model_dump()})
    assert response.status_code == 200


def test_create_tracking_item(client):
    test_tracking_item = TrackingItem(tracking_id=0)
    response = client.post('/add_data', json={'model_name': 'trackingItem', 'data': test_tracking_item.model_dump()})
    assert response.status_code == 200

def test_get_data_by_id(client):
    response = client.get('/get_data_by_id?model_name=evironment&item_id=0')
    assert response.status_code == 200

def test_update_data(client):
    test_environment = Environment(env_name='test_environment_updated', description='test_description_updated')
    response = client.post('/update_data', json={'model_name': 'environment', 'item_id': 0, 'data': test_environment.model_dump()})
    assert response.status_code == 200

def test_delete_data(client):
    response = client.delete('/delete_data?model_name=environment&item_id=0')
    assert response.status_code == 200

def test_health(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'message': 'OK'}


def test_ui_configs(client):
    response = client.get('/ui_configs')
    assert response.status_code == 200


def test_save_ui_config(client):
    test_ui_config = {'test': 'test', 'name': 'test_ui_config'}
    response = client.post('/save_ui_config', json=test_ui_config)
    assert response.status_code == 200

    response = client.get('/ui_configs')
    assert response.status_code == 200


def test_retreive_logs(client):
    response = client.get('/retreive_logs')
    assert response.status_code == 200


def test_retreive_demos(client):
    response = client.get('/retreive_demos')
    assert response.status_code == 200


def test_retreive_feature_feedback(client):
    response = client.get('/retreive_feature_feedback')
    assert response.status_code == 200