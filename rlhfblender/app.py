import argparse
import json
import os
import sys
import uuid
import zipfile
from datetime import datetime

import uvicorn
from databases import Database
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rlhfblender.data_collection.feedback_translator import FeedbackTranslator
from rlhfblender.data_collection.sampler import Sampler
from rlhfblender.data_handling import database_handler as db_handler
from rlhfblender.data_models import get_model_by_name
from rlhfblender.data_models.global_models import (
    Dataset,
    Environment,
    Experiment,
    Project,
    TrackingItem,
)
from rlhfblender.logger import CSVLogger, GoogleSheetsLogger, JSONLogger, SQLLogger
from rlhfblender.routes import data

# from fastapi_sessions.backends.implementations import InMemoryBackend
# from fastapi_sessions.frontends.implementations import SessionCookie, CookieParameters
# from session import SessionData, BasicVerifier


# Create some cache directories which are expected later on (do before startup to allow for mounting)
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("data/action_labels", exist_ok=True)

# == App ==
app = FastAPI(
    title="Test Python Backend",
    description="""This is a template for a Python backend.
                   It provides access via REST API.""",
    version="0.5.0",
)
app.include_router(data.router)

app.mount("/files", StaticFiles(directory=os.path.join("rlhfblender", "static_files")), name="files")
app.mount("/action_labels", StaticFiles(directory=os.path.join("data", "action_labels")), name="action_labels")
app.mount("/logs", StaticFiles(directory="logs"), name="logs")

database = Database(os.environ.get("RLHFBLENDER_DB_HOST", "sqlite:///rlhfblender.db"))


@app.on_event("startup")
async def startup():
    await database.connect()
    await db_handler.create_table_from_model(database, Project)
    await db_handler.create_table_from_model(database, Experiment)
    await db_handler.create_table_from_model(database, Environment)
    await db_handler.create_table_from_model(database, Dataset)
    await db_handler.create_table_from_model(database, TrackingItem)

    # initialize logger
    logger_type = os.environ.get("RLHFBLENDER_LOGGER_TYPE", "csv")
    if logger_type == "sql":
        app.state.logger = SQLLogger(None, None, os.path.join("logs"))
    elif logger_type == "json":
        app.state.logger = JSONLogger(None, None, os.path.join("logs"))
    else:
        # check if credentials file is provided, if so use Google Sheets logger
        credentials_file = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE", "google-service-account.json")
        if os.path.isfile(credentials_file):
            print("Using Google Sheets logger.")
            app.state.logger = GoogleSheetsLogger(None, None, os.path.join("logs"), credentials_file)
        else:
            print("[INFO] You can provide a Google service account file to use Google Sheets logger. Defaulting to CSV.")
            app.state.logger = CSVLogger(None, None, os.path.join("logs"))

    # add sampler and feedback model to app state
    app.state.sampler = Sampler(None, None, os.path.join("data", "renders"), logger=app.state.logger)
    app.state.feedback_translator = FeedbackTranslator(None, None, logger=app.state.logger)

    # Run the startup script as a separate process
    startup_script_path = os.path.join("rlhfblender", "startup_script.py")
    if os.path.isfile(startup_script_path):
        print("Running startup script...")
        os.system(f"python3 {startup_script_path}")
    else:
        print("No startup script found. Skipping...")
    print("Startup script finished.")


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()


# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse, tags=["ROOT"])
async def root():
    html_content = """
        <html>
            <head>
                <title>RL Workbench</title>
            </head>
            <body>
                <h1>Test Python Backend</h1>
                Visit the <a href="/docs">API doc</a> (<a href="/redoc">alternative</a>) for usage information.
            </body>
        </html>
        """
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/get_all", tags=["DATA"])
async def get_all(model_name: str):
    model = get_model_by_name(model_name)
    if model is None:
        return {"message": f"Model {model_name} not found."}
    return await db_handler.get_all(database, model)


@app.get("/get_data_by_id", response_model=BaseModel, tags=["DATA"])
async def get_data_by_id(model_name: str, item_id: int):
    model = get_model_by_name(model_name)
    if model is None:
        return {"message": f"Model {model_name} not found."}
    return await db_handler.get_single_entry(database, model, item_id)


class AddDataRequest(BaseModel):
    model_name: str
    data: dict


@app.post("/add_data", response_model=BaseModel, tags=["DATA"])
async def add_data(req: AddDataRequest):
    model = get_model_by_name(req.model_name)
    if model is None:
        return {"message": f"Model {req.model_name} not found."}
    await db_handler.add_entry(database, model, req.data)
    return {"message": f"Added {req.model_name}"}


class UpdateDataRequest(BaseModel):
    model_name: str
    item_id: int
    data: dict


@app.post("/update_data", response_model=BaseModel, tags=["DATA"])
async def update_data(req: UpdateDataRequest):
    model = get_model_by_name(req.model_name)
    if model is None:
        return {"message": f"Model {req.model_name} not found."}
    await db_handler.update_entry(database, model, req.item_id, data=req.data)
    return {"message": f"Updated {req.model_name} with id {req.item_id}"}


@app.delete("/delete_data", response_model=BaseModel, tags=["DATA"])
async def delete_data(model_name: str, item_id: int):
    model = get_model_by_name(model_name)
    if model is None:
        return {"message": f"Model {model_name} not found."}
    await db_handler.delete_entry(database, model, item_id)
    return {"message": f"Deleted {model_name} with id {item_id}"}


@app.get("/health", tags=["HEALTH"])
async def health():
    return {"message": "OK"}


@app.get("/ui_configs", tags=["UI"])
async def ui_configs():
    # Return list of JSONs with UI configs from configs/ui_configs directory
    ui_confs = []
    for filename in os.listdir("configs/ui_configs"):
        if filename.endswith(".json"):
            with open(os.path.join("configs", "ui_configs", filename)) as f:
                ui_confs.append(json.load(f))
    ui_confs.sort(key=lambda x: datetime.strptime(x["created_at"], "%Y-%m-%d %H:%M:%S"), reverse=True)
    # Check if there
    return ui_confs


@app.post("/save_ui_config", tags=["UI"])
async def save_ui_config(ui_config: dict):
    # Save UI config to configs/ui_configs directory
    ui_config_id = uuid.uuid4().hex[:8]
    ui_config["id"] = ui_config_id
    ui_config["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join("configs/ui_configs", f"{ui_config_id}.json"), "w") as f:
        json.dump(ui_config, f)
    return {"message": "OK", "id": ui_config_id}


class DeleteUIConfigRequest(BaseModel):
    ui_config_id: str


@app.post("/delete_ui_config", tags=["UI"])
async def delete_ui_config(req: DeleteUIConfigRequest):
    # Delete UI config from configs/ui_configs directory
    os.remove(os.path.join("configs", "ui_configs", req.ui_config_id + ".json"))
    return {"message": "OK"}


@app.get("/backend_configs", tags=["BACKEND"])
async def backend_configs():
    # Return list of JSONs with backend configs from configs/backend_configs directory
    backend_confs = []
    for filename in os.listdir("configs/backend_configs"):
        if filename.endswith(".json"):
            with open(os.path.join("configs", "backend_configs", filename)) as f:
                backend_confs.append(json.load(f))
    # sort by date (created_at)
    backend_confs.sort(key=lambda x: datetime.strptime(x["created_at"], "%Y-%m-%d %H:%M:%S"), reverse=True)
    return backend_confs


@app.post("/save_backend_config", tags=["BACKEND"])
async def save_backend_config(backend_config: dict):
    # Save backend config to configs/backend_configs directory
    backend_config_id = uuid.uuid4().hex[:8]
    backend_config["id"] = backend_config_id
    backend_config["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join("configs", "backend_configs", f"{backend_config_id}.json"), "w") as f:
        json.dump(backend_config, f)
    return {"message": "OK"}


class DeleteBackendConfigRequest(BaseModel):
    backend_config_name: str


@app.post("/delete_backend_config", tags=["BACKEND"])
async def delete_backend_config(req: DeleteBackendConfigRequest):
    # Delete backend config from configs/backend_configs directory
    os.remove(os.path.join("configs", "backend_configs", req.backend_config_name + ".json"))
    return {"message": "OK"}


class SaveSetupRequest(BaseModel):
    project: dict
    experiment: dict
    ui_config: dict
    backend_config: dict


@app.post("/save_setup", tags=["SETUP"])
async def save_setup(req: SaveSetupRequest):
    """
    Save to file in configs/setups, generate a unique ID and return it
    """
    # Save setup to configs/setups directory
    setup_id = uuid.uuid4().hex[:8]
    setup = {
        "id": setup_id,
        "project": req.project,
        "experiment": req.experiment,
        "ui_config": req.ui_config,
        "backend_config": req.backend_config,
    }
    with open(os.path.join("configs", "setups", f"{setup_id}.json"), "w") as f:
        json.dump(setup, f)
    return {"study_code": setup_id}


class LoadSetupRequest(BaseModel):
    study_code: str


@app.post("/load_setup", tags=["SETUP"])
async def load_setup(req: LoadSetupRequest):
    """
    Load setup from file in configs/setups
    """
    study_code = req.study_code
    try:
        with open(os.path.join("configs/setups", f"{study_code}.json")) as f:
            return json.load(f)
    except FileNotFoundError:
        # return error message and status code
        return {"message": "Setup not found."}, 404


@app.get("/retrieve_logs", tags=["LOGS"])
async def retrieve_logs():
    # Return list of CSV files from logs directory, zip them and proide download link
    logs = []
    try:
        for filename in os.listdir("logs"):
            if filename.endswith(".csv"):
                logs.append(filename)
        with zipfile.ZipFile("logs.zip", "w") as zip:
            for log in logs:
                zip.write(os.path.join("logs", log))
        return FileResponse("logs.zip", media_type="application/zip", filename="logs.zip")
    except FileNotFoundError:
        return {"message": "No logs found."}


@app.get("/retrieve_demos", tags=["LOGS"])
async def retrieve_demos():
    # Return list of CSV files from logs directory, zip them and proide download link
    demos = []
    try:
        for filename in os.listdir(os.path.join("logs", "generated_demos")):
            if filename.endswith(".npz"):
                demos.append(filename)
        with zipfile.ZipFile("logs.zip", "w") as zip:
            for log in demos:
                zip.write(os.path.join("logs", "generated_demos", log))
        return FileResponse("demos.zip", media_type="application/zip", filename="demos.zip")
    except FileNotFoundError:
        return {"message": "No demos found."}


@app.get("/retrieve_feature_feedback", tags=["LOGS"])
async def retrieve_feature_feedback():
    # Return list of CSV files from logs directory, zip them and proide download link
    feedbacks = []
    try:
        for filename in os.listdir(os.path.join("logs", "feature_feedback")):
            if filename.endswith(".png"):
                feedbacks.append(filename)
        with zipfile.ZipFile("logs.zip", "w") as zip:
            for log in feedbacks:
                zip.write(os.path.join("logs", "feature_feedback", log))
        return FileResponse(
            "feature_selections.zip",
            media_type="application/zip",
            filename="feature_selections.zip",
        )
    except FileNotFoundError:
        return {"message": "No feature feedback data found."}


def main(args):
    parser = argparse.ArgumentParser(description="Test Python Backend")

    parser.add_argument("--port", type=int, default=8080, help="Port to run server on.")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="If true, restart the server as changes occur to the code.",
    )
    parser.add_argument("--ui-config", type=str, default=None, help="Path to UI config file.")
    parser.add_argument("--backend-config", type=str, default=None, help="Path to backend config file.")
    parser.add_argument("--db-host", type=str, default="sqlite:///rlhfblender.db", help="Path to database file.")
    parser.add_argument("--logger-type", type=str, default="csv", help="Type of logger to use (sql, json, csv).")

    args = parser.parse_args(args)

    # Write config paths to environment variables
    os.environ["RLHFBLENDER_UI_CONFIG_PATH"] = args.ui_config if args.ui_config is not None else ""
    os.environ["RLHFBLENDER_BACKEND_CONFIG_PATH"] = args.backend_config if args.backend_config is not None else ""
    os.environ["RLHFBLENDER_DB_HOST"] = args.db_host if args.db_host is not None else ""
    os.environ["RLHFBLENDER_LOGGER_TYPE"] = args.logger_type if args.logger_type is not None else ""

    if args.dev:
        print(f"Serving on port {args.port} in development mode.")

        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=args.port,
            reload=True,
            access_log=False,
            workers=1,
        )
    else:
        print(f"Serving on port {args.port} in live mode.")
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=args.port,
            reload=False,
            access_log=False,
            workers=1,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
