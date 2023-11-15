import argparse
import json
import os
import sys
import zipfile

import uvicorn
from databases import Database
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rlhfblender.data_handling import database_handler as db_handler
from rlhfblender.config import DB_HOST
from rlhfblender.data_collection.feedback_translator import FeedbackTranslator
from rlhfblender.data_collection.sampler import Sampler
from rlhfblender.data_models import get_model_by_name
from rlhfblender.data_models.global_models import (
    Dataset,
    Environment,
    Experiment,
    Project,
    TrackingItem,
)
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
    version="0.1.0",
)
app.include_router(data.router)

app.mount("/files", StaticFiles(directory="rlhfblender/static_files"), name="files")
app.mount(
    "/action_labels", StaticFiles(directory="data/action_labels"), name="action_labels"
)
app.mount("/logs", StaticFiles(directory="logs"), name="logs")

database = Database(os.environ.get("RLHFBLENDER_DB_HOST", DB_HOST))


@app.on_event("startup")
async def startup():
    await database.connect()
    await db_handler.create_table_from_model(database, Project)
    await db_handler.create_table_from_model(database, Experiment)
    await db_handler.create_table_from_model(database, Environment)
    await db_handler.create_table_from_model(database, Dataset)
    await db_handler.create_table_from_model(database, TrackingItem)

    # add sampler and feedback model to app state
    app.state.sampler = Sampler(None, None, os.path.join("data", "renders"))
    app.state.feedback_translator = FeedbackTranslator(None, None)

    # Run the startup script as a separate process
    startup_script_path = os.path.join("app", "startup_script.py")
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


@app.post("/add_data", response_model=BaseModel, tags=["DATA"])
async def add_data(model_name: str, data: dict):
    model = get_model_by_name(model_name)
    if model is None:
        return {"message": f"Model {model_name} not found."}
    await db_handler.add_entry(database, model, data)
    return {"message": f"Added {model_name}"}


@app.post("/update_data", response_model=BaseModel, tags=["DATA"])
async def update_data(model_name: str, item_id: int, data: dict):
    model = get_model_by_name(model_name)
    if model is None:
        return {"message": f"Model {model_name} not found."}
    await db_handler.update_entry(database, model, item_id, data)
    return {"message": f"Updated {model_name} with id {item_id}"}


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
            with open(os.path.join("configs/ui_configs", filename)) as f:
                ui_confs.append(json.load(f))
    return ui_confs


@app.post("/save_ui_config", tags=["UI"])
async def save_ui_config(ui_config: dict):
    # Save UI config to configs/ui_configs directory
    with open(
        os.path.join("configs/ui_configs", ui_config["name"] + ".json"), "w"
    ) as f:
        json.dump(ui_config, f)
    return {"message": "OK"}


@app.post("delete_ui_config", tags=["UI"])
async def delete_ui_config(ui_config_name: str):
    # Delete UI config from configs/ui_configs directory
    os.remove(os.path.join("configs/ui_configs", ui_config_name + ".json"))
    return {"message": "OK"}


@app.get("/retreive_logs", tags=["LOGS"])
async def retreive_logs():
    # Return list of CSV files from logs directory, zip them and proide download link
    logs = []
    for filename in os.listdir("logs"):
        if filename.endswith(".csv"):
            logs.append(filename)
    with zipfile.ZipFile("logs.zip", "w") as zip:
        for log in logs:
            zip.write(os.path.join("logs", log))
    return FileResponse("logs.zip", media_type="application/zip", filename="logs.zip")


@app.get("/retreive_demos", tags=["LOGS"])
async def retreive_logs():
    # Return list of CSV files from logs directory, zip them and proide download link
    logs = []
    for filename in os.listdir(os.path.join("data", "generated_demos")):
        if filename.endswith(".npz"):
            logs.append(filename)
    with zipfile.ZipFile("logs.zip", "w") as zip:
        for log in logs:
            zip.write(os.path.join("data", "generated_demos", log))
    return FileResponse("demos.zip", media_type="application/zip", filename="demos.zip")


@app.get("/retreive_feature_feedback", tags=["LOGS"])
async def retreive_logs():
    # Return list of CSV files from logs directory, zip them and proide download link
    logs = []
    for filename in os.listdir(os.path.join("data", "feature_feedback")):
        if filename.endswith(".png"):
            logs.append(filename)
    with zipfile.ZipFile("logs.zip", "w") as zip:
        for log in logs:
            zip.write(os.path.join("data", "feature_feedback", log))
    return FileResponse(
        "feature_selections.zip",
        media_type="application/zip",
        filename="feature_selections.zip",
    )


def main(args):
    parser = argparse.ArgumentParser(description="Test Python Backend")

    parser.add_argument("--port", type=int, default=8080, help="Port to run server on.")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="If true, restart the server as changes occur to the code.",
    )

    args = parser.parse_args(args)

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
