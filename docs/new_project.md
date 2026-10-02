# Start a new Dash or MiniGrid project

These instructions describe the `dash_driving` branch. They create a project, register its environment and a named random-policy experiment, generate example episodes, and open them in RLHF-Blender. No trained model or model repository is needed.

Dash driving is included in this repository as **`dash-rl-env/`**. Its Compose service is `dash-driving`; its Gym ID is `dash-driving-v0`. There is no separate Dash repository to clone for this branch.

Run steps 1–2 on your host, then steps 3–5 in the backend shell opened in step 2. Choose either Dash or MiniGrid in step 3. For installation without Docker, use the [local alternative](#local-alternative-without-docker) before continuing with steps 3–6.

## 1. Get this branch

Install Git and Docker with Compose, and start Docker Desktop or your Docker daemon. The commands below use `docker-compose`; if your installation uses the Compose plugin, replace it with `docker compose` throughout.

```bash
git clone --branch dash_driving https://github.com/ymetz/rlhfblender.git
cd rlhfblender
git submodule update --init rlhfblender-ui
```

If you received an existing checkout, use that checkout instead and check `git branch --show-current`. The commands below assume the repository root as the working directory. Use the frontend revision recorded by this branch.

Compose supplies the database and Dash addresses for a local Docker setup. No `.env.local` file or manual database export is required. For remote access or a different database, see [Docker settings](#docker-settings-and-overrides).

## 2. Start the services and open a backend shell

On the host:

```bash
docker-compose up -d --build
docker-compose ps
docker-compose logs --tail=30 backend dash-driving
```

This starts all three services, including Dash when following the MiniGrid recipe:

| Service | Host address | Purpose |
| --- | --- | --- |
| `frontend` | http://localhost:3000 | RLHF-Blender interface |
| `backend` | http://localhost:8080/docs | API and API documentation |
| `dash-driving` | http://localhost:5173 | Dash simulator |

Wait for Dash to report `healthy` and for the backend's startup to complete. The Dash health check uses `http://dash-driving:5173/`, exercising Docker DNS and Vite's host allowlist. The image includes Playwright and Chromium; you do not need to install them manually inside Docker.

If you use Colima, run `colima start` before starting Compose. Both services communicate on the Compose network inside the VM; `DASH_PLAYER_URL` remains `http://dash-driving:5173`.

Open a shell with the backend's Python environment activated:

```bash
docker-compose exec backend micromamba run -n base bash
```

Keep this shell open for steps 3–5. Compose sets `RLHFBLENDER_DB_HOST=sqlite:///data/rlhfblender.db` for both the API and all commands run with `docker-compose exec`, so they use the same database automatically. The shell also inherits `DASH_PLAYER_URL=http://dash-driving:5173`; this is Dash's address inside the container network.

To run a command directly from the host instead of opening a shell, prefix `python -m rlhfblender.generate_data ...` with `docker-compose exec backend micromamba run -n base`.

## 3. Register your project, environment, and experiment

A **project** groups experiments. An **environment** identifies the simulator/task. An **experiment** identifies a policy and its configuration; here it is a random baseline. A **saved setup** is created later in the UI and stores the selections for a study.

Choose one recipe. Change the project and experiment names for your own work. Experiment names are global in the database, so use a new experiment name for each new baseline/project. Always pass both `--project` and `--exp`.

### Option A: Dash driving

In the backend shell:

```bash
export PROJECT_NAME="Dash tutorial"
export EXPERIMENT_NAME="dash-tutorial-random"
export ENV_ID="dash-driving-v0"

python -m rlhfblender.generate_data \
  --project "$PROJECT_NAME" \
  --exp "$EXPERIMENT_NAME" \
  --env "$ENV_ID" \
  --env-gym-entrypoint rlhfblender.data_collection.dash_driving_gym_env:DashDrivingGymEnv \
  --env-display-name "Dash driving" \
  --action-names steer gas brake \
  --env-config '{"env_kwargs":{"default_reset_options":{"scenarioName":"rough_road","startMode":"manual","clearRecording":true}}}' \
  --random \
  --register-only
```

This creates the project, links the environment/experiment, and configures the `rough_road` scenario in one command. `--register-only` creates metadata, not episodes. Keep the Gym ID beginning with `dash-driving` so the UI recognizes the Dash demonstration flow.

### Option B: MiniGrid

MiniGrid is optional and is not installed by the current backend Dockerfile. In the backend shell:

```bash
python -m pip install "minigrid==3.0.0"

export PROJECT_NAME="MiniGrid tutorial"
export EXPERIMENT_NAME="minigrid-tutorial-random"
export ENV_ID="MiniGrid-Empty-5x5-v0"

python -m rlhfblender.generate_data \
  --project "$PROJECT_NAME" \
  --exp "$EXPERIMENT_NAME" \
  --env "$ENV_ID" \
  --additional-gym-packages minigrid \
  --env-display-name "MiniGrid Empty 5x5" \
  --action-names left right forward pickup drop toggle done \
  --env-wrapper minigrid.wrappers.ImgObsWrapper \
  --random \
  --register-only
```

`--additional-gym-packages minigrid` imports MiniGrid's built-in Gym registrations; it does not install the package. This simple environment needs no custom entry point. `--env-wrapper` configures image observations because MiniGrid's default textual mission cannot be handled directly by the recorder's vectorized environment.

The package installed with `docker-compose exec` stays in that container, but is lost when the container is recreated. Repeat the install after recreation, or add `RUN pip install "minigrid==3.0.0"` to the backend Dockerfile after its Python dependency installation and rebuild for a persistent MiniGrid image.

## 4. Adjust configuration via CLI (optional)

Step 3 already applies the required configuration and project links. You can go straight to step 5. To change an existing experiment later, use the same CLI with `--register-only`; no Python script or manual SQL is needed.

For Dash, for example, change the scenario to `negotiating_crosswalks`:

```bash
python -m rlhfblender.generate_data \
  --project "$PROJECT_NAME" --exp "$EXPERIMENT_NAME" --env "$ENV_ID" \
  --env-config '{"env_kwargs":{"default_reset_options":{"scenarioName":"negotiating_crosswalks"}}}' \
  --register-only
```

`--env-config` accepts a JSON object with typed values, including nested dictionaries, numbers, and booleans. It recursively merges the supplied keys into the experiment's stored `environment_config`. The example keeps `startMode`, `clearRecording`, and all other settings. Omitted settings are preserved; lists and scalar values are replaced. These flags require `--exp`.

For MiniGrid, apply or repair the wrapper setting with:

```bash
python -m rlhfblender.generate_data \
  --project "$PROJECT_NAME" --exp "$EXPERIMENT_NAME" --env "$ENV_ID" \
  --env-wrapper minigrid.wrappers.ImgObsWrapper \
  --register-only
```

`--env-wrapper` sets the top-level `env_wrapper` field, not a Gym constructor argument. Both flags also work in the initial registration command, as shown in step 3. Changing configuration affects future recordings, not existing videos.

Every registration call attaches existing environments/experiments to the requested project, without duplicating membership or removing other projects. Existing experiment policy/model fields and environment metadata are retained. Reusing an experiment name with a different environment is rejected; choose a new name instead. Experiment configuration is shared across projects that reference the same experiment, so use a new experiment name when you need independent settings.

Dash scenarios include `rough_road`, `lane_blockage_with_oncoming_traffic`, and `negotiating_crosswalks`. Gym captures use `ui=gym`, hide the welcome modal, and place the dashboard below the simulator.

## 5. Generate the first episodes

Still in the backend shell, for either recipe:

```bash
python -m rlhfblender.generate_data \
  --project "$PROJECT_NAME" \
  --exp "$EXPERIMENT_NAME" \
  --env "$ENV_ID" \
  --random \
  --num-episodes 3 \
  --max-steps-per-episode 100
```

The existing environment entry point, additional packages, and experiment configuration are read from the database. The `--env` argument remains required. This records a random baseline; it does not train a policy or reward model. Start with `--num-episodes 1 --max-steps-per-episode 5` for a quick check.

Check for generated episodes and videos:

```bash
find "data/episodes/$ENV_ID" -name '*.npz'
find "data/renders/$ENV_ID" -name '*.mp4'
```

The experiment now has checkpoint `-1`, displayed as **Random** in the UI. The CLI exits with a nonzero status on recording errors and prints `Data generation finished.` only on success.

With Docker, `data/` maps to **`remote_data/` on the host**. The SQLite database, episodes, rewards, thumbnails, and videos are persisted there. `data/saved_benchmarks/` holds intermediate recordings which are deleted after successful processing. Recording again with the same experiment/checkpoint can replace its artifacts; use a fresh experiment name to keep a separate dataset.

## 6. Open the project and save a study setup

1. Open or reload **http://localhost:3000**. The plain URL opens configuration mode. Expand the top menu with the chevron if necessary.
2. Select **Project** → `Dash tutorial` or `MiniGrid tutorial`.
3. Select **Experiment** → the experiment name you registered.
4. Select **Checkpoint** → **Random**.
5. Select a **Backend Config** and **UI Config**. Use their adjacent controls to configure sampling and feedback types. Confirm that the recorded episodes load before saving the setup.
6. Click **Save Setup** and keep the returned study code/link. The participant link is `http://localhost:3000/?study=CODE`. For the active-learning interface, use `http://localhost:3000/?study_mode=active-learning` during configuration, or `http://localhost:3000/?study_mode=active-learning&study=CODE` with a saved setup.

Choose a checkpoint explicitly after changing the project or experiment. **Load** and **Save Setup** stay disabled until a checkpoint is selected. **Random** is a valid checkpoint; selecting it loads the recorded baseline.

No separate "create project" command or manual SQL insert is required. Registration with `--project` creates the project; **Save Setup** creates a study configuration, not a second project.

To verify what the API sees, open http://localhost:8080/get_all?model_name=project and http://localhost:8080/get_all?model_name=experiment. A browser refresh is needed after registering new records because the frontend initially loads these lists at startup.

Saved setup and UI/backend configuration JSON files live under `configs/` inside the backend. Unlike `data/` and `logs/`, `configs/` is **not bind-mounted in this branch's Compose file**. Before recreating the backend, copy any newly saved configuration files to the host, for example from a host terminal:

```bash
mkdir -p study-config-backup
docker-compose cp backend:/home/mambauser/rlhfblender/configs/. ./study-config-backup/
```

Keep the relevant JSON files with the data when handing a study to someone else. A study code alone does not include the configuration or recordings.

## Docker settings and overrides

Compose preconfigures:

| Setting | Default | Used by |
| --- | --- | --- |
| `RLHFBLENDER_DB_HOST` | `sqlite:///data/rlhfblender.db` | Backend API and registration/generation CLI |
| `DASH_PLAYER_URL` | `http://dash-driving:5173` | Backend's headless Gym browser |
| `VITE_DASH_PLAYER_URL` | `http://localhost:5173` | Frontend build; interactive Dash iframe |
| `VITE_DASH_PLAYER_ORIGIN` | `http://localhost:5173` | Frontend build; iframe message validation |
| `DISPLAY` | Empty | Headless Chromium software WebGL |

For access from another computer, create a `.env` file **at the repository root** with browser-reachable addresses, for example:

```dotenv
VITE_DASH_PLAYER_URL=http://my-server:5173
VITE_DASH_PLAYER_ORIGIN=http://my-server:5173
```

Then run `docker-compose up -d --build`. Vite embeds these values in the frontend JavaScript during the image build; changing only runtime environment variables is insufficient. Keep the origin to the scheme, hostname, and port. The backend's `DASH_PLAYER_URL` stays at the internal service address.

You can override `RLHFBLENDER_DB_HOST` in the same root `.env` file or host shell. The API honors it unless explicitly started with `--db-host`, which takes precedence. Keep SQLite files under the mounted `data/` directory for persistence. For the default local Docker setup, none of these overrides is necessary.

## Local alternative without Docker

Use Python 3.10+ and a Node version supported by the frontend (`20.19+` within Node 20, or `22.12+`; the Docker images use Node 22). Perform step 1. For local Dash demonstrations, add these settings to `rlhfblender-ui/.env.local` before starting the frontend:

```dotenv
VITE_DASH_PLAYER_URL=http://localhost:5173
VITE_DASH_PLAYER_ORIGIN=http://localhost:5173
```

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --only-binary=av -e .
python -m playwright install chromium
mkdir -p data logs configs/setups
```

On Linux, use `python -m playwright install --with-deps chromium` if Chromium's system libraries are missing. MiniGrid users also run the `pip install` command from option B.

Start the backend in one terminal, from the repository root with the virtual environment active:

```bash
python rlhfblender/app.py --port 8080 --db-host sqlite:///data/rlhfblender.db
```

Start the frontend in another terminal:

```bash
cd rlhfblender-ui
npm install
npm run start
```

For Dash, start its server in a third terminal:

```bash
cd dash-rl-env
npm ci
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

In a fourth terminal, from the repository root, prepare the shell used for steps 3–5:

```bash
source .venv/bin/activate
export RLHFBLENDER_DB_HOST=sqlite:///data/rlhfblender.db
export DASH_PLAYER_URL=http://localhost:5173
```

Then follow the same registration, configuration, generation, and UI steps above. Locally the files live in `data/`, not `remote_data/`. Run the API and generation CLI from the same repository root so their relative database/data paths agree.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Project or experiment missing in UI | Rebuild/recreate the backend after updating this branch so it receives the Compose defaults. Register with the intended `--project` and `--exp`, then reload the browser. For local runs, export the same `RLHFBLENDER_DB_HOST` as the API. |
| Project exists but no recordings/checkpoints | `--register-only` creates no data. Run step 5 and check for actual `.npz`/`.mp4` files. |
| Random selected but status stays Waiting and no sampler request is sent | Update/rebuild the frontend. Older `mujoco_3d` frontend code treated Random (`-1`) as an empty selection. The current frontend uses `null` for an empty selection. |
| Status is Active but no episode cards appear despite existing recordings | Update the backend sampler: artifact paths must use the Gym registration ID (e.g. `dash-driving-v0`), not the display label (`Dash driving`). |
| `Cannot import minigrid` / missing MiniGrid environment | Install MiniGrid in the Python environment running the backend and CLI. Use `--additional-gym-packages minigrid` during first registration. |
| MiniGrid fails in `DummyVecEnv` with `NoneType` or mission/text observations | Apply step 4's `ImgObsWrapper` configuration to the selected experiment. |
| Dash has no road | Apply step 4's `default_reset_options` with a built-in `scenarioName`. |
| Dash Playwright reports `ERR_NAME_NOT_RESOLVED` | The backend cannot resolve the Dash service name. Check that both containers are attached to the same Compose network; see the connectivity checks below. |
| Dash Playwright reports `ERR_CONNECTION_REFUSED` | Check `docker-compose logs dash-driving`, the service health, and the backend's `DASH_PLAYER_URL`. Docker uses `http://dash-driving:5173`; local execution uses `http://localhost:5173`. |
| Dash returns HTTP 403 / `This host ("dash-driving") is not allowed` | Rebuild the Dash image with this branch's `preview.allowedHosts: ['dash-driving']` setting in `dash-rl-env/vite.config.mjs`. |
| Interactive Dash iframe is blank or never becomes ready | Compose defaults both `VITE_DASH_PLAYER_URL` and `VITE_DASH_PLAYER_ORIGIN` to `http://localhost:5173`. For remote access, override both in the root `.env` and rebuild the frontend. Local development uses the frontend’s `.env.local`. |
| Headless Docker browser cannot create WebGL | Keep Compose's empty `DISPLAY` override; the image's `:99` display has no running X server. |
| Existing record ignores changed registration flags | Use `--env-config` / `--env-wrapper` to merge experiment settings (step 4). Existing environment metadata and experiment policy/model fields are retained; choose a new registration name to change those. |

Environment IDs are shared across projects. Do not delete an environment or the database just to start a second project. Choose new project/experiment names and reuse the environment ID; the CLI establishes membership automatically.

### Docker/Colima connectivity checks

Run these from a **host terminal** in the repository root:

```bash
docker-compose ps
docker inspect dash-driving-ui rlhfblender-backend --format '{{.Name}} {{json .NetworkSettings.Networks}}'
docker-compose exec -T backend micromamba run -n base python -c 'import os, urllib.request; url = os.environ["DASH_PLAYER_URL"]; print(url, urllib.request.urlopen(url, timeout=10).status)'
```

Both containers should list the same Compose network, normally `rlhfblender_default`, and the HTTP check should print `http://dash-driving:5173 200`. An empty network map (`{}`) for Dash means it is detached. Earlier versions checked only `localhost` and could report `healthy` even in that state; the current health check also detects missing DNS or a rejected hostname.

To repair a detached/stale Dash container or apply the Vite fix, recreate only the simulator:

```bash
docker-compose up -d --build --force-recreate --no-deps dash-driving
```

Wait for Dash to become healthy, repeat the HTTP check, and retry generation. This leaves the backend, database, and existing project registrations intact. There is no need to register the project again or change the backend URL to `localhost`, which would address the backend container itself.
