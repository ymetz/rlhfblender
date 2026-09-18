# RLHF-Blender

Implementation for RLHF-Blender: A Configurable Interactive Interface for Learning from Diverse Human Feedback
Paper: https://arxiv.org/abs/2308.04332 (Presented at the ICML2023 Interactive Learning from Implicit Human Feedback Workshop)

<div align="center">

Website + Demo: https://sites.google.com/view/rlhfblender

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/github/license/ymetz/rlhfblender)](https://github.com/ymetz/rlhfblender/blob/master/LICENSE)
![CI](https://github.com/ymetz/rlhfblender/workflows/CI/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/rlhfblender/badge/?version=latest)](https://readthedocs.org/projects/rlhfblender/badge/?version=latest)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

Documentation: https://rlhfblender.readthedocs.io/en/latest/

> [!NOTE]  
> The following repository is part of of the RLHF-Blender project. The frontend is part of a separate repository: [RLHF-Blender-UI](https://github.com/ymetz/rlhfblender-ui)
> you may follow the installation instructions to also install the frontend

## Installation

1. Clone the repository

```bash
git clone https://github.com/ymetz/rlhfblender.git
cd rlhfblender
git submodule update --init rlhfblender-ui
```
to get both the main repository,user interface. If you want to download both the repository and demo models, you can also run ```git clone --recurse-submodules https://github.com/ymetz/rlhfblender.git```.

2. Docker-Installation

```bash
python3 scripts/update_dockerignore.py
docker compose up --build
```

The backend `.dockerignore` allows only Git-tracked build inputs. Run the generator
after adding or removing tracked files; `scripts/build_docker.sh` does this automatically.
Edits to tracked files are included, while untracked files are excluded even inside
source directories. The frontend has its own build context in `rlhfblender-ui`.

Compose mounts local datasets (`remote_data`), models, training artifacts, and logs
instead of baking them into the backend image. The database stays at
`remote_data/rlhfblender.db`; changes made by the container persist locally. These
directories must be accessible to the container user. Standalone `docker run` deployments
must supply their own mounts for any required datasets or models.

(3. Optional: Local/Dev. Install):

```bash
pip install -e .
python -m playwright install chromium
python rlhfblender/app.py
```

and

```bash
cd rlhfblender-ui
npm install
npm run start
```

The user interface is then available at http://localhost:3000

## 📦 Features

RLHF-Blender allows to configure experimental setups for RLHF-experiments based on several modular components:

- A freely configurable user interface for different feedback type interactions
- Feedback processors, handling the translation of different types of feedback, incl. meta-data, into a common format
- Adaptor to different reward models (e.g. reward model ensembles, AIRL-style models, etc.)

## 📖 Example

RLHF-Blender allows to quickly setup experiments for experimenting with different types of feedback and reward models across different environments. 
The following example shows how to setup an experiment for the CartPole environment with a reward model ensemble and a textual feedback interface.

## Environment Registration and Benchmark Collection

You can register an environment and collect benchmark episodes with `rlhfblender.generate_data`.
This populates the benchmark/episode artifacts used by the UI (`data/saved_benchmarks`, `data/episodes`, `data/renders`, `data/rewards`, ...).

```bash
# Register only
python -m rlhfblender.generate_data \
  --env MyEnv-v0 \
  --env-gym-entrypoint my_package.envs:MyEnv \
  --register-only

# Register + collect random benchmark episodes
python -m rlhfblender.generate_data \
  --env MyEnv-v0 \
  --env-gym-entrypoint my_package.envs:MyEnv \
  --random \
  --num-episodes 10
```

Dash-driving integration can be registered similarly:

For local execution, first start the Dash player in `dash-rl-env` with `npm install` and `npm run dev`.
The environment uses `http://localhost:5173` by default; set `DASH_PLAYER_URL` to use another address.
The Gym wrapper adds `ui=gym` to the player URL by default, moving the HUD below the simulator,
and suppresses the welcome modal. An explicit `ui` query parameter overrides the layout.

```bash
python -m rlhfblender.generate_data \
  --env dash-driving-v0 \
  --env-gym-entrypoint rlhfblender.data_collection.dash_driving_gym_env:DashDrivingGymEnv \
  --random \
  --num-episodes 10
```

For Docker, start the services and run data generation in the backend container:

```bash
docker-compose up -d --build
docker-compose exec backend micromamba run -n base python -m rlhfblender.generate_data \
  --env dash-driving-v0 \
  --env-gym-entrypoint rlhfblender.data_collection.dash_driving_gym_env:DashDrivingGymEnv \
  --random \
  --num-episodes 1
```

Compose sets `DASH_PLAYER_URL=http://dash-driving:5173` for the backend and waits for the player to be ready.
Inside the backend container, `localhost` refers to the backend itself. Open `http://localhost:5173` from your host browser to access Dash.

For active-learning demo/correction UI, `dash-driving*` environments use the Dash iframe flow (instead of WebRTC).


## 🎯 What's next

We hope, that we can extend the functionality of RLHF-Blender in the future. In case you are interested, feel free to contribute.
Planned features are:
- Support of additional environments
- Support of additional feedback types (e.g. textual feedback)
- Further improvements of user interface, analysis capabilities
- Improved model training support

## 🛡 License

[![License](https://img.shields.io/github/license/ymetz/rlhfblender)](https://github.com/ymetz/rlhfblender/blob/master/LICENSE)

This project is licensed under the terms of the `MIT` license. See [LICENSE](https:/ymetz/rlhfblender/blob/master/LICENSE) for more details.

## 📃 Citation

```bibtex
@article{metz2023rlhf,
  title={RLHF-Blender: A Configurable Interactive Interface for Learning from Diverse Human Feedback},
  author={Metz, Yannick and Lindner, David and Baur, Rapha{\"e}l and Keim, Daniel A and El-Assady, Mennatallah},
  year={2023},
  journal={https://openreview.net/pdf?id=JvkZtzJBFQ},
  howpublished = {\url{https://github.com/ymetz/rlhfblender}}
}
```
