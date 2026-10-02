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

## Start a new Dash or MiniGrid project

**[Step-by-step guide: create a Dash or MiniGrid project](docs/new_project.md)**

This is the guide to send to a colleague using the **`dash_driving` branch**. It includes the exact commands for Docker and local installation, project/environment/experiment registration, sample recording, and opening the result in the UI.

The workflow is:

1. Check out this branch and initialize the `rlhfblender-ui` submodule.
2. Start the backend, frontend, and bundled Dash player.
3. Register a named project and experiment with `python -m rlhfblender.generate_data --project ... --exp ... --env ... --random --register-only`.
4. Set a Dash scenario with `--env-config` or the MiniGrid wrapper with `--env-wrapper`. The guide includes these flags in registration; a separate configuration step is optional.
5. Generate random episodes using the same project and experiment names.
6. Select **Project → Experiment → Checkpoint: Random** in the UI, then **Save Setup** to create a study link.

A project is created automatically during registration. A saved study setup is a separate record containing the selected project, experiment, checkpoint, and feedback configuration. Registration alone does not generate episodes or train a model.

| Environment | Code / dependency | Registration |
| --- | --- | --- |
| Dash driving | Bundled in `dash-rl-env/`; Compose service `dash-driving` | `dash-driving-v0`, entry point `rlhfblender.data_collection.dash_driving_gym_env:DashDrivingGymEnv` |
| MiniGrid | Optional Python package `minigrid` | For example `MiniGrid-Empty-5x5-v0`, with `--additional-gym-packages minigrid` |

Dash does not require a separate clone on this branch. MiniGrid requires an additional package installation and `ImgObsWrapper` configuration; both are covered in the guide. The frontend remains a Git submodule. Trained model files are optional for these random-baseline examples.

## Installation on this branch

### Docker

Install Git and Docker with Compose, and start your Docker daemon. These examples use `docker-compose`; use `docker compose` instead if that is how Compose is installed on your machine.

```bash
git clone --branch dash_driving https://github.com/ymetz/rlhfblender.git
cd rlhfblender
git submodule update --init rlhfblender-ui
```

Compose preconfigures the shared database URL, the backend’s Dash address, and the frontend’s browser-facing Dash URLs. No manual environment exports or `.env.local` file are needed for local Docker use.

```bash
docker-compose up -d --build
docker-compose ps
```

| Interface | URL |
| --- | --- |
| RLHF-Blender | http://localhost:3000 |
| Backend API documentation | http://localhost:8080/docs |
| Dash simulator | http://localhost:5173 |

For registration and data generation, open a backend shell:

```bash
docker-compose exec backend micromamba run -n base bash
```

Then follow the [Dash or MiniGrid registration recipe](docs/new_project.md#3-register-your-project-environment-and-experiment) in that shell. The shell inherits `RLHFBLENDER_DB_HOST=sqlite:///data/rlhfblender.db` from Compose, matching the API. Existing records are automatically linked to the requested project.

Compose mounts `remote_data/` on the host as `data/` in the backend. The shared database is `remote_data/rlhfblender.db`; generated episodes, videos, rewards, and thumbnails are stored under `remote_data/`. Models, training artifacts, and logs have their own mounts. These directories must be writable where needed by the container user.

The backend reaches Dash at `http://dash-driving:5173`, configured by Compose. The browser uses `http://localhost:5173`, supplied through frontend build arguments. For remote access, see [Docker overrides](docs/new_project.md#docker-settings-and-overrides). Compose waits for Dash to be healthy and clears the image's invalid `DISPLAY` setting for headless WebGL. Gym captures suppress the welcome modal and put the HUD below the simulator.

For Colima, start the VM with `colima start` first; the same Compose commands and service URLs apply. Dash's health check tests its service hostname, including DNS and Vite host validation. If you see `ERR_NAME_NOT_RESOLVED` or HTTP 403, follow the [Docker/Colima connectivity checks](docs/new_project.md#dockercolima-connectivity-checks).

The backend build uses the checked-in `.dockerignore` allowlist. Edits to included source files are packaged on rebuild; new source files require corresponding allowlist entries. No regeneration is needed merely to follow this guide. Dash and the frontend have separate build contexts.

Newly saved setup/configuration JSON files under `configs/` are **not bind-mounted** by Compose. See the [guide's backup instructions](docs/new_project.md#6-open-the-project-and-save-a-study-setup) before recreating a container containing a study you want to keep.

### Local development

Use Python 3.10+ and Node compatible with the frontend (`^20.19.0` or `>=22.12.0`). The [local installation instructions](docs/new_project.md#local-alternative-without-docker) cover the Python environment, frontend, Dash server, Playwright browser, and matching database settings. All registration and generation commands are the same once the shell is configured.

## Features

RLHF-Blender supports configurable interfaces for collecting several types of human feedback, feedback processors, and reward-model integrations. The project recipes above create random baseline recordings to start exploring these interfaces; training and study configuration are subsequent steps.

## Further documentation

- [New Dash or MiniGrid project](docs/new_project.md): complete branch-specific walkthrough and troubleshooting.
- [Registration and data-generation reference](docs/source/guide/add_new_experiment.rst): CLI flags and artifact layout.
- [Experiment setup](docs/source/guide/setup_experiment.rst): configuration mode and study links.
- [Dash simulator documentation](dash-rl-env/README.md): scenarios, simulator controls, and Python integration.

The hosted documentation may describe older revisions. Use the checked-in guide for this branch.

## 🎯 What's next

We hope, that we can extend the functionality of RLHF-Blender in the future. In case you are interested, feel free to contribute.
Planned features are:

- Support of additional environments
- Support of additional feedback types (e.g. textual feedback)
- Further improvements of user interface, analysis capabilities
- Improved model training support

## 🛡 License

[![License](https://img.shields.io/github/license/ymetz/rlhfblender)](https://github.com/ymetz/rlhfblender/blob/master/LICENSE)

This project is licensed under the terms of the `MIT` license. See [LICENSE](LICENSE) for more details.

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
