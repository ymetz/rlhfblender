# RLHF-Blender


> [!WARNING]  
> Right now, RLHF-Blender is still in preview mode and might therefore contain bugs or will not immediately run on each system.
> We are working on a stable release. 


Implementation for RLHF-Blender: A Configurable Interactive Interface for Learning from Diverse Human Feedback

<div align="center">

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/github/license/ymetz/rlhfblender)](https://github.com/ymetz/rlhfblender/blob/master/LICENSE)

</div>

Documentation: https://rlhfblender.readthedocs.io/en/latest/

> [!NOTE]  
> The following repository is part of of the RLHF-Blender project. The frontend is part of a separate repository: [RLHF-Blender-UI](https://github.com/ymetz/rlhfblender-ui
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
docker-compose up
```

(3. Optional: Local/Dev. Install):

```bash
cd rlhfblender
pip install -r requirements.txt
python app.py
```

and

```bash
cd rlhfblender-ui
npm install
npm run start
```

## Features

RLHF-Blender allows to configure experimental setups for RLHF-experiments based on several modular components:

- A freely configurable user interface for different feedback type interactions
- Feedback processors, handling the translation of different types of feedback, incl. meta-data, into a common format
- Adaptor to different reward models (e.g. reward model ensembles, AIRL-style models, etc.) 

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
