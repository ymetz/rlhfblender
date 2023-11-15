.. RLHF-Blender documentation master file

RLHF-Blender Docs - A Configurable Interactive Interface for Learning from Diverse Human Feedback
========================================================================

`RLHF-Blender <https://github.com/ymetz/rlhfblender>`_ is an library to train reward models from diverse human feedback.
It encompasses both a Python library and a TypeScript-based user interface for collecting human feedback.

Github repository for backend: https://github.com/ymetz/rlhfblender

Github repository for frontend: https://github.com/ymetz/rlhfblender-ui

Paper: https://arxiv.org/abs/2308.04332


Main Features
--------------

- Comprehensive backend and frontend implementations for collecting human feedback
- Implementation for diverse feedback types, including: 
    - Evaluative feedback
    - Comparative feedback
    - Demonstrative Feedback
    - Corrective Feedback
    - Description Feedback
- Highly configurable user interface for different experimental setups
- Wrappers for reward model training
- Comprenhensive logging of feedback and user interactions

RLHF-BLender is designed to be fully compatible with `gymnasium <https://github.com/Farama-Foundation/Gymnasium>`_ and `Stable Baselines3 <https://github.com/DLR-RM/stable-baselines3>`_.
A list of currently supported environments:

  - Atari
  - Minigrid/BabyAI
  - SafetyGym

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   guide/install
   guide/quickstart
   guide/setup_experiment
   guide/run_experiment
   guide/add_new_experiment

.. toctree::
  :maxdepth: 1
  :caption: Misc

  misc/changelog
  misc/projects


Citing RLHF-Blender
------------------------
To cite this project in publications:

.. code-block:: bibtex

    @article{metz2023rlhf,
    title={RLHF-Blender: A Configurable Interactive Interface for Learning from Diverse Human Feedback},
    author={Metz, Yannick and Lindner, David and Baur, Rapha{\"e}l and Keim, Daniel and El-Assady, Mennatallah},
    journal={arXiv preprint arXiv:2308.04332},
    year={2023}
    }

Contributing
------------

To any interested in making RLHF-Blender better, there are may rooms for potential improvements.
We strongly encourage and welcome your contribution.
You can check issues in the `repo <https://github.com/ymetz/rlhfblender/issues>`_.

If you want to contribute, please read `CONTRIBUTING.md <https://github.com/ymetz/rlhfblender/blob/master/CONTRIBUTING.md>`_ first.

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`search`
* :ref:`modindex`