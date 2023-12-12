.. _add_new_experiment:

========================================
Customization and adding new environments
========================================


Adding new environments/experiments
-----------------------------------

To add a new environment, you need to register it to the internal env registry (handled via a sqlite database).

.. code-block:: python

    from rlhfblender.register import register_env, register_experiment

    # register a pre-existing gymnasium environment
    register(
        id='CartPole-v1',
    )
    
    # register a custom environment with a local entry point
    register_env(
        id='MyEnv-v0',
        entry_point='rlhfblender.envs:MyEnv',
        display_name='My Environment', # optional display name, by default it is the id
        ..gym kwargs..
    )

    register_experiment(
        exp_name='MyExperiment',
        pre_generated_data=True,
        project_name='MyProject', # optional project name, by default it is "Multi Feedback Experiments", can be used to group experiments
        env_id='MyEnv-v0', # the id of the environment you registered above/ or any already registered environment
        ..other kwargs..
    )

or via the command line:

.. code-block:: bash

    #! register a pre-existing gymnasium environment
    python -m rlhfblender.register --env Cartpole-v1

    #! register a custom environment with a local entry point & additional packages to load
    python -m rlhfblender.register --env MyEnv-v0 --env-gym-entrypoint rlhfblender.envs:MyEnv --additional-gym-packages package1 package2

    #! register an environment and an experiment simultaneously
    python -m rlhfblender.register --env MyEnv-v0 --exp MyExperiment --project MyProject --exp-kwargs framework: StableBaselines3 algorithm: PPO ... --env-kwargs truncate: True max_episode_steps: 1000 ...

    #! register a new experiment with an already registered environment (and different environment kwargs)
    python -m rlhfblender.register --exp MyExperiment --project MyProject --exp_env MyEnv-v0 --exp-kwargs framework: StableBaselines3 algorithm: PPO ... --env-kwargs truncate: False max_episode_steps: 500 ...

To create customized environments, check the `StableBaselines3 documentation <https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html>`_. 

The following arguments are necessary for live training and inference. Otherwise, we can train with pre-generated data (see below).
    - ``framework``: Optional - The framework used for the experiment, by default it is "StableBaselines3" (necessary for live training and inference)
    - ``algorithm``: Optional - The algorithm used for the experiment, by default it is "PPO" (necessary for training and inference)
    - ``path``: Optional - The path to the experiment folder, by default it is ``èxp_name`` "experiments"
    - ``hyperparams``: Optional - The hyperparameters used for the experiment, by default the default algorithm parameters are used
    - ``parallel_envs``: Optional - The number of parallel environments used for the experiment, by default it is 1
    - ``seed``: Optional - The seed used for the experiment, by default it is -1

If you have registered an environment and experiment, you should be able to see it in the configuration page of the UI.
Depending on your choice, you can either generate data and use it for analysis or passive reward model training.

Alternatively, if you have provided the necessary arguments for live training and inference, you can configure live-training in the UI.


Generate data 
-------------

The easiest way to generate the data is to use the ``generate_data.py`` script running inference with a trained model. You can run it with the following command:

.. code-block:: bash

    #! generate data for a pre-registered experiment and environment (with a random policy)
    python -m rlhfblender.generate_data --exp MyExperiment --random

    #! generate data for a pre-registered environment and create a new experiment (with a random policy)
    python -m rlhfblender.generate_data --env MyEnv-v0 --exp MyNewEnvironment --random -n-episodes 10

    #! generate data for a pre-registered environment and use checkpoints for inference
    python -m rlhfblender.generate_data --env MyEnv-v0 --exp MyNewEnvironment --model-path path/to/checkpoints --checkpoints 100000 200000 300000


Using pre-generated data
----------------------

In case you want to use pre-generated data, you need to put the data in the ``data`` folder. The data needs to be in the following format:


| data
| ├── renders
| │   ├── MyExperiment
| │   │   ├── subfolder1
| │   │   │   ├── 0.mp4
| │   │   │   ├── 1.mp4
| │   │   ...
| ├── thumbnails
| │   ├── MyExperiment
| │   │   ├── subfolder1
| │   │   │   ├── 0.png
| │   │   │   ├── 1.png
| │   │   ...
| ├── episodes
| │   ├── MyExperiment
| │   │   ├── subfolder1
| │   │   │   ├── 0.npz
| │   │   │   ├── 1.npz
| │   │   ...
| ├── rewards
| │   ├── MyExperiment
| │   │   ├── subfolder1
| │   │   │   ├── 0.npy
| │   │   │   ├── 1.npy
| │   │   ...
| ├── uncertainty
| │   ├── MyExperiment
| │   │   ├── subfolder1
| │   │   │   ├── 0.npy
| │   │   │   ├── 1.npy
| │   │   ...


Adding action labels and images
-------------------------------

When registering an environment, you can also add action labels and images. Text labels can be displayed in the UI
and might help users to give proper feedback, e.g. for demonstrations.

Action Labels are currently supported for flat action spaces (e.g. discrete actions or Box actions with a single dimension).
To register the action labels, you can call the ``get_action_dims`` call followed by the ``set_action_labels`` call:

.. code-block:: bash

    #! get the action dimensions for a pre-registered environment
    python -m rlhfblender.register --env MyEnv-v0 --get-action-dims

    #! Expected output:
    #! Action dimensions: 1

    #! set the action labels for a pre-registered environment
    python -m rlhfblender.register --env MyEnv-v0 --set-action-labels up down left right

These action labels will be displayed in the UI and can be used for demonstrations. You can change them by calling the ``set_action_labels`` call again.

To add visual action labels, you need to put the data in the ``data`` folder. The data needs to be in the following format:

| data
| ├── action_labels
| │   ├── MyEnv-v0
| │   │   ├── up.npy
| │   │   ├── down.npy
| │   │   ├── left.npy
| │   │   ├── right.npy


Running live training and inference
-----------------------------------


Comming soon

