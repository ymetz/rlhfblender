.. _add_new_experiment:

========================================
Customization and adding new environments
========================================


Adding new environments
-----------------------

To add a new environment, you need to register it to the internal env registry (handled via a sqlite database).

.. code-block:: python

    from rlhfblender.register import register_env, register_experiment

    register_env(
        id='MyEnv-v0',
        entry_point='rlhfblender.envs:MyEnv',
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

    python -m rlhfblender.register --env_id MyEnv-v0 --entry_point rlhfblender.envs:MyEnv --exp_name MyExperiment --pre_generated_data True

To register customized environments, check the `stable-baselienes3 documentation <https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html>`_. 

Optional arguments for the experiment registration are:
    - ``pre_generated_data``: If you want to use pre-generated data, by default it is True (requires data in the ``data`` folder, for details see following section)
    - ``project_name``: The name of the project, by default it is "Multi Feedback Experiments", can be used to group experiments

The following arguments are necessary for live training and inference. Otherwise, we can train with pre-generated data.
    - ``framework``: Optional - The framework used for the experiment, by default it is "StableBaselines3" (necessary for live training and inference)
    - ``path``: Optional - The path to the experiment folder, by default it is ``èxp_name`` "experiments"
    - ``algorithm``: Optional - The algorithm used for the experiment, by default it is "PPO" (necessary for training and inference)
    - ``hyperparams``: Optional - The hyperparameters used for the experiment, by default it is {} (necessary for training and inference)
    - ``parallel_envs``: Optional - The number of parallel environments used for the experiment, by default it is 1 (necessary for training and inference)
    - ``seed``: Optional - The seed used for the experiment, by default it is -1 (necessary for training and inference)

Then, you can use the environment in the same way as the other environments:


Using pre-generated data
----------------------

In case you want to use pre-generated data, you need to put the data in the ``data`` folder. The data needs to be in the following format:

.. code-block:: python
    data
    ├── renders
    │   ├── MyExperiment
    │   │   ├── subfolder1
    │   │   │   ├── 0.mp4
    │   │   │   ├── 1.mp4
    │   │   ...
    ├── thumbnails
    │   ├── MyExperiment
    │   │   ├── subfolder1
    │   │   │   ├── 0.png
    │   │   │   ├── 1.png
    │   │   ...
    ├── episodes
    │   ├── MyExperiment
    │   │   ├── subfolder1
    │   │   │   ├── 0.npz
    │   │   │   ├── 1.npz
    │   │   ...
    ├── rewards
    │   ├── MyExperiment
    │   │   ├── subfolder1
    │   │   │   ├── 0.npz
    │   │   │   ├── 1.npz
    │   │   ...
    ├── uncertainty
    │   ├── MyExperiment
    │   │   ├── subfolder1
    │   │   │   ├── 0.npz
    │   │   │   ├── 1.npz
    │   │   ...


The easiest way to generate the data is to use the ``generate_data.py`` script running inference with a trained model. You can run it with the following command:

.. code-block:: bash

    python generate_data.py --exp_name MyExperiment --env_id MyEnv-v0 --model_path path/to/model.zip --num_episodes 100 --num_parallel 10



Running live training and inference
-----------------------------------

Comming soon

