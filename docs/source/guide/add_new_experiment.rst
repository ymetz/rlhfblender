.. _add_new_experiment:

========================================
Generate Data and Add New Experiments
========================================


Generate Data
----------------------

To generate data for your experiments and environments, you can use the ``generate_data.py`` script. This script handles both the registration of environments and experiments (if they are not already registered) and the data generation process.

Usage:

You can run the script using the following command-line options:

.. code-block:: bash

    # Generate data for a new environment with a random policy
    python generate_data.py --env MyEnv-v0 --random --num-episodes 10

    # Generate data using a trained model and specific checkpoints
    python generate_data.py --env MyEnv-v0 --exp MyExperiment --model-path path/to/checkpoints --checkpoints 100000 200000 300000

    # Generate data for a pre-registered experiment and environment with a random policy
    python generate_data.py --exp MyExperiment --random --num-episodes 10

Command-line Arguments:

    - ``--env``: The Gym environment ID (e.g., CartPole-v1). If the environment is not registered, it will be automatically registered.
    - ``--exp``: The experiment name. If not provided, a default experiment name will be created based on the environment ID and benchmark type.
    - ``--num-episodes``: The number of episodes to run for data generation. Default is 10.
    - ``--random``: Use a random agent for data generation.
    - ``--model-path``: The path to the trained model for inference. Required if not using --random.
    - ``--checkpoints``: The checkpoint steps to use from the trained model. Default is ``-1`` (latest checkpoint).
    - ``--project``: (Optional) The project name. Defaults to RLHF-Blender.

Optional Arguments for Environment Registration:

    - ``--env-gym-entrypoint``: The Gym entry point for the environment. Useful for custom environments.
    - ``--env-display-name``: The display name for the environment.
    - ``--additional-gym-packages``: Additional Gym packages to import for custom environments.
    - ``--env-kwargs``: Environment keyword arguments in the format key:value. For example: ``--env-kwargs max_episode_steps:1000``.

Example Usage:

.. code-block:: bash

    # Generate data for a custom environment with specific environment arguments
    python generate_data.py --env MyCustomEnv-v0
        --env-gym-entrypoint my_package.envs:MyCustomEnv
        --additional-gym-packages my_package
        --env-kwargs max_episode_steps:1000
        --random
        --num-episodes 10

Notes:

    If the specified environment or experiment is not registered in the internal registry (handled via a SQLite database), the script will automatically register them.
    The data generated will be stored in the data directory, organized into subdirectories for episodes, rewards, renders, and thumbnails.
    The script supports both random agents and trained agents for data generation.

Generating Data with a Trained Model:

To generate data using a trained model, specify the --model-path to your trained model directory and provide the checkpoints you wish to use.

.. code-block:: bash

    python generate_data.py --env MyEnv-v0
        --exp MyExperiment
        --model-path path/to/model
        --checkpoints 100000 200000
        --num-episodes 10

Environment Keyword Arguments:

When using ``--env-kwargs``, you can pass environment-specific arguments that will be used during environment registration and data generation.

Example:

.. code-block:: bash

    python generate_data.py --env MyEnv-v0 
        --env-kwargs max_episode_steps:1000 reward_threshold:200 
        --random 
        --num-episodes 10

Custom Environments:

For custom environments, you may need to specify the entry point and any additional packages required.

Example:

.. code-block:: bash

    python generate_data.py --env MyCustomEnv-v0 
        --env-gym-entrypoint my_package.envs:MyCustomEnv 
        --additional-gym-packages my_package 
        --random 
        --num-episodes 10

Accessing the Generated Data:

After running the script, the generated data will be available in the data directory:

    - ``data/episodes``: Contains the episode data saved as .npz files.
    - ``data/rewards``: Contains cumulative reward data for each episode.
    - ``data/renders``: Contains rendered videos of the episodes.
    - ``data/thumbnails``: Contains thumbnail images for each episode.

This data is used by the RLHF-Blender UI to display episode information, rewards, and visualizations.


Example with All Arguments:

.. code-block:: bash

    python generate_data.py --env MyCustomEnv-v0 
        --exp MyExperiment 
        --project MyProject 
        --env-gym-entrypoint my_package.envs:MyCustomEnv 
        --additional-gym-packages my_package 
        --env-display-name "My Custom Environment" 
        --env-kwargs max_episode_steps:1000 difficulty:"'hard'" 
        --model-path path/to/model 
        --checkpoints 50000 100000 
        --num-episodes 20

In this example:

    A custom environment MyCustomEnv-v0 is registered with the specified entry point and additional packages.
    Environment keyword arguments max_episode_steps and difficulty are set.
    A new experiment MyExperiment under the project MyProject is registered.
    Data is generated using the trained model at path/to/model using checkpoints at steps 50000 and 100000.
    A total of 20 episodes are generated for each checkpoint.

Troubleshooting:

    Environment Registration Errors: Ensure that custom environments are correctly installed and accessible. The ``--env-gym-entrypoint`` should point to the correct module and class.
    Model Loading Issues: Verify that the model path and checkpoints are correct and that the model files are not corrupted.
    Additional Packages: When using custom environments that require additional packages, make sure those packages are installed in your environment and listed using ``--additional-gym-packages``.


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

