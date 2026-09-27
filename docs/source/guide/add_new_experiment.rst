.. _add_new_experiment:

========================================
Generate Data and Add New Experiments
========================================

For complete Dash and MiniGrid examples, use the
:download:`new-project walkthrough <../../new_project.md>`.
This reference describes the current ``dash_driving`` branch.

Run from the repository root
----------------------------

The entry point is ``python -m rlhfblender.generate_data``. The ``--env`` argument
is required even when ``--exp`` already exists. Set ``RLHFBLENDER_DB_HOST`` to the
same database used by the API before running registration or generation.

For Docker, Compose sets the database URL automatically. Open an activated shell:

.. code-block:: bash

    docker-compose exec backend micromamba run -n base bash

The remaining examples run in that shell, or from the repository root in an
activated local Python environment. For local use of ``data/rlhfblender.db``, export
``RLHFBLENDER_DB_HOST=sqlite:///data/rlhfblender.db`` in both terminals or start the
API with ``--db-host sqlite:///data/rlhfblender.db``. An explicit ``--db-host`` takes
precedence over the environment variable.

Register metadata
-----------------

.. code-block:: bash

    python -m rlhfblender.generate_data \
      --project "CartPole tutorial" \
      --exp cartpole-tutorial-random \
      --env CartPole-v1 \
      --random \
      --register-only

``--project`` creates the project automatically and links the environment and
experiment, including records that already exist. ``--register-only`` creates no
episodes or checkpoints. Use ``--exp`` for a named baseline; it is required when
configuring an experiment. Without ``--exp``, registration-only creates just the
environment/project; generation creates an automatically named experiment.

Environment IDs and experiment names are global. Existing environment metadata
and experiment policy/model fields are preserved. Explicit configuration flags
merge into the existing experiment; omitted configuration is preserved. Reusing
an experiment name with a different environment is rejected. Use a new experiment
name for independent settings or a different policy.

Collect a random baseline
-------------------------

.. code-block:: bash

    python -m rlhfblender.generate_data \
      --project "CartPole tutorial" \
      --exp cartpole-tutorial-random \
      --env CartPole-v1 \
      --random \
      --num-episodes 3 \
      --max-steps-per-episode 100

The recorded checkpoint is ``-1`` (shown as **Random** in the UI). This records a
random policy; it does not train an agent or reward model. Reusing an experiment
and checkpoint can overwrite existing artifacts; use a new experiment name to
keep a separate dataset.

Registration and generation options
-----------------------------------

- ``--env``: required Gymnasium environment ID.
- ``--project``: project name; default ``RLHF-Blender``.
- ``--exp``: experiment name; use an explicit, distinct name for each baseline.
- ``--env-gym-entrypoint``: import path for a custom environment, such as
  ``rlhfblender.data_collection.dash_driving_gym_env:DashDrivingGymEnv``.
- ``--additional-gym-packages``: modules to import for Gym registrations, such as
  ``minigrid``. Install these packages separately.
- ``--env-display-name`` and ``--env-description``: human-readable environment metadata.
- ``--action-names``: labels in action-index order. For Dash, use ``steer gas brake``;
  for MiniGrid, ``left right forward pickup drop toggle done``.
- ``--register-only``: register metadata without recording.
- ``--env-config``: JSON object merged recursively into ``environment_config``;
  supports typed and nested values. Requires ``--exp``.
- ``--env-wrapper``: wrapper import path, stored at the top level of
  ``environment_config``; requires ``--exp``.
- ``--random``: record a random policy.
- ``--num-episodes``: number of episodes, default ``10``.
- ``--max-steps-per-episode``: recording limit per episode, default ``200``.
- ``--consistent-start-state``: request persistent initial states where supported by
  the environment; this does not add exact state restoration to Dash.
- ``--model-path``, ``--algorithm``, ``--framework``, and ``--checkpoints``: trained
  policy loading options. Use the format expected by the selected framework.

``--env-kwargs`` accepts ``key:value`` pairs as strings, preserves colons in values,
and requires ``--exp``. For typed or nested constructor arguments, use
``--env-config '{"env_kwargs":{"max_steps":100}}'``. Use ``--env-wrapper`` for a
wrapper, rather than passing it through ``--env-kwargs``.

Configure an existing experiment
--------------------------------

For Dash:

.. code-block:: bash

    python -m rlhfblender.generate_data \
      --project "Dash tutorial" --exp dash-tutorial-random --env dash-driving-v0 \
      --env-config '{"env_kwargs":{"default_reset_options":{"scenarioName":"rough_road","startMode":"manual","clearRecording":true}}}' \
      --register-only

For MiniGrid:

.. code-block:: bash

    python -m rlhfblender.generate_data \
      --project "MiniGrid tutorial" --exp minigrid-tutorial-random --env MiniGrid-Empty-5x5-v0 \
      --env-wrapper minigrid.wrappers.ImgObsWrapper \
      --register-only

Nested dictionaries are merged; supplied lists and scalars replace existing
values. Dedicated ``--env-wrapper`` and ``--env-kwargs`` flags take precedence
over the corresponding values in ``--env-config``. These flags also work during
initial registration. No separate project-membership update is needed.

Generated files
---------------

The pipeline writes ``data/<kind>/<environment>/<environment>_<experiment-id>_<checkpoint>/``:

- ``episodes/``: ``benchmark_<episode>.npz`` observations, actions, and episode data.
- ``renders/``: ``<episode>.mp4`` recordings.
- ``thumbnails/``: ``<episode>.jpg`` preview images.
- ``rewards/``: ``rewards_<episode>.npy`` per-step rewards.
- ``uncertainty/`` and ``env_states/``: additional recorded data where available.

Intermediate ``data/saved_benchmarks/`` files are removed after successful
processing. With Docker, all these paths are under ``remote_data/`` on the host.
When transferring pre-generated data, keep the associated database records and
matching directory names; experiment IDs are part of the paths.

The CLI exits with a nonzero status on recording errors. It prints
``Data generation finished.`` only after successful processing. Check the generated
artifacts before opening the project in the UI.
