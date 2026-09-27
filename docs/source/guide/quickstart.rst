.. _quickstart:

===============
Getting Started
===============

For the ``dash_driving`` branch, follow the
:download:`step-by-step Dash and MiniGrid project guide <../../new_project.md>`.
It includes Docker and local startup commands, environment registration, project
membership, MiniGrid's observation wrapper, Dash scenarios, and data generation.
The same guide is linked prominently from the repository README.

The application is available at http://localhost:3000, the API at
http://localhost:8080/docs, and the bundled Dash player at http://localhost:5173.

Project workflow
----------------

1. Start the services. Compose preconfigures the API/CLI database and Dash addresses.
2. Register the environment and a named experiment using explicit ``--project``
   and ``--exp`` values. Registration creates the project automatically.
3. Include ``--env-config`` for Dash or ``--env-wrapper`` for MiniGrid, as shown in
   the guide. The CLI configures the experiment and attaches project links.
4. Generate episodes. ``--register-only`` creates metadata but no recordings.
5. Reload the UI and select **Project**, **Experiment**, and **Checkpoint**.
   Random recordings appear as checkpoint **Random** (``-1``).
6. Select/configure **Backend Config** and **UI Config**, then click **Save Setup**.

See :ref:`add_new_experiment` for CLI details and :ref:`setup_experiment` for study links.

Storage
-------

Local runs write episodes, renders, rewards, and thumbnails under ``data/``.
Docker maps this directory to ``remote_data/`` on the host. Feedback/session logs
are stored under ``logs/``. The Docker database is ``remote_data/rlhfblender.db``.

Saved setup and configuration JSON files live under ``configs/``. The current
Compose file does not mount that directory; copy new configuration files out of
the container before recreating it. The project guide includes the backup command.
