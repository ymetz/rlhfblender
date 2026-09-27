.. _installation:

============
Installation
============

For this branch, use the :download:`Dash and MiniGrid project guide <../../new_project.md>`
for complete, ordered Docker or local installation commands. Dash is bundled under
``dash-rl-env/`` and is included in Compose; only ``rlhfblender-ui`` needs submodule
initialization for the random-baseline walkthrough.

Requirements
------------

Docker installation requires Git and a running Docker daemon with Compose.
For local installation, use Python 3.10+ and Node matching the frontend's
``^20.19.0 || >=22.12.0`` requirement. MiniGrid is an optional Python dependency.

Docker startup
--------------

.. code-block:: bash

    git clone --branch dash_driving https://github.com/ymetz/rlhfblender.git
    cd rlhfblender
    git submodule update --init rlhfblender-ui

Compose supplies the database and Dash URL defaults, including the frontend's
build-time settings. No manual exports or frontend ``.env.local`` file are needed
for local Docker use. Start the services:

.. code-block:: bash

    docker-compose up -d --build
    docker-compose ps

Use ``docker compose`` instead if your machine uses the Compose plugin.
The UI is at http://localhost:3000, the API at http://localhost:8080/docs,
and Dash at http://localhost:5173.

Compose sets ``RLHFBLENDER_DB_HOST=sqlite:///data/rlhfblender.db`` for the API and
registration commands, including new backend shells opened with ``exec``.
For remote browser access, set ``VITE_DASH_PLAYER_URL`` and
``VITE_DASH_PLAYER_ORIGIN`` in the repository-root ``.env`` and rebuild; see the walkthrough.
The Docker data directory maps to ``remote_data/`` on the host. Newly saved
``configs/`` files need a separate backup before container recreation.

Local startup
-------------

The walkthrough covers installing the package with ``pip install --only-binary=av -e .``,
installing Playwright's Chromium browser, and starting the backend, frontend, and
Dash player in separate terminals. Use the same working directory and database
path for the API and registration CLI. MiniGrid additionally needs its package and
observation wrapper; installing the base package alone is insufficient.

Kubernetes Deployment
---------------------

The following is an example of a Kubernetes deployment. The deployment contains two containers, one for the API and one for the user interface.

Backend/API

.. code-block:: yaml

    app:
    image: 
        repository: "<YOUR_REPOSTIRY>/rlhfblender-backend" 
        tag: latest
        
    replicaCount: 1

    regcred: regcred-rlworkbench
    port: 8080

    livenessProbe: "null"

    readinessProbe: "null"

    startupProbe: "null"

    requests:
        cpu: 100m
        memory: 500Mi
        # Optional: If you want to use a GPU
        gpu: 1
    limits:
        cpu: 4000m
        memory: 12Gi

    gpu:
        devices: 0,...

    extraEnv:
        BACKEND_PORT: "{{ .Values.app.port }}"

    ingress:
        enabled: false


Frontend/UI

.. code-block:: yaml

    app:
    image:
        repository: '${CI_REGISTRY_IMAGE}/frontend'
        tag: '$VERSION'
    replicaCount: $REPLICA_COUNT
    regcred: regcred-rlworkbench
    port: 3000

    requests:
        cpu: 100m
        memory: 250Mi
    limits:
        cpu: 1000m
        memory: 4Gi

    livenessProbe: |
        httpGet:
        path: "/"
        port: {{ .Values.app.port }}
        scheme: HTTP
        initialDelaySeconds: 60
        timeoutSeconds: 10
        periodSeconds: 30
        failureThreshold: 3
        successThreshold: 1

    readinessProbe: |
        httpGet:
        path: "/"
        port: {{ .Values.app.port }}
        scheme: HTTP
        initialDelaySeconds: 60
        timeoutSeconds: 10
        periodSeconds: 30
        failureThreshold: 3
        successThreshold: 1

    extraEnv:
        # Choose hostname
        BACKEND_HOST: 'rlhfblender-backend'
        BACKEND_PORT: '8080'
        # Choose URL
        HOST: 'rlhfblender.example.com'

    ingress:
        enabled: true
        url: 'rlhfblender.example.com'
        extraAnnotations: |
        nginx.ingress.kubernetes.io/proxy-body-size: 8m
