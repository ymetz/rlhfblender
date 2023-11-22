.. _run_experiment:

============
Installation
============

Requirements
------------

- Python 3.8 or higher
- Node.js 16 or higher
- Docker (optional)

You have the option to install rlhfblender either locally or via Docker. The Docker installation is recommended for users who do not have Python or Node.js installed on their system.


Clone the repository
--------------------

.. code-block:: bash

    git clone https://github.com/ymetz/rlhfblender.git
    cd rlhfblender
    git submodule update --init rlhfblender-ui

to get both the main repository,user interface. If you want to download both the repository and demo models, you can also run ``git clone --recurse-submodules https://github.com/ymetz/rlhfblender.git``.


Local Installation
------------------

.. code-block:: bash

    cd rlhfblender
    pip install -r requirements.txt
    python rlhfblender/app.py

and

.. code-block:: bash

    cd rlhfblender-ui
    npm install
    npm run start

Docker Installation
-------------------

.. code-block:: bash

    docker-compose up

Usage
-----

After starting the aplication:
The user interface is available at http://localhost:3000.
The API is available at http://localhost:8080/docs.


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
