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

```bash
git clone https://github.com/ymetz/rlhfblender.git
cd rlhfblender
git submodule update --init rlhfblender-ui
```
to get both the main repository,user interface. If you want to download both the repository and demo models, you can also run ```git clone --recurse-submodules https://github.com/ymetz/rlhfblender.git```.


Local Installation
------------------

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

Docker Installation
-------------------

```bash
docker-compose up
```

Usage
-----

After starting the aplication:
The user interface is available at http://localhost:3000.
The API is available at http://localhost:8080/docs.