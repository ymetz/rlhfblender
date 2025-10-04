import os

from setuptools import find_packages, setup

# Get absolute paths to dependencies
base_dir = os.path.abspath(os.path.dirname(__file__))


def get_abs_path(rel_path):
    return f"file://{os.path.join(base_dir, rel_path)}"


setup(
    name="multi-type-feedback",
    version="0.1.0",
    description="Reward Learning from Multiple Feedback Types",
    author="YANNICK Metz, Andras Geiszl",
    author_email="yannick.metz@uni-konstanz.de",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch",
        "gymnasium>=1.0.0",
        "lightning",
        "minigrid",
        "mujoco",
        "ale-py",
        "wandb",
        "sb3_contrib>=2.5.0,<3.0",
        # "highway-env",
        # "gym3",
        # "procgen @ git+https://github.com/juancroldan/procgen",
        "opencv-python",
        # "stable-baselines3",
        f"imitation @ {get_abs_path('dependencies/imitation')}",
        f"masksembles @ {get_abs_path('dependencies/masksembles')}",
        f"train_baselines @ {get_abs_path('train_baselines')}",
    ],
    python_requires=">=3.9",
)
import os

from setuptools import find_packages, setup

with open(os.path.join("rlhfblender", "version.txt")) as file_handler:
    __version__ = file_handler.read().strip()

# Get absolute paths to dependencies
base_dir = os.path.abspath(os.path.dirname(__file__))


def get_abs_path(rel_path):
    return f"file://{os.path.join(base_dir, rel_path)}"


long_description = """

# RLHF-Blender

RLHF-Blender is an library to train reward models from diverse human feedback. 
It encompasses both a Python library and a TypeScript-based user interface for collecting human feedback.

## Links

Repository:
https://github.com/ymetz/rlhfblender

Documentation:
https://rlhfblender.readthedocs.io/en/latest/

RLHF-Blender UI:
https://github.com/ymetz/rlhfblender-ui
"""

setup(
    name="rlhfblender",
    packages=[package for package in find_packages() if package.startswith("rlhfblender")],
    package_data={"rlhfblender": ["py.typed", "version.txt"]},
    install_requires=[
        # "safety-gymnasium>=1.0.0",
        "stable-baselines3==2.3.2",
        "mujoco==3.2.3",
        "rl_zoo3==2.3.2",
        #"minigrid==2.0.0",
        #"highway-env==1.8.2",
        "metaworld==3.0.0",
        "sb3-contrib==2.0.0",
        # "imitation>=1.0.0",
        "fastapi",
        "uvicorn",
        "databases[sqlite]",
        "python-multipart",
        "gspread",
        "umap-learn",
        f"multi-type-feedback @ {get_abs_path('multi-type-feedback')}",
        "aiortc==1.12.0",
        "av",
        "httpx"
    ],
    extras_require={
        "tests": [
            # Run tests and coverage
            "pytest",
            "pytest-cov",
            "pytest-env",
            "pytest-xdist",
            "pytest-dependency",
            # Type check
            "mypy",
            # Lint code and sort imports (flake8 and isort replacement)
            "ruff>=0.0.288",
            # Reformat
            "black",
        ],
        "docs": [
            "sphinx>=5,<8",
            "sphinx-autobuild",
            "sphinx-rtd-theme>=1.3.0",
            # For spelling
            "sphinxcontrib.spelling",
            # Copy button for code snippets
            "sphinx_copybutton",
        ],
    },
    description="Implementation for RLHF-Blender: A Configurable Interface for Learning from Human Feedback",
    author="Yannick Metz",
    url="https://github.com/ymetz/rlhfblender",
    author_email="yannick.metz@uni-konstanz.de",
    keywords="react reinforcement-learning experimentation "
    "human-ai-interaction reinforcement-learning-from-human-feedback python",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
    python_requires=">=3.10",
    # PyPI package information.
    project_urls={
        "Code": "https://github.com/ymetz/rlhfblender",
        "Documentation": "https://rlhfblender.readthedocs.io",
        "Changelog": "https://rlhfblender.readthedocs.io/en/main/misc/changelog.html",
        "RLHF-Blender UI": "https://github.com/ymetz/rlhfblender-ui",
        "RLHF-Blender Models": "https://github.com/ymetz/rlhfblender_model",
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
