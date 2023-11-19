import os

from setuptools import find_packages, setup

with open(os.path.join("rlhfblender", "version.txt")) as file_handler:
    __version__ = file_handler.read().strip()


long_description = """

# RLHF-Blender

RLHF-Blender is an library to train reward models from diverse human feedback. It encompasses both a Python library and a TypeScript-based user interface for collecting human feedback.

## Links

Repository:
https://github.com/ymetz/rlhfblender

Documentation:
https://rlhfblender.readthedocs.io/en/latest/

RLHF-Blender UI:
https://github.com/ymetz/rlhfblender-ui
"""  # noqa:E501

setup(
    name="rlhfblender",
    packages=[package for package in find_packages() if package.startswith("rlhfblender")],
    package_data={"rlhfblender": ["py.typed", "version.txt"]},
    install_requires=[
        "gymnasium[atari,accept-rom-license,mujoco]>=0.29.1,<0.30",
        "minigrid>=2.0.0",
        "highway-env>=1.8.2",
        # "safety-gymnasium>=1.0.0",
        "stable-baselines3>=2.0.0",
        "sb3-contrib>=2.0.0",
        "imitation>=1.0.0",
        "numpy>=1.20",
        "torch>=1.13",
        "torchvision",
        "opencv-python>=4.8",
        "fastapi",
        "uvicorn",
        "databases[sqlite]",
        "python-multipart",
    ],
    extras_require={
        "tests": [
            # Run tests and coverage
            "pytest",
            "pytest-cov",
            "pytest-env",
            "pytest-xdist",
            # Type check
            "mypy",
            # Lint code and sort imports (flake8 and isort replacement)
            "ruff>=0.0.288",
            # Reformat
            "black>=23.9.1,<24",
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
    description="Implementation for RLHF-Blender: A Configurable Interactive Interface for Learning from Diverse Human Feedback",
    author="Yannick Metz",
    url="https://github.com/ymetz/rlhfblender",
    author_email="yannick.metz@uni-konstanz.de",
    keywords="react reinforcement-learning experimentation human-ai-interaction reinforcement-learning-from-human-feedback python",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
    python_requires=">=3.8",
    # PyPI package information.
    project_urls={
        "Code": "https://github.com/ymetz/rlhfblender",
        "Documentation": "https://rlhfblender.readthedocs.io",
        "Changelog": "https://rlhfblender.readthedocs.io/en/main/misc/changelog.html",
        "RLHF-Blender UI": "https://github.com/ymetz/rlhfblender-ui",
        "RLHF-Blender Models": "https://github.com/ymetz/rlhfblender_demo_models",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
