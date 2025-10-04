import os
import shutil

from setuptools import setup

with open(os.path.join("train_baselines", "version.txt")) as file_handler:
    __version__ = file_handler.read().strip()

# Copy hyperparams files for packaging
shutil.copytree("hyperparams", os.path.join("train_baselines", "hyperparams"))

long_description = """
# RL Baselines3 Zoo: A Training Framework for Stable Baselines3 Reinforcement Learning Agents

See https://github.com/DLR-RM/rl-baselines3-zoo
"""
install_requires = [
]

test_requires = [
]

setup(
    name="train_baselines",
    packages=["train_baselines", "train_baselines.plots"],
    package_data={
        "train_baselines": [
            "py.typed",
            "version.txt",
            "hyperparams/*.yml",
        ]
    },
    entry_points={"console_scripts": ["train_baselines=train_baselines.cli:main"]},
    install_requires=install_requires,
    extras_require={"tests": test_requires},
    description="Adapted from: A Training Framework for Stable Baselines3 Reinforcement Learning Agents",
    author="Antonin Raffin",
    url="https://github.com/DLR-RM/rl-baselines3-zoo",
    author_email="antonin.raffin@dlr.de",
    keywords="reinforcement-learning-algorithms reinforcement-learning machine-learning "
    "gymnasium openai stable baselines sb3 toolbox python data-science",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
    python_requires=">=3.9",
    # PyPI package information.
    project_urls={
        "Code": "https://github.com/DLR-RM/rl-baselines3-zoo",
        "Documentation": "https://rl-baselines3-zoo.readthedocs.io/en/master/",
        "Changelog": "https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/CHANGELOG.md",
        "Stable-Baselines3": "https://github.com/DLR-RM/stable-baselines3",
        "RL-Zoo": "https://github.com/DLR-RM/rl-baselines3-zoo",
        "SBX": "https://github.com/araffin/sbx",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

# Remove copied files after packaging
shutil.rmtree(os.path.join("train_baselines", "hyperparams"))
