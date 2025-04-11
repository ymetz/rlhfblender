"""Setup for imitation: a reward and imitation learning library."""

from setuptools import find_packages, setup

setup(
    name="imitation",
    version="1.0.1",  # Static version instead of setuptools_scm
    description="Implementation of modern reward and imitation learning algorithms.",
    author="Center for Human-Compatible AI and Google",
    python_requires=">=3.8.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"imitation": ["py.typed", "scripts/config/tuned_hps/*.json"]},
    install_requires=[
        "gymnasium[classic-control]>=1.0.0",
        "matplotlib",
        "numpy>=1.15",
        "torch>=2.0.0",
        "tqdm",
        "rich",
        "scikit-learn>=0.21.2",
        "seals~=0.2.1",
        "stable-baselines3>=2.5.0",
        "tensorboard>=1.14",
        "huggingface_sb3~=3.0",
        "optuna>=3.0.1",
        "datasets>=2.8.0",
    ],
    url="https://github.com/HumanCompatibleAI/imitation",
    license="MIT",
)