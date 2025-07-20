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
    install_requires=[],  # install with empty requirements, as we do not want to train models
    # but only use imitation as a library, all necessary rquirements are already installed by other packages
    url="https://github.com/HumanCompatibleAI/imitation",
    license="MIT",
)
