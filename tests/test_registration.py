"""
Test the CLI (registration)
"""

import os

import pytest

from rlhfblender.register import register_env, register_experiment

# Set DB environment variable (the backend still uses the DB)
os.environ["RLHFBLENDER_DB_HOST"] = "sqlite:///test_api.db"


class RegistrationTests:

    # ========= Test register_env via Python =========
    @pytest.mark.dependency(name="test_register_env")
    def test_register_env():
        """
        Test register_env via Python
        """
        os.environ["RLHFBLENDER_DB_HOST"] = "sqlite:///test_api.db"

        register_env(
            "MountainCar-v0", display_name="Mountain Car", env_description="Mountain Car environment", project="test_project"
        )

    def test_atari_register_env():
        """
        Test register_env via Python
        """
        os.environ["RLHFBLENDER_DB_HOST"] = "sqlite:///test_api.db"

        register_env(
            "Breakout-v0",
            display_name="Breakout",
            env_description="Breakout environment",
            project="test_project",
            env_kwargs={"env_wrapper": "stable_baselines3.common.atari_wrappers.AtariWrapper", "frame_stack": 4},
        )

    @pytest.mark.dependency(name="test_register_experiment")
    def test_register_experiment():
        """
        Test register_experiment via Python
        """
        register_experiment(
            "Test Experiment 1", env_id="MountainCar-v0", framework="Stable Baselines3", project="test_project"
        )

    # ========= Test register_env via CLI =========
    def test_register_env_cli():
        """
        Test register_env via CLI
        """
        os.environ["RLHFBLENDER_DB_HOST"] = "sqlite:///test_api.db"

        exit_code = os.system(
            "python -m rlhfblender.register --env MountainCar-v0 "
            "--env-display-name 'Mountain Car' "
            "--env-description 'Mountain Car environment' "
            "--project test_project"
        )
        assert exit_code == 0

    def test_atari_register_env_cli():
        """
        Test register_env via CLI
        """
        os.environ["RLHFBLENDER_DB_HOST"] = "sqlite:///test_api.db"

        exit_code = os.system(
            """python -m rlhfblender.register --env ALE/Breakout-v5 --env-display-name Breakout \
                --env-description 'Breakout environment'"""
            """--project test_project"""
        )
        assert exit_code == 0
