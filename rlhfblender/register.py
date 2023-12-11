import argparse
import asyncio
import os
import time
from typing import Dict, Optional

from databases import Database

import rlhfblender.data_collection.environment_handler as environment_handler
import rlhfblender.data_handling.database_handler as db_handler
from rlhfblender.data_models.global_models import Environment, Experiment, Project
from rlhfblender.utils.utils import StoreDict

database = Database(f"sqlite:///./{os.environ.get('RLHFBLENDER_DB_HOST', 'test.db')}")


async def init_db():
    # Make sure all database tables exist
    await db_handler.create_table_from_model(database, Project)
    await db_handler.create_table_from_model(database, Experiment)
    await db_handler.create_table_from_model(database, Environment)


async def add_to_project(project: str = "RLHF-Blender", env: Optional[str] = None, exp: Optional[str] = None):
    """Add an environment or experiment to a project.

    Args:
        project (str, optional): The project name. Defaults to "RLHF-Blender".
        env (Optional[str], optional): The environment id. Defaults to None.
        exp (Optional[str], optional): The experiment name. Defaults to None.
    """

    # check if project exists
    if not await db_handler.check_if_exists(database, Project, key=project, key_column="project_name"):
        # register new project
        await db_handler.add_entry(
            database,
            Project,
            Project(project_name=project, created_timestamp=int(time.time())).model_dump(),
        )
        existing_envs = []
        existing_exps = []
    else:
        # get the project and existing envs and exps
        project_obj: Project = await db_handler.get_single_entry(database, Project, key=project, key_column="project_name")
        existing_envs = project_obj.project_environments
        existing_exps = project_obj.project_experiments

    # now add env or exp to project
    if env is not None:
        await db_handler.update_entry(
            database,
            Project,
            key=project,
            key_column="project_name",
            data={"project_environments": [*existing_envs, env]},
        )
    if exp is not None:
        await db_handler.update_entry(
            database,
            Project,
            key=project,
            key_column="project_name",
            data={"project_experiments": [*existing_exps, exp]},
        )


async def register_env(
    id: str = "Cartpole-v1",
    entry_point: Optional[str] = "",
    display_name: str = "",
    env_kwargs: Optional[Dict] = None,
    project: str = "RLHF-Blender",
):
    """Register an environment in the database.

    Args:
        id (str, optional): The environment id. Defaults to "Cartpole-v1".
        entry_point (Optional[str], optional): The entry point for the environment class. Defaults to "".
        kwargs (Optional[Dict], optional): The kwargs for the environment class. Defaults to None.
    """
    env_name = display_name if display_name != "" else id
    additional_gym_packages = env_kwargs.get("additional_gym_packages", []) if env_kwargs is not None else []
    env: Environment = environment_handler.initial_registration(
        env_id=id, entry_point=entry_point, additional_gym_packages=additional_gym_packages
    )

    env.env_name = env_name
    for key, value in env_kwargs.items():
        setattr(env, key, value)

    if not await db_handler.check_if_exists(database, Environment, key=id, key_column="registration_id"):
        await db_handler.add_entry(
            database,
            Environment,
            env.model_dump(),
        )
        await add_to_project(project=project, env=id)
        print(f"Registered experiment {args.exp} in project {args.project}")

    else:
        print(f"Environment with id {id} already exists. Skipping registration.")


async def register_experiment(
    exp_name: str,
    env_id: Optional[str] = "Cartpole-v1",
    path: Optional[str] = "",
    exp_kwargs: Optional[Dict] = None,
    project: Optional[str] = "RLHF-Blender",
):
    """Register an experiment in the database.

    Args:
        env_id (str, optional): The environment id. Defaults to "Cartpole-v1".
        path (str, optional): The path to the experiment. Defaults to "".
        benchmark_type (str, optional): The benchmark type. Defaults to "random".
        benchmark_id (int, optional): The benchmark id. Defaults to -1.
        checkpoint_step (int, optional): The checkpoint step. Defaults to -1.
        n_episodes (int, optional): The number of episodes. Defaults to 1.
        force_overwrite (bool, optional): Force overwrite. Defaults to False.
        render (bool, optional): Render. Defaults to True.
        deterministic (bool, optional): Deterministic. Defaults to False.
        reset_state (bool, optional): Reset state. Defaults to False.
        split_by_episode (bool, optional): Split by episode. Defaults to False.
    """
    exp = Experiment(exp_name=exp_name, env_id=env_id, path=path, **exp_kwargs)

    if not await db_handler.check_if_exists(database, Experiment, key=exp_name, key_column="exp_name"):
        await db_handler.add_entry(
            database,
            Experiment,
            exp.model_dump(),
        )
        await add_to_project(project=project, exp=exp_name)
        print(f"Registered environment {args.env} in project {args.project}")

    else:
        print(f"Experiment with name {exp_name} already exists. Skipping registration.")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Register an environment or experiment in the database.")
    argparser.add_argument("--env", type=str, help="The environment id.", default="")
    argparser.add_argument("--exp", type=str, help="The experiment name.", default="")

    # project as an optional argument (default: RLHF-Blender)
    argparser.add_argument(
        "--project",
        type=str,
        help="(Optional) The project name. Defaults to RLHF-Blender.",
        default="RLHF-Blender",
    )

    # args for env registration
    argparser.add_argument(
        "--env-gym-entrypoint",
        type=str,
        help="(Optional) The gym entry point for the environment. Relevant for local custom environments",
        default="",
    )
    argparser.add_argument(
        "--env-display-name",
        type=str,
        help="(Optional) The display name for the environment. Relevant for local custom environments",
        default="",
    )
    argparser.add_argument(
        "--env-kwargs",
        type=str,
        nargs="+",
        action=StoreDict,
        help='Environment Kwargs (e.g. description:"An optional env description")',
    )

    # args for exp registration
    argparser.add_argument(
        "--exp-env-id",
        type=str,
        help="(Optional) A separate environment-id for the experiment. By default the environment-id is used.",
        default="",
    )
    argparser.add_argument(
        "--exp-path",
        type=str,
        help='(Optional) The path to the experiment. Defaults to "".',
        default="",
    )
    argparser.add_argument(
        "--exp-kwargs",
        type=str,
        nargs="+",
        action=StoreDict,
        help='Experiment Kwargs (e.g. description:"An optional exp description")',
    )

    args = argparser.parse_args()

    asyncio.run(init_db())

    if args.env != "":
        env_kwargs = args.env_kwargs if args.env_kwargs is not None else {}
        asyncio.run(
            register_env(
                args.env,
                entry_point=args.env_gym_entrypoint,
                display_name=args.env_display_name,
                env_kwargs=env_kwargs,
                project=args.project,
            )
        )
    if args.exp != "":
        exp_kwargs = args.exp_kwargs if args.exp_kwargs is not None else {}
        exp_env_id = args.exp_env_id if args.exp_env_id != "" else args.env
        asyncio.run(
            register_experiment(args.exp, env_id=exp_env_id, path=args.exp_path, exp_kwargs=exp_kwargs, project=args.project)
        )
