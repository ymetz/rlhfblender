#!/usr/bin/env python
"""
Run 5 phases of simulated DynamicRLHF training and save output in a format
compatible with the rlhfblender UI.

Must be run from the project root:
    python scripts/run_simulated_phases.py \\
        --env metaworld-sweep-into-v3 \\
        --algorithm ppo \\
        --exp-name sweep-sim-run1 \\
        --num-phases 5

Prerequisites:
    pip install metaworld  (for metaworld envs)
    The rlhfblender.db database must exist (start the server once to create it).
"""

import argparse
import asyncio
import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from databases import Database

# Must run from project root so all relative data paths resolve correctly
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from multi_type_feedback.dynamic_rlhf import DynamicRLHF
from multi_type_feedback.feedback_oracle import FeedbackOracle
from multi_type_feedback.generate_feedback import generate_feedback
from multi_type_feedback.utils import TrainingUtils
from rlhfblender.data_handling.database_handler import add_entry, get_single_entry
from rlhfblender.data_models.global_models import Experiment
from train_baselines.exp_manager import ExperimentManager

DATABASE_URL = os.environ.get("RLHFBLENDER_DB_HOST", "sqlite:///rlhfblender.db")

# Feedback types that work without an expert model.
# Add "demonstrative" only if you have a trained expert in gt_agents/.
#FEEDBACK_TYPES = ["evaluative", "comparative", "demonstrative"]
FEEDBACK_TYPES = ["comparative"]


# ---------------------------------------------------------------------------
# Reference-data generation
# ---------------------------------------------------------------------------

def generate_reference_data(env_name: str, out_path: Path, n_segments: int = 200, segment_len: int = 50, seed: int = 0) -> None:
    """Collect random-policy trajectories and save them as oracle calibration data."""
    print(f"Generating reference data → {out_path}  ({n_segments} segments × {segment_len} steps)")
    rng = np.random.default_rng(seed)
    env = TrainingUtils.setup_environment(env_name, seed=seed)
    env.reset(seed=seed)

    segments = []
    opt_gaps = []

    for _ in range(n_segments):
        obs, _ = env.reset()
        segment = []
        for _ in range(segment_len):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            segment.append((obs, action, float(reward), terminated or truncated))
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()

        segments.append(segment)
        # Compute opt_gap using discounted return, zeroing out reward on termination
        # steps. This is critical for envs like CartPole where reward=1 on every step
        # (including the fail step), which would otherwise make all segments identical.
        ret = sum((0.0 if step[3] else step[2]) * (0.99 ** t) for t, step in enumerate(segment))
        opt_gaps.append(-ret)  # oracle uses -return as gap (lower is better)

    env.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"segments": segments, "opt_gaps": opt_gaps}, f)
    print(f"  Saved {len(segments)} segments")


# ---------------------------------------------------------------------------
# Trajectory saving (rlhfblender-compatible format)
# ---------------------------------------------------------------------------

def collect_renders(trajectories, env_name: str, seed: int, fixed_state=None) -> list[list[np.ndarray]]:
    """Replay trajectories in a render-capable env to collect RGB frames per step.

    Returns a list (one per trajectory) of lists of (H,W,3) uint8 arrays.
    Falls back to blank frames if rendering is unavailable.

    fixed_state: optional state dict from FixedStartWrapper.save_state(). When provided,
                 every episode resets to the same initial configuration so that the replay
                 starts from the same position as the original collection.
    """
    import gymnasium as gym

    try:
        if "metaworld" in env_name:
            # Metaworld envs are registered under "Meta-World/MT1", not their own ID
            environment_name = env_name.replace("metaworld-", "")
            render_env = gym.make(
                "Meta-World/MT1",
                env_name=environment_name,
                seed=seed,
                render_mode="rgb_array",
            )
        else:
            render_env = gym.make(env_name, render_mode="rgb_array")

        if fixed_state is not None:
            from multi_type_feedback.fixed_start_wrapper import FixedStartWrapper
            render_env = FixedStartWrapper(render_env, fixed_state=fixed_state)
    except Exception as e:
        print(f"  [WARN] Could not create render env: {e}. Using blank frames.")
        return [[np.zeros((128, 128, 3), dtype=np.uint8)] * len(t) for t in trajectories]

    all_renders = []
    try:
        for trajectory in trajectories:
            ep_renders = []
            obs, _ = render_env.reset()
            for step in trajectory:
                _, action, _, _ = step[:4]
                # Decode one-hot encoded actions back to integer for discrete envs only.
                # For continuous envs (Box) the action is already the right array.
                if isinstance(render_env.action_space, gym.spaces.Discrete):
                    if isinstance(action, np.ndarray) and action.ndim == 1:
                        action = int(np.argmax(action))
                    else:
                        action = int(action)
                render_env.step(action)
                frame = render_env.render()
                if frame is None:
                    frame = np.zeros((128, 128, 3), dtype=np.uint8)
                ep_renders.append(frame.astype(np.uint8))
            all_renders.append(ep_renders)
    except Exception as e:
        print(f"  [WARN] Render collection failed mid-trajectory: {e}. Using blank frames.")
        # Pad remaining trajectories with blanks
        while len(all_renders) < len(trajectories):
            all_renders.append([np.zeros((128, 128, 3), dtype=np.uint8)] * len(trajectories[len(all_renders)]))
    finally:
        render_env.close()

    return all_renders


def save_trajectories(trajectories, env_name: str, exp_id: int, checkpoint_step: int,
                      renders: list[list[np.ndarray]] | None = None,
                      initial_states: list | None = None) -> str:
    """Save trajectories as a single npz in data/saved_benchmarks/.

    renders:        optional per-episode, per-step RGB frames; falls back to blank if None.
    initial_states: optional list of env state dicts (one per episode) from SaveResetEnvWrapper.
                    Stored at step 0 of each episode so the episode can be replayed later.
    """
    from rlhfblender.routes.dynamic_rlhf import process_env_name

    buffers = {k: [] for k in ["obs", "actions", "rewards", "dones", "infos",
                                "probs", "renders", "env_states", "uncertainty"]}
    episode_rewards, episode_lengths = [], []

    for ep_idx, trajectory in enumerate(trajectories):
        if not trajectory:
            continue
        ep_renders = renders[ep_idx] if renders else None
        ep_initial_state = initial_states[ep_idx] if initial_states and ep_idx < len(initial_states) else None
        for step_idx, step in enumerate(trajectory):
            # simulated collect_trajectories returns 4-tuples: (obs, action, reward, done)
            obs, action, reward, done = step[:4]
            is_last = step_idx == len(trajectory) - 1
            buffers["obs"].append(np.squeeze(obs))
            buffers["actions"].append(np.squeeze(action))
            buffers["rewards"].append(reward)
            buffers["dones"].append(is_last or done)
            buffers["uncertainty"].append(0.0)
            # Store the initial state only at step 0; None for subsequent steps
            buffers["env_states"].append(ep_initial_state if step_idx == 0 else None)
            buffers["infos"].append({"timestep": step_idx, "episode_id": ep_idx})
            buffers["probs"].append(0.0)
            if ep_renders and step_idx < len(ep_renders):
                buffers["renders"].append(ep_renders[step_idx])
            else:
                buffers["renders"].append(np.zeros((128, 128, 3), dtype=np.uint8))

        episode_rewards.append(sum(s[2] for s in trajectory))
        episode_lengths.append(len(trajectory))

    for k in buffers:
        buffers[k] = np.array(buffers[k], dtype=object if k == "env_states" else None)

    env_proc = process_env_name(env_name)
    save_dir = Path("data", "saved_benchmarks", env_proc)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{env_proc}_{exp_id}_{checkpoint_step}.npz"

    np.savez(
        save_path,
        obs=buffers["obs"],
        actions=buffers["actions"],
        rewards=buffers["rewards"],
        dones=buffers["dones"],
        infos=buffers["infos"],
        probs=buffers["probs"],
        renders=buffers["renders"],
        uncertainty=buffers["uncertainty"],
        env_states=buffers["env_states"],
        episode_rewards=np.array(episode_rewards),
        episode_lengths=np.array(episode_lengths),
        additional_metrics={},
    )
    print(f"  Saved {len(trajectories)} trajectories → {save_path}")
    return str(save_path)


def split_benchmarks_to_episodes(env_name: str, exp_id: int, checkpoint_step: int) -> None:
    """Split the bulk saved-benchmark npz into per-episode files and encode videos.

    Produces:
        data/episodes/{env}/{env}_{exp_id}_{ckpt}/benchmark_{N}.npz
        data/rewards/{env}/{env}_{exp_id}_{ckpt}/rewards_{N}.npy
        data/renders/{env}/{env}_{exp_id}_{ckpt}/{N}.mp4
        data/thumbnails/{env}/{env}_{exp_id}_{ckpt}/{N}.jpg
    """
    import cv2
    from rlhfblender.routes.dynamic_rlhf import process_env_name
    from rlhfblender.utils.data_generation import encode_video

    env_proc = process_env_name(env_name)
    bulk_path = Path("data", "saved_benchmarks", env_proc,
                     f"{env_proc}_{exp_id}_{checkpoint_step}.npz")
    if not bulk_path.exists():
        print(f"  [WARN] No bulk benchmark found at {bulk_path}, skipping episode split.")
        return

    data = np.load(bulk_path, allow_pickle=True)
    dones = data["dones"]
    episode_ends = np.argwhere(dones).flatten()
    n_episodes = len(episode_ends) + 1

    # Split scalar-array fields (skip object arrays and per-episode summaries)
    skip = {"additional_metrics", "env_states", "episode_rewards", "episode_lengths"}
    fields = [k for k in data.files if k not in skip and k != "renders"]
    episode_data: dict[str, list] = {}
    for name in fields:
        arr = data[name]
        episode_data[name] = np.split(arr, episode_ends + 1) if arr.ndim > 0 else [arr] * n_episodes

    # Split renders separately (may be uint8 or object array)
    renders_raw = data["renders"] if "renders" in data.files else None
    if renders_raw is not None and renders_raw.ndim > 0:
        render_episodes = np.split(renders_raw, episode_ends + 1)
    else:
        render_episodes = [np.zeros((1, 128, 128, 3), dtype=np.uint8)] * n_episodes

    # Output directories
    base_name = f"{env_proc}_{exp_id}_{checkpoint_step}"
    out_dir      = Path("data", "episodes",   env_proc, base_name)
    rewards_dir  = Path("data", "rewards",    env_proc, base_name)
    renders_dir  = Path("data", "renders",    env_proc, base_name)
    thumbs_dir   = Path("data", "thumbnails", env_proc, base_name)
    for d in (out_dir, rewards_dir, renders_dir, thumbs_dir):
        d.mkdir(parents=True, exist_ok=True)

    for ep_idx in range(n_episodes):
        # Per-episode npz
        ep = {k: episode_data[k][ep_idx] for k in fields}
        np.savez(out_dir / f"benchmark_{ep_idx}.npz", **ep)

        # Rewards
        np.save(rewards_dir / f"rewards_{ep_idx}.npy", episode_data["rewards"][ep_idx])

        # Video (encode_video handles both real and blank frames)
        ep_renders = render_episodes[ep_idx]
        if ep_renders.dtype != np.uint8:
            ep_renders = ep_renders.astype(np.uint8)
        if ep_renders.shape[0] == 0:
            ep_renders = np.zeros((1, 128, 128, 3), dtype=np.uint8)
        encode_video(ep_renders, str(renders_dir / str(ep_idx)))

        # Thumbnail: second-to-last frame, or blank
        if ep_renders.shape[0] > 1:
            thumb = ep_renders[-2]
        else:
            thumb = ep_renders[0]
        thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(thumbs_dir / f"{ep_idx}.jpg"), thumb_bgr)

    # Delete the trailing empty episode that np.split always produces after the
    # last done=True (mirrors the same clean-up in data_generation.py)
    last_idx = n_episodes - 1
    for path in [
        out_dir    / f"benchmark_{last_idx}.npz",
        rewards_dir / f"rewards_{last_idx}.npy",
        renders_dir / f"{last_idx}.mp4",
        thumbs_dir  / f"{last_idx}.jpg",
    ]:
        if path.exists():
            path.unlink()

    n_valid = n_episodes - 1
    print(f"  Split into {n_valid} per-episode files + videos → {out_dir}")


# ---------------------------------------------------------------------------
# Projection generation (calls the existing scripts as subprocesses)
# ---------------------------------------------------------------------------

def run_projections(exp_name: str, exp_id: int, checkpoint_step: int, reward_model_path: str | None, agent_path: str | None, episode_path: str, algorithm: str = "ppo") -> None:
    """Generate PCA projection + inverse predictions for one checkpoint."""
    # 1. Generate projection
    proj_cmd = [
        sys.executable,
        "rlhfblender/projections/generate_projections.py",
        "--experiment-name", exp_name,
        "--checkpoint", str(checkpoint_step),
        "--projection-method", "PCA",
        "--compute-inverse",
        "--auto-grid-range",
        "--no-feature",
        "--no-transition",
        "--no-clustering",
    ]
    print(f"  Running projection generation for checkpoint {checkpoint_step}...")
    result = subprocess.run(proj_cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  [WARN] Projection failed:\n{result.stderr[-2000:]}")
        return
    print("  Projection done.")

    # 2. Generate reward/uncertainty predictions (only if we have a reward model)
    if reward_model_path and agent_path and Path(reward_model_path).exists() and Path(agent_path).exists():
        from rlhfblender.routes.dynamic_rlhf import process_env_name
        env_proc = process_env_name(exp_name.split("_")[0] if "_" in exp_name else exp_name)
        # Try to derive env_name from the projection file that was just written
        proj_file = Path("data", "saved_projections", f"{exp_name.replace('/', '_').replace(':', '_')}_{exp_id}_{checkpoint_step}_PCA.json")
        # Fall back to the episode path prefix if projection hash differs
        predict_cmd = [
            sys.executable,
            "rlhfblender/projections/predict_reward_and_uncertainty.py",
            "--experiment-name", exp_name,
            "--checkpoint", str(checkpoint_step),
            "--reward-model", reward_model_path,
            "--output-dir", "data/saved_projections",
            "--policy-algorithm", algorithm,
            "--policy-model", agent_path,
            "--episode-path", episode_path,
            "--reward-model-type", "separate",
        ]
        print(f"  Running reward/uncertainty prediction...")
        presult = subprocess.run(predict_cmd, capture_output=True, text=True, timeout=300)
        if presult.returncode != 0:
            print(f"  [WARN] Prediction failed:\n{presult.stderr[-2000:]}")
        else:
            print("  Prediction done.")


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

async def get_or_create_experiment(exp_name: str, env_id: str, algorithm: str) -> int:
    """Return experiment ID, creating the DB entry if it doesn't exist yet."""
    import time
    db = Database(DATABASE_URL)
    await db.connect()
    try:
        existing = await get_single_entry(db, Experiment, exp_name, key_column="exp_name")
        if existing:
            print(f"Found existing experiment '{exp_name}' with id={existing.id}")
            return existing.id
    except Exception:
        pass

    await add_entry(db, Experiment, {
        "exp_name": exp_name,
        "env_id": env_id,
        "algorithm": algorithm,
        "created_timestamp": int(time.time()),
        "run_timestamp": int(time.time()),
        "status": ["created"],
        "checkpoint_list": [],
        "framework": "stable-baselines3",
    })
    exp = await get_single_entry(db, Experiment, exp_name, key_column="exp_name")
    await db.disconnect()
    print(f"Created experiment '{exp_name}' with id={exp.id}")
    return exp.id


async def ensure_experiment_in_project(exp_name: str, project_name: str) -> None:
    """Add exp_name to project_experiments of the named project (creating it if needed)."""
    import time
    from rlhfblender.data_handling.database_handler import get_all, update_entry
    from rlhfblender.data_models.global_models import Project

    db = Database(DATABASE_URL)
    await db.connect()

    projects = await get_all(db, Project)
    project = next((p for p in projects if p.project_name == project_name), None)

    if project is None:
        await add_entry(db, Project, {
            "project_name": project_name,
            "created_timestamp": int(time.time()),
            "project_path": "",
            "project_description": "Auto-created by run_simulated_phases.py",
            "project_tags": [],
            "project_environments": [],
            "project_datasets": [],
            "project_experiments": [exp_name],
        })
        print(f"Created project '{project_name}' and added experiment '{exp_name}'")
    else:
        exps = project.project_experiments or []
        if exp_name not in exps:
            exps.append(exp_name)
            await update_entry(db, Project, project.id, data={"project_experiments": exps})
            print(f"Added experiment '{exp_name}' to project '{project_name}'")
        else:
            print(f"Experiment '{exp_name}' already in project '{project_name}'")

    await db.disconnect()


async def update_checkpoint_list(exp_id: int, checkpoint_step: int) -> None:
    """Append checkpoint_step to the experiment's checkpoint_list in the DB."""
    from rlhfblender.data_handling.database_handler import update_entry
    db = Database(DATABASE_URL)
    await db.connect()
    exp = await get_single_entry(db, Experiment, exp_id)
    checkpoints = exp.checkpoint_list or []
    if checkpoint_step not in checkpoints:
        checkpoints.append(checkpoint_step)
        checkpoints.sort()
        await update_entry(db, Experiment, exp_id, data={"checkpoint_list": checkpoints})
    await db.disconnect()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run simulated DynamicRLHF training phases")
    parser.add_argument("--env", default="metaworld-sweep-into-v3", help="Environment name (use metaworld- prefix for metaworld envs)")
    parser.add_argument("--algorithm", default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--expert-algorithm", default=None, choices=["ppo", "sac"],
                        help="Algorithm used by the expert models for oracle feedback (default: same as --algorithm). "
                             "Use e.g. --algorithm ppo --expert-algorithm sac to train PPO online with an SAC oracle.")
    parser.add_argument("--exp-name", default=None, help="Experiment name (default: {env}_{algorithm}_sim)")
    parser.add_argument("--project-name", default="RLHF-Blender", help="Project to register the experiment in (default: RLHF-Blender)")
    parser.add_argument("--num-phases", type=int, default=5, help="Number of training phases")
    parser.add_argument("--feedback-budget", type=int, default=500, help="Total oracle feedback budget")
    parser.add_argument("--feedback-buffer-size", type=int, default=None,
                        help="Max feedback buffer size per type (default: feedback_budget). "
                             "Smaller values expire old data faster, reducing buffer staleness.")
    parser.add_argument("--initial-feedback", type=int, default=50, help="Feedback for phase 0 (random agent)")
    parser.add_argument("--rl-steps", type=int, default=10000, help="RL training steps per phase")
    parser.add_argument("--reward-epochs", type=int, default=10, help="Reward model training epochs per phase")
    parser.add_argument("--segment-len", type=int, default=50, help="Trajectory segment length for oracle")
    parser.add_argument("--n-trajectories", type=int, default=10, help="Trajectories to collect per phase")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--feedback-types", nargs="+", default=FEEDBACK_TYPES)
    parser.add_argument("--skip-projections", action="store_true", help="Skip projection generation (faster)")
    parser.add_argument("--reference-data-path", default=None,
                        help="Path to a pre-generated reference .pkl file (expert rollouts). "
                             "When supplied, reference data generation is skipped entirely.")
    parser.add_argument("--expert-model-path", default=None,
                        help="Path to trained baseline agents (e.g. multi-type-feedback/train_baselines/gt_agents). "
                             "When supplied, expert models are loaded for oracle demonstrative feedback and "
                             "used via generate_feedback to produce reference calibration data.")
    parser.add_argument("--top-n-models", type=int, default=3,
                        help="Number of top-ranked expert models to load (ranked by eval score if available).")
    parser.add_argument("--max-episode-steps", type=int, default=None,
                        help="Override the default max steps per episode (e.g. 200 for Metaworld instead of 500). "
                             "Passed as max_episode_steps to gym.make.")
    parser.add_argument("--uncertainty-penalty", type=float, default=0.0,
                        help="Subtract this coefficient * ensemble_std from the reward during RL training. "
                             "Penalises the agent for visiting states where the reward model is uncertain, "
                             "reducing reward hacking. Good starting values: 0.1–0.5.")
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Override the number of parallel training environments (default: use YAML value). "
                             "Set to 1 for short RLHF phases (avoids a rollout buffer larger than total_timesteps).")
    parser.add_argument("--results-only", action="store_true",
                        help="Skip render collection, model checkpoints, and episode splits. "
                             "Only save episode_rewards per phase (much smaller disk footprint, "
                             "suitable for large ablation sweeps).")
    parser.add_argument("--reward-model-type", type=str, default="unified",
                        choices=["separate", "multi-head", "unified", "film-unified"],
                        help="Reward model architecture. 'film-unified' uses FiLM conditioning "
                             "with optional ResponseRank loss.")
    parser.add_argument("--reward-normalization", type=str, default="welford",
                        choices=["welford", "quantile"],
                        help="How to normalize rewards across feedback types before aggregation. "
                             "'quantile' maps each type to [0,1] via running quantile buffers "
                             "(scale-invariant; recommended with film-unified).")
    parser.add_argument("--reward-batch-size", type=int, default=0,
                        help="Batch size for unified reward model training. "
                             "0 = auto (8 for film-unified, 1 otherwise).")
    parser.add_argument("--responserank-weight", type=float, default=0.5,
                        help="Weight for ResponseRank Plackett-Luce loss vs Bradley-Terry NLL "
                             "in pairwise feedback types (only used with film-unified). "
                             "0.0 = pure BT, 1.0 = pure ResponseRank.")
    parser.add_argument("--eval-freq", type=int, default=2000,
                        help="Evaluate RL agent on GT env reward every N steps (0 to disable). "
                             "Results printed as eval/mean_reward.")
    parser.add_argument("--feedback-sampling", type=str, default="random",
                        choices=["random", "uncertainty"],
                        help="Strategy for selecting which trajectories to query: 'random' (uniform) or "
                             "'uncertainty' (prefer segments with high ensemble variance, maximising "
                             "information gain per labelling session).")
    parser.add_argument("--reward-model-hidden-dim", type=int, default=256,
                        help="Hidden layer width of the reward model MLP (default: 256). "
                             "Smaller values (e.g. 64, 128) reduce capacity and can improve "
                             "generalisation when the feedback budget is small.")
    parser.add_argument("--reward-model-layer-num", type=int, default=6,
                        help="Number of layers in the reward model MLP (default: 6). "
                             "Smaller values (e.g. 2, 3) reduce overfitting with limited labels.")
    parser.add_argument("--hyperparams", nargs="+", default=None,
                        help="Override PPO/SAC hyperparameters as KEY:VALUE pairs. "
                             "Example: --hyperparams learning_rate:1e-4 batch_size:128 gamma:0.995 "
                             "Values are auto-cast to int/float/bool/str.")
    parser.add_argument("--fix-start-state", action="store_true",
                        help="Fix starting/object/goal positions across all episodes by locking the first reset state. "
                             "Recommended for Metaworld to simplify the task without requiring full training convergence.")
    parser.add_argument("--state-seed", type=int, default=None,
                        help="Seed used exclusively for sampling the fixed start state. "
                             "Use the SAME value here and in train_local.sh (--state-seed) so that "
                             "baseline expert policies and RLHF training share an identical starting configuration. "
                             "Defaults to --seed when not set.")
    args = parser.parse_args()

    env_name = args.env
    algorithm = args.algorithm
    expert_algorithm = args.expert_algorithm or algorithm  # defaults to same as online algo
    exp_name = args.exp_name or f"{env_name.replace('/', '_')}_{algorithm}_sim"

    # Parse --hyperparams KEY:VALUE pairs into a dict for ExperimentManager
    custom_hyperparams = None
    if args.hyperparams:
        custom_hyperparams = {}
        for kv in args.hyperparams:
            key, val_str = kv.split(":", 1)
            # Auto-cast value
            if val_str.lower() in ("true", "false"):
                val = val_str.lower() == "true"
            else:
                try:
                    val = int(val_str)
                except ValueError:
                    try:
                        val = float(val_str)
                    except ValueError:
                        val = val_str
            custom_hyperparams[key] = val
        print(f"  PPO hyperparameter overrides: {custom_hyperparams}")

    # Build env_kwargs — currently only max_episode_steps if the user requested it.
    env_kwargs = {}
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps

    # --- Step 1: Create/find experiment in DB and register in project ---
    exp_id = asyncio.run(get_or_create_experiment(exp_name, env_name, algorithm))
    asyncio.run(ensure_experiment_in_project(exp_name, args.project_name))

    # --- Step 2: Load expert models (optional) ---
    from rlhfblender.routes.dynamic_rlhf import process_env_name
    from stable_baselines3 import PPO, SAC

    gen_env = TrainingUtils.setup_environment(env_name, seed=args.seed, env_kwargs=env_kwargs or None)
    expert_models = []

    if args.expert_model_path:
        expert_model_path = Path(args.expert_model_path)
        algo_dir = expert_model_path / expert_algorithm
        if algo_dir.exists() and any(env_name in d for d in os.listdir(algo_dir)):
            print(f"Loading expert models from {expert_model_path} (algo={expert_algorithm})...")
            expert_models = TrainingUtils.load_expert_models(
                env_name=env_name,
                algorithm=expert_algorithm,
                checkpoints_path=str(expert_model_path),
                environment=gen_env,
                top_n_models=args.top_n_models,
            )
            print(f"  Loaded {len(expert_models)} expert model(s)")
        else:
            print(f"[WARN] No models found for '{env_name}' in {algo_dir}, falling back to random-policy reference data.")

    # --- Step 3: Build reference data for oracle ---
    env_proc = process_env_name(env_name)
    ref_data_dir = Path("multi-type-feedback/feedback")
    ref_data_dir.mkdir(parents=True, exist_ok=True)
    ref_data_path = ref_data_dir / f"{env_proc}_reference.pkl"

    if args.reference_data_path:
        ref_data_path = Path(args.reference_data_path)
        if not ref_data_path.exists():
            raise FileNotFoundError(f"--reference-data-path not found: {ref_data_path}")
        print(f"Using supplied reference data: {ref_data_path}")
    elif not ref_data_path.exists():
        if expert_models:
            print(f"Generating expert reference data via generate_feedback → {ref_data_path}")
            import gymnasium as gym
            feedback_data = generate_feedback(
                model_class=PPO if expert_algorithm == "ppo" else SAC,
                expert_models=expert_models,
                environment=gen_env,
                environment_name=env_name,
                checkpoints_path=str(args.expert_model_path),
                n_feedback=200,
                total_steps_factor=200,  # 200×200=40k rollout steps; enough across many checkpoints
                segment_len=args.segment_len,
                algorithm=expert_algorithm,
                device=args.device,
                action_one_hot=isinstance(gen_env.action_space, gym.spaces.Discrete),
            )
            with open(ref_data_path, "wb") as f:
                pickle.dump(feedback_data, f)
            print(f"  Saved {len(feedback_data['segments'])} segments")
        else:
            generate_reference_data(env_name, ref_data_path, n_segments=200, segment_len=args.segment_len, seed=args.seed)
    else:
        print(f"Using existing reference data: {ref_data_path}")

    # --- Step 4: Create oracle ---
    # Enable demonstrative feedback automatically if expert models are available.
    feedback_types = list(args.feedback_types)
    #if expert_models and "demonstrative" not in feedback_types:
    #    feedback_types.append("demonstrative")
    #    print("  Expert models found → enabling demonstrative feedback")
    if not expert_models and "demonstrative" in feedback_types:
        feedback_types.remove("demonstrative")
        print("  [WARN] --feedback-types included 'demonstrative' but no expert models were loaded. "
              "Removing 'demonstrative' to avoid a crash (oracle._get_best_demonstration returns None).")

    oracle = FeedbackOracle(
        expert_models=expert_models,
        environment=gen_env,
        reference_data_path=str(ref_data_path),
        segment_len=args.segment_len,
        # Use stochastic expert actions so that demonstrations are diverse even when
        # --fix-start-state pins every episode to the same start (deterministic expert
        # would produce identical demos each call, giving the reward model zero new signal).
        deterministic_expert=not args.fix_start_state,
    )

    # --- Step 4b: Sample a fixed start state (if requested) ---
    # All envs (train, eval, collect, render) will restore to this exact state on
    # every reset, fixing object/goal positions across episodes.
    from multi_type_feedback.fixed_start_wrapper import FixedStartWrapper, sample_fixed_state

    fixed_start_state = None
    if args.fix_start_state:
        state_seed = args.state_seed if args.state_seed is not None else args.seed
        print(f"  Sampling/loading fixed start state for {env_name} (state_seed={state_seed})…")
        fixed_start_state = sample_fixed_state(env_name, state_seed)
        print("  Fixed start state locked.")

    # --- Step 5: Create ExperimentManager for RL hyperparameters ---
    exp_manager = ExperimentManager(
        args=SimpleNamespace(),
        algo=algorithm,
        env_id=env_name,
        log_folder=f"dynamic_rlhf_models/sim_{exp_name}",
        n_timesteps=args.rl_steps,
        eval_freq=args.eval_freq,
        n_eval_episodes=5,
        env_kwargs=env_kwargs or None,
        hyperparams=custom_hyperparams,
    )

    # Override n_envs before setup_experiment() is called (inside DynamicRLHF.__init__)
    if args.n_envs is not None:
        exp_manager._n_envs_override = args.n_envs

    # Apply the fixed-start wrapper to train *and* eval envs created by ExperimentManager.
    if fixed_start_state is not None:
        exp_manager.env_wrapper = lambda env: FixedStartWrapper(env, fixed_state=fixed_start_state)

    # --- Step 6: Initialise DynamicRLHF ---
    print(f"\nInitialising DynamicRLHF for {env_name} ({algorithm}), {args.num_phases} phases")
    print(f"  Feedback types: {feedback_types}")
    drlhf = DynamicRLHF(
        oracle=oracle,
        env_name=env_name,
        algorithm=algorithm,
        feedback_types=feedback_types,
        nr_of_iterations=args.num_phases,
        feedback_budget=args.feedback_budget,
        feedback_buffer_size=args.feedback_buffer_size if args.feedback_buffer_size is not None else max(args.feedback_budget, 200),
        reward_training_epochs=args.reward_epochs,
        num_ensemble_models=4,
        initial_feedback_count=args.initial_feedback,
        rl_steps_per_iteration=args.rl_steps,
        device=args.device,
        seed=args.seed,
        reward_model_type=args.reward_model_type,
        exp_manager=exp_manager,
        env_kwargs=env_kwargs or None,
        uncertainty_penalty=args.uncertainty_penalty,
        reward_normalization=args.reward_normalization,
        responserank_weight=args.responserank_weight,
        reward_batch_size=args.reward_batch_size,
        reward_model_hidden_dim=args.reward_model_hidden_dim,
        reward_model_layer_num=args.reward_model_layer_num,
    )

    # --- Step 6: Phase loop ---
    for phase in range(args.num_phases):
        checkpoint_step = phase  # phase number doubles as checkpoint ID
        print(f"\n{'='*60}")
        print(f"Phase {phase} / checkpoint {checkpoint_step}")
        print(f"{'='*60}")

        # Collect trajectories — use a fixed-start env when requested so that
        # every episode (and every phase) begins from the same configuration.
        print(f"  Collecting {args.n_trajectories} trajectories...")
        if fixed_start_state is not None:
            _collect_env = FixedStartWrapper(
                TrainingUtils.setup_environment(env_name, seed=args.seed, env_kwargs=env_kwargs or None),
                fixed_state=fixed_start_state,
            )
            trajectories, initial_states = drlhf.collect_trajectories(
                n_trajectories=args.n_trajectories, env=_collect_env
            )
            _collect_env.close()
        else:
            trajectories, initial_states = drlhf.collect_trajectories(n_trajectories=args.n_trajectories)

        if args.results_only:
            # Minimal save: only episode_rewards (no renders, no episode splits, no model ckpts)
            from rlhfblender.routes.dynamic_rlhf import process_env_name as _penv
            _env_proc = _penv(env_name)
            _save_dir = Path("data", "saved_benchmarks", _env_proc)
            _save_dir.mkdir(parents=True, exist_ok=True)
            _save_path = _save_dir / f"{_env_proc}_{exp_id}_{checkpoint_step}.npz"
            _ep_rewards = np.array([sum(s[2] for s in traj) for traj in trajectories if traj])
            _ep_lengths = np.array([len(traj) for traj in trajectories if traj])
            np.savez(_save_path, episode_rewards=_ep_rewards, episode_lengths=_ep_lengths)
            print(f"  Saved episode rewards → {_save_path}  ({len(_ep_rewards)} episodes)")
        else:
            # Collect render frames by replaying in a render-capable env
            print(f"  Collecting render frames...")
            renders = collect_renders(trajectories, env_name, seed=args.seed, fixed_state=fixed_start_state)

            # Save in rlhfblender-compatible format (with real render frames + initial env states)
            save_trajectories(trajectories, env_name, exp_id, checkpoint_step,
                              renders=renders, initial_states=initial_states)

            # Split bulk benchmark into per-episode files + encode videos/thumbnails
            split_benchmarks_to_episodes(env_name, exp_id, checkpoint_step)

            # Save DynamicRLHF checkpoint (reward models + RL agent + state)
            ckpt_base = f"dynamic_rlhf_models/sim_{exp_name}_checkpoint_{checkpoint_step}"
            drlhf.save(ckpt_base, checkpoint_step=checkpoint_step, exp_id=str(exp_id))

        # Update DB checkpoint list
        asyncio.run(update_checkpoint_list(exp_id, checkpoint_step))

        # Derive paths for reward prediction script
        ckpt_dir = Path("multi-type-feedback/reward_models/checkpoints")
        # For "separate" the file is named per feedback type; for "unified"/"multi-head" it uses the type name directly.
        reward_model_type = drlhf.reward_model_type
        if reward_model_type == "separate":
            ckpt_suffix = feedback_types[0]
        else:
            ckpt_suffix = reward_model_type
        reward_model_path = str(ckpt_dir / f"{algorithm}_{env_name.lower().replace('-', '_')}_{exp_id}_{ckpt_suffix}_{checkpoint_step}.ckpt")
        agent_path = str(Path("multi-type-feedback/train_baselines/dynamic_rlhf_agents") / f"{algorithm}_{env_name.lower().replace('-', '_')}_{exp_id}_{checkpoint_step}.zip")
        episode_path = str(Path("data", "episodes", env_proc, f"{env_proc}_{exp_id}_{checkpoint_step}"))

        # Run projections (subprocess, matches server behaviour)
        if not args.skip_projections:
            run_projections(
                exp_name=exp_name,
                exp_id=exp_id,
                checkpoint_step=checkpoint_step,
                reward_model_path=reward_model_path,
                agent_path=agent_path,
                episode_path=episode_path,
                algorithm=algorithm,
            )

        # For phases > 0: collect oracle feedback, train reward model, train RL agent
        if phase < args.num_phases - 1:
            print(f"  Sampling oracle feedback ({drlhf.n_feedback_per_iteration} queries, strategy={args.feedback_sampling})...")
            if args.feedback_sampling == "uncertainty":
                drlhf.sample_feedback_uncertainty(trajectories, initial_states)
            else:
                drlhf.sample_feedback_random(trajectories, initial_states)

            print(f"  Training reward models ({args.reward_epochs} epochs)...")
            metrics = drlhf.train_reward_models()
            print(f"  Reward model losses: { {k: f'{v:.4f}' for k, v in metrics.items()} }")
            drlhf.print_reward_model_diagnostics()

            print(f"  Training RL agent ({args.rl_steps} steps)...")
            # reset_num_timesteps=True resets the timestep counter so PPO's linear
            # lr schedule restarts from lr_init each phase.  Without this the schedule
            # computes lr * (1 - elapsed/n_timesteps) and hits 0 after phase 1.
            drlhf.rl_agent.learn(total_timesteps=args.rl_steps, reset_num_timesteps=True)

    gen_env.close()
    print(f"\nDone! {args.num_phases} phases complete.")
    print(f"Experiment ID: {exp_id}  |  Name: {exp_name}")
    print(f"Open the rlhfblender UI and select experiment '{exp_name}' to view results.")


if __name__ == "__main__":
    main()
