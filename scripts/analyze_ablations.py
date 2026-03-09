"""
Analyze ablation results from submit_ablations.sh runs.

Usage (from repo root):
    python scripts/analyze_ablations.py [--output-dir plots] [--show]

Outputs:
    plots/group_A.png, plots/group_B.png, ...  — per-group learning curves
    plots/final_performance.png               — bar chart of final-phase mean reward
    plots/summary.csv                         — table: config, seed, phase, mean_reward
"""

import argparse
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Config ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = "rlhfblender.db"
BENCHMARKS_DIR = REPO_ROOT / "data" / "saved_benchmarks"
MONITOR_BASE = REPO_ROOT / "dynamic_rlhf_models"
EXP_NAME_PREFIX = "mw_sweep-into-v3_ppo_"

# Map group letter → human-readable axis label for plot titles
GROUP_LABELS = {
    "A": "RL steps / phase",
    "B": "Phase granularity (total=20k steps)",
    "C": "Feedback budget",
    "D": "Reward model epochs",
    "E": "Initial feedback count",
    "F": "Feedback buffer size",
    "G": "Segment length",
    "H": "Best-guess combinations",
    "I": "Feedback type combinations",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_experiments_from_db(db_path: Path) -> dict[str, int]:
    """Return {exp_name: exp_id} for all ablation experiments."""
    
    db_path = Path(db_path)

    if not db_path.exists():
        print(f"[WARN] Database not found: {db_path}")
        return {}
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT exp_name, id FROM experiment WHERE exp_name LIKE ?",
            (EXP_NAME_PREFIX + "%",),
        ).fetchall()
    except sqlite3.OperationalError as e:
        print(f"[WARN] DB query failed: {e}")
        return {}
    finally:
        con.close()
    return {name: eid for name, eid in rows}


def load_npz_rewards(exp_id: int, env_proc: str = "sweep_into_v3") -> dict[int, np.ndarray]:
    """
    Load episode_rewards arrays from all checkpoint NPZ files for one experiment.
    Returns {checkpoint_step: episode_rewards_array}.
    """
    bench_dir = BENCHMARKS_DIR / env_proc
    results = {}
    if not bench_dir.exists():
        return results
    pattern = f"{env_proc}_{exp_id}_*.npz"
    for npz_path in sorted(bench_dir.glob(pattern)):
        # filename: {env_proc}_{exp_id}_{checkpoint_step}.npz
        stem = npz_path.stem  # e.g. sweep_into_v3_42_3
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        try:
            step = int(parts[1])
        except ValueError:
            continue
        try:
            data = np.load(npz_path, allow_pickle=True)
            rewards = data["episode_rewards"]
            results[step] = rewards.astype(float)
        except Exception as e:
            print(f"[WARN] Failed to load {npz_path}: {e}")
    return results


def load_monitor_csv(exp_name: str) -> pd.DataFrame | None:
    """
    Load the SB3 monitor CSV for the RL training environment.
    Returns a DataFrame with columns [r, l, t, success (optional)], or None.
    Note: 'r' here is the REWARD MODEL reward, not ground truth.
    """
    run_dir = MONITOR_BASE / f"sim_{exp_name}"
    if not run_dir.exists():
        return None
    csvs = sorted(run_dir.rglob("monitor.csv"))
    if not csvs:
        return None
    frames = []
    for csv in csvs:
        try:
            df = pd.read_csv(csv, comment="#")
            frames.append(df)
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else None


# ── Parse experiment names ────────────────────────────────────────────────────

def parse_exp_name(exp_name: str) -> tuple[str, str, int] | None:
    """
    Parse 'mw_sweep-into-v3_ppo_{label}_s{seed}' →  (group_letter, label, seed).
    Returns None if the name doesn't match the ablation pattern.
    """
    suffix = exp_name.removeprefix(EXP_NAME_PREFIX)
    # seed is always the last component: _s{N}
    m = re.search(r"_s(\d+)$", suffix)
    if not m:
        return None
    seed = int(m.group(1))
    label = suffix[: m.start()]           # e.g. "A_steps_2000"
    group = label[0] if label else "?"    # first character is the group letter
    return group, label, seed


# ── Build summary DataFrame ───────────────────────────────────────────────────

def build_summary(exp_map: dict[str, int], env_proc: str = "sweep_into_v3") -> pd.DataFrame:
    rows = []
    for exp_name, exp_id in exp_map.items():
        parsed = parse_exp_name(exp_name)
        if parsed is None:
            continue
        group, label, seed = parsed

        phase_rewards = load_npz_rewards(exp_id, env_proc)
        if not phase_rewards:
            print(f"  [missing] {exp_name} (id={exp_id}) — no NPZ files found")
            continue

        for phase, rewards in sorted(phase_rewards.items()):
            rows.append({
                "exp_name": exp_name,
                "exp_id": exp_id,
                "group": group,
                "label": label,
                "seed": seed,
                "phase": phase,
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "n_episodes": len(rewards),
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["label", "seed", "phase"])


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_group(df: pd.DataFrame, group: str, output_dir: Path) -> None:
    gdf = df[df["group"] == group]
    if gdf.empty:
        return

    labels = sorted(gdf["label"].unique())
    n_phases = gdf["phase"].max() + 1

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("tab20")

    for i, label in enumerate(labels):
        ldf = gdf[gdf["label"] == label]
        # Aggregate across seeds: mean ± stderr
        agg = (
            ldf.groupby("phase")["mean_reward"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg["stderr"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))

        color = cmap(i / max(len(labels) - 1, 1))
        # Strip the group prefix for a shorter legend entry (e.g. "A_steps_2000" → "steps_2000")
        short = re.sub(r"^[A-Z]_", "", label)
        ax.plot(agg["phase"], agg["mean"], label=short, color=color, marker="o", markersize=4)
        ax.fill_between(
            agg["phase"],
            agg["mean"] - agg["stderr"],
            agg["mean"] + agg["stderr"],
            alpha=0.15,
            color=color,
        )

    axis_label = GROUP_LABELS.get(group, f"Group {group}")
    ax.set_title(f"Group {group}: {axis_label}", fontsize=13)
    ax.set_xlabel("Phase (checkpoint)")
    ax.set_ylabel("Mean episode reward (ground truth)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, framealpha=0.7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = output_dir / f"group_{group}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_final_performance(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of final-phase mean reward for every config, grouped by ablation group."""
    last_phase = df.groupby(["label", "seed"])["phase"].max().reset_index()
    last_phase = last_phase.rename(columns={"phase": "last_phase"})
    merged = df.merge(last_phase, on=["label", "seed"])
    final = merged[merged["phase"] == merged["last_phase"]]

    agg = (
        final.groupby(["group", "label"])["mean_reward"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["stderr"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    agg = agg.sort_values(["group", "mean"], ascending=[True, False])

    fig, ax = plt.subplots(figsize=(max(12, len(agg) * 0.5), 6))
    colors = [plt.get_cmap("tab10")(ord(g) - ord("A")) for g in agg["group"]]
    short_labels = [re.sub(r"^[A-Z]_", "", l) for l in agg["label"]]

    bars = ax.bar(range(len(agg)), agg["mean"], color=colors, alpha=0.8,
                  yerr=agg["stderr"], capsize=3)

    # Group separators and labels
    prev_group = None
    for x, (_, row) in enumerate(agg.iterrows()):
        if row["group"] != prev_group:
            if prev_group is not None:
                ax.axvline(x - 0.5, color="gray", linewidth=0.8, linestyle="--")
            ax.text(x, ax.get_ylim()[1] * 0.97, f" {row['group']}", fontsize=9,
                    color="gray", va="top")
            prev_group = row["group"]

    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean episode reward (final phase)")
    ax.set_title("Final-phase performance — all ablation configs")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = output_dir / "final_performance.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_phase0_vs_final(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter: phase-0 reward vs final-phase reward — reveals which configs recover."""
    phase0 = df[df["phase"] == 0][["label", "seed", "mean_reward"]].rename(
        columns={"mean_reward": "phase0_reward"}
    )
    last_phase = df.groupby(["label", "seed"])["phase"].max().reset_index()
    last_phase = last_phase.rename(columns={"phase": "last_phase"})
    merged = df.merge(last_phase, on=["label", "seed"])
    final = merged[merged["phase"] == merged["last_phase"]][
        ["label", "seed", "mean_reward", "group"]
    ].rename(columns={"mean_reward": "final_reward"})

    scatter_df = phase0.merge(final, on=["label", "seed"])

    fig, ax = plt.subplots(figsize=(8, 6))
    groups = scatter_df["group"].unique()
    cmap = plt.get_cmap("tab10")
    for i, g in enumerate(sorted(groups)):
        gdf = scatter_df[scatter_df["group"] == g]
        ax.scatter(gdf["phase0_reward"], gdf["final_reward"],
                   label=f"Group {g}", alpha=0.7, s=50,
                   color=cmap(i / max(len(groups) - 1, 1)))

    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.4, label="no change")
    ax.set_xlabel("Mean reward — phase 0")
    ax.set_ylabel("Mean reward — final phase")
    ax.set_title("Phase 0 vs. final performance (each point = one config×seed)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_path = output_dir / "phase0_vs_final.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze ablation results")
    parser.add_argument("--output-dir", default="plots", help="Directory for output plots")
    parser.add_argument("--env-proc", default="metaworld-sweep-into-v3",
                        help="Processed env name used in benchmark file paths (must match process_env_name output)")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--csv-only", action="store_true",
                        help="Only write summary CSV, skip plots")
    args = parser.parse_args()

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading experiments from database...")
    exp_map = load_experiments_from_db(DB_PATH)
    if not exp_map:
        # Fallback: scan benchmark directories directly and try to infer exp_ids
        print("[WARN] No DB entries found. Scanning benchmark files directly.")
        bench_dir = BENCHMARKS_DIR / args.env_proc
        exp_ids = set()
        if bench_dir.exists():
            for p in bench_dir.glob("*.npz"):
                parts = p.stem.split("_")
                # Format: {env_proc}_{exp_id}_{phase} — env_proc has 3 parts for sweep_into_v3
                # e.g. sweep_into_v3_42_3 → env_proc=sweep_into_v3, id=42, phase=3
                if len(parts) >= 2:
                    try:
                        exp_ids.add(int(parts[-2]))
                    except ValueError:
                        pass
        print(f"  Found {len(exp_ids)} unique experiment IDs in benchmark files (no names available).")
        print("  Run with a valid rlhfblender.db to get named experiments.")
        return

    print(f"Found {len(exp_map)} ablation experiments in DB.")

    print("Building summary DataFrame...")
    df = build_summary(exp_map, args.env_proc)
    if df.empty:
        print("No data found. Have any phases completed yet?")
        return

    # Save summary CSV
    csv_path = output_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Summary saved → {csv_path}")
    print(f"\n{len(df)} rows across {df['label'].nunique()} configs, "
          f"{df['seed'].nunique()} seeds, phases 0–{df['phase'].max()}")

    # Print quick text table: final-phase mean by config
    last_phase_per_run = df.groupby(["label", "seed"])["phase"].max().reset_index()
    last_phase_per_run.rename(columns={"phase": "last_phase"}, inplace=True)
    merged = df.merge(last_phase_per_run, on=["label", "seed"])
    final_df = merged[merged["phase"] == merged["last_phase"]]
    summary_table = (
        final_df.groupby(["group", "label"])["mean_reward"]
        .agg(["mean", "std", "count"])
        .round(3)
        .reset_index()
        .sort_values(["group", "mean"], ascending=[True, False])
    )
    print("\n── Final-phase performance (mean ± std across seeds) ──")
    print(summary_table.to_string(index=False))

    if args.csv_only:
        return

    print("\nGenerating plots...")
    for group in sorted(df["group"].unique()):
        plot_group(df, group, output_dir)

    plot_final_performance(df, output_dir)
    plot_phase0_vs_final(df, output_dir)

    if args.show:
        matplotlib.use("TkAgg")
        plt.show()

    print(f"\nDone. All plots in {output_dir}/")


if __name__ == "__main__":
    main()
