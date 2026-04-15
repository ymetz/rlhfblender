from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from rlhfblender.projections.generate_projections import process_env_name

logger = logging.getLogger(__name__)

JOINT_OBS_STATE_DIR = Path("data") / "saved_projections" / "joint_obs_state"
JOINT_DIR = Path("data") / "saved_projections" / "joint"

_RANGE_RE = re.compile(r"_(\d+)_(\d+)_metadata\.json$")


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _parse_filename_checkpoint_range(path: Path) -> tuple[int | None, int | None]:
    match = _RANGE_RE.search(path.name)
    if not match:
        return None, None
    low = _safe_int(match.group(1))
    high = _safe_int(match.group(2))
    return low, high


def _extract_checkpoint_stats(path: Path, metadata: dict[str, Any]) -> tuple[set[int], int, int]:
    checkpoints_raw = metadata.get("checkpoints", [])
    checkpoints = {_safe_int(cp) for cp in checkpoints_raw}
    checkpoints = {cp for cp in checkpoints if cp is not None}

    if checkpoints:
        span = max(checkpoints) - min(checkpoints)
        return checkpoints, len(checkpoints), span

    low, high = _parse_filename_checkpoint_range(path)
    if low is not None and high is not None and high >= low:
        return set(), (high - low + 1), (high - low)

    return set(), 0, 0


def _contains_checkpoint(path: Path, metadata: dict[str, Any], checkpoint_step: int | None) -> bool:
    if checkpoint_step is None:
        return True

    checkpoints, _, _ = _extract_checkpoint_stats(path, metadata)
    if checkpoints:
        return checkpoint_step in checkpoints

    low, high = _parse_filename_checkpoint_range(path)
    if low is None or high is None:
        return False
    return low <= checkpoint_step <= high


def select_joint_projection_metadata(
    environment_id: str,
    projection_method: str,
    *,
    experiment_id: int | None = None,
    checkpoint_step: int | None = None,
    prefer_obs_state: bool = True,
) -> Path | None:
    """
    Select the most suitable joint projection metadata file.

    Preference order:
    1) Matches requested checkpoint (if provided)
    2) joint_obs_state over joint (if prefer_obs_state is True)
    3) Wider / richer checkpoint coverage
    4) Newer modification time
    """
    env_component = process_env_name(environment_id)
    method = projection_method or "PCA"

    if experiment_id is None:
        obs_state_pattern = f"{env_component}_*_joint_obs_state_{method}_*_metadata.json"
        joint_pattern = f"{env_component}_*_joint_{method}_*_metadata.json"
    else:
        obs_state_pattern = f"{env_component}_{experiment_id}_joint_obs_state_{method}_*_metadata.json"
        joint_pattern = f"{env_component}_{experiment_id}_joint_{method}_*_metadata.json"

    candidates: list[Path] = []
    if JOINT_OBS_STATE_DIR.exists():
        candidates.extend(JOINT_OBS_STATE_DIR.glob(obs_state_pattern))
    if JOINT_DIR.exists():
        candidates.extend(JOINT_DIR.glob(joint_pattern))

    if not candidates:
        return None

    scored: list[tuple[tuple[int, int, int, float], Path, bool]] = []
    for candidate in candidates:
        metadata = _read_metadata(candidate)
        contains_checkpoint = _contains_checkpoint(candidate, metadata, checkpoint_step)
        _, checkpoint_count, checkpoint_span = _extract_checkpoint_stats(candidate, metadata)

        is_obs_state = "_joint_obs_state_" in candidate.name
        type_priority = 1 if (prefer_obs_state and is_obs_state) else 0

        score = (
            type_priority,
            checkpoint_count,
            checkpoint_span,
            candidate.stat().st_mtime,
        )
        scored.append((score, candidate, contains_checkpoint))

    if checkpoint_step is not None:
        matching = [entry for entry in scored if entry[2]]
        if matching:
            scored = matching
        else:
            logger.warning(
                "No joint metadata contains checkpoint=%s for env=%s exp=%s method=%s; using best available candidate.",
                checkpoint_step,
                environment_id,
                experiment_id,
                method,
            )

    best = max(scored, key=lambda entry: entry[0])[1]
    return best

