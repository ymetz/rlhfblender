from typing import Any
import numpy as np

def process_env_name(env_name):
    """
    Process environment name to be compatible with RLHF-Blender
    """
    if "ALE" in env_name:
        env_name = env_name.replace("/", "-")
    return env_name


def convert_to_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray) and obj.size == 1:
        # dict must be recursively serializable
        obj = obj.item()
        if not isinstance(obj, (int, float, str)):
            return convert_to_serializable(obj)
        return obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj
