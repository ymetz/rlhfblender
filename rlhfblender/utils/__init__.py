def process_env_name(env_name):
    """
    Process environment name to be compatible with RLHF-Blender
    """
    if "ALE" in env_name:
        env_name = env_name.replace("/", "-")
    return env_name