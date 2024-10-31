import numpy as np

from rlhfblender.data_collection import RecordedEpisodesContainer


def process_metrics(benchmark_results: RecordedEpisodesContainer) -> dict:
    """
    Compute additional metrics on a per model/per benchmark basis
    :param benchmark_results: (RecordedEpisodes) Container of benchmark results
    :return metrics: (dict) Metrics
    """

    avg_reward = np.mean(benchmark_results.episode_rewards)
    avg_length = np.mean(benchmark_results.episode_lengths)
    avg_entropy = np.mean([info.item().get("entropy", 0.0) for info in benchmark_results.infos])
    avg_value = np.mean([info.item().get("value", 0.0) for info in benchmark_results.infos])
    avg_action_prob = (
        1.0 if len(benchmark_results.probs.shape) == 1 else np.mean(np.max(benchmark_results.probs, axis=1)).astype(float)
    )
    if len(benchmark_results.rewards) > 0:
        avg_reward_freq = np.count_nonzero(benchmark_results.rewards) / benchmark_results.rewards.shape[0]
    else:
        avg_reward_freq = 0.0

    metrics = {
        "avg_reward": avg_reward,
        "avg_length": avg_length,
        "avg_entropy": avg_entropy,
        "avg_value": avg_value,
        "avg_action_prob": avg_action_prob,
        "avg_reward_freq": avg_reward_freq,
    }
    return metrics
