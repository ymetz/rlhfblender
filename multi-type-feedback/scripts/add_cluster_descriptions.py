import os
import pickle as pkl

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def process_pickle_file(file_path):
    with open(file_path, "rb") as file:
        data = pkl.load(file)

    all_obs = []
    all_rewards = []
    for seg in data["segments"]:
        obs = np.array([np.concatenate((s[0].squeeze(0), s[1])) for s in seg])
        rewards = np.array([s[2] for s in seg])
        all_obs.append(obs)
        all_rewards.append(rewards)
    states = np.concatenate(all_obs, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)

    n_clusters = 10000
    batch_size = 1000
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, batch_size=batch_size, random_state=42
    )
    kmeans.fit(states)
    cluster_assignments = kmeans.predict(states)

    cluster_representatives = []
    cluster_rewards = []
    for i in range(n_clusters):
        cluster_mask = cluster_assignments == i
        cluster_states = states[cluster_mask]
        cluster_state_rewards = rewards[cluster_mask]
        if not np.any(np.isnan(np.mean(cluster_states, axis=0))):
            cluster_representatives.append(np.mean(cluster_states, axis=0))
            cluster_rewards.append(np.mean(cluster_state_rewards))
    cluster_representatives = np.array(cluster_representatives)
    cluster_rewards = np.array(cluster_rewards)

    # Separate obs and actions
    obs_dim = data["segments"][0][0][0].squeeze(0).shape[0]
    cluster_description = [
        (rep[:obs_dim], rep[obs_dim:], reward)
        for rep, reward in zip(cluster_representatives, cluster_rewards)
    ]

    # Update the data dictionary with the new cluster description
    data["cluster_description"] = cluster_description

    # Save the updated data back to the pickle file
    with open(file_path, "wb") as file:
        pkl.dump(data, file)


def main():
    directory = "feedback_descript"
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            file_path = os.path.join(directory, filename)
            print(f"Processing {file_path}...")
            process_pickle_file(file_path)
            print(f"Finished processing {file_path}")


if __name__ == "__main__":
    main()
