import minigrid

from train_baselines.train import train

minigrid.register_minigrid_envs()


if __name__ == "__main__":
    train()
