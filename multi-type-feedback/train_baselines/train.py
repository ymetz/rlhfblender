from train_baselines.train import train
import minigrid

minigrid.register_minigrid_envs()


if __name__ == "__main__":
    train()
