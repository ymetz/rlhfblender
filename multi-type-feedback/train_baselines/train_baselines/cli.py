import sys

from train_baselines.enjoy import enjoy
from train_baselines.plots import all_plots, plot_from_file, plot_train
from train_baselines.train import train


def main():
    script_name = sys.argv[1]
    # Remove script name
    del sys.argv[1]
    # Execute known script
    known_scripts = {
        "train": train,
        "enjoy": enjoy,
        "plot_train": plot_train,
        "plot_from_file": plot_from_file,
        "all_plots": all_plots,
    }
    if script_name not in known_scripts.keys():
        raise ValueError(f"The script {script_name} is unknown, please use one of {known_scripts.keys()}")
    known_scripts[script_name]()


if __name__ == "__main__":
    main()
