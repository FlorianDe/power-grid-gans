from pathlib import Path

from matplotlib import pyplot as plt

from experiments.utils import get_experiments_folder


def set_latex_plot_params():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"]
    })  # Computer Modern
    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "sans-serif",
    #     "font.sans-serif": ["Helvetica"]
    # })  # Helvetica

    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


def get_generated_images_path_folder() -> Path:
    return get_experiments_folder().joinpath("generated_images")