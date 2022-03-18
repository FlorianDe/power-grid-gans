from pathlib import Path

from matplotlib import pyplot as plt

from src.utils.path_utils import get_root_project_path


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


def get_experiments_folder() -> Path:
    return get_root_project_path().joinpath('runs').joinpath('experiments')
