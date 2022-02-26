from pathlib import Path

from experiments.utils import get_experiments_folder


def get_generated_images_path_folder() -> Path:
    return get_experiments_folder().joinpath("00_generated_images")