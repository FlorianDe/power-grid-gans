from pathlib import Path

from src.utils.path_utils import get_root_project_path


def get_experiments_folder() -> Path:
    return get_root_project_path().joinpath('runs').joinpath('experiments')
