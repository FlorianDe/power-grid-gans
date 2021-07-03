import os
from pathlib import Path
from zipfile import ZipFile


def is_root(path: Path) -> bool:
    # rules to determine whether this is the root dir, currently hardcoded
    correct_root_path_name = path.name == 'power-grid-gans'
    includes_requirements_file = path.joinpath('requirements.txt').exists()

    return correct_root_path_name and includes_requirements_file


def get_root_project_path() -> Path:
    current = Path(__file__).resolve()
    while True:
        parent = current.parent
        if current == parent:
            raise Exception('Could not determine the project root path while traversing the parents.')
        current = parent
        if is_root(current):
            break
    return current


def unzip(path: str, zip_file_name: str, target: str):
    with ZipFile(os.path.join(path, zip_file_name), "r") as zip_ref:
        zip_ref.extractall(os.path.join(path, target))


if __name__ == '__main__':
    get_root_project_path()
