import unittest
from dataclasses import dataclass

from data.serializer.csv_serializer import CsvSerializer
from utils.path_utils import get_root_project_path


@dataclass
class Feature:
    label: str
    idx: int


class TestDataHolder(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.features: list[Feature] = [
            Feature('a', 1),
            Feature('b', 0),
        ]

    def test_save_and_load_data(self):
        root_tests_path = get_root_project_path() / 'cached-data/tests'
        root_tests_path.mkdir(parents=True, exist_ok=True)
        test_file = str(root_tests_path / 'test.csv')

        CsvSerializer.save(self.features, test_file, Feature)

        loaded_features = CsvSerializer.load(test_file, Feature)

        self.assertEqual(self.features, loaded_features)

