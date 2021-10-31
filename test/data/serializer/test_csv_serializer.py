import unittest
from dataclasses import dataclass

from data.serializer.csv_serializer import CsvSerializer
from utils.path_utils import get_root_project_path


@dataclass
class FeatureTestClass:
    label: str
    idx: int


class TestDataHolder(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.features: list[FeatureTestClass] = [
            FeatureTestClass('a', 1),
            FeatureTestClass('b', 0),
        ]

    def test_save_and_load_data(self):
        root_tests_path = get_root_project_path() / 'cached-data/tests'
        root_tests_path.mkdir(parents=True, exist_ok=True)
        test_file = root_tests_path / 'test.csv'

        CsvSerializer.save(test_file, self.features, FeatureTestClass)
        # stats = test_file.stat()

        loaded_features = CsvSerializer.load(test_file, FeatureTestClass)

        self.assertEqual(self.features, loaded_features)

