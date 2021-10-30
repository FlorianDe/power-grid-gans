import unittest

import numpy
from torch.utils.data import DataLoader

from data.data_holder import DataHolder


class TestDataHolder(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.data = numpy.array([
            [0.1, 1.0],
            [3.5, 5.0],
            [10.2, 7.5]
        ])
        self.feature_labels = ['a', 'b']
        self.x = numpy.array([
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 1]
        ])
        self.dh = DataHolder(self.data, self.feature_labels, self.x)

        data_loader = DataLoader(
            self.dh.get_tensor_dataset(),
            batch_size=1,
            shuffle=True
        )
        for i, (real_data, labels) in enumerate(data_loader):
            print(f'{i}: {labels} -> {real_data}')

    def test_get_tensor_dataset(self):
        self.assertTrue(numpy.allclose(self.dh.data, self.data))

    def test_get_feature_size(self):
        self.assertTrue(self.dh.get_feature_size(), len(self.feature_labels))

    def test_get_feature_labels(self):
        self.assertTrue(self.dh.get_feature_labels(), self.feature_labels)
