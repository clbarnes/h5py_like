from abc import ABC

import numpy as np

from .common import check_attrs_rw


class DatasetTestBase(ABC):
    dataset_name = "dataset"
    dataset_shape = (10, 10, 10)
    dataset_dtype = np.dtype('uint16')

    def dataset(self, parent, name=None, shape=None, dtype=None, **kwargs):
        if name is None:
            name = self.dataset_name
        if shape is None:
            shape = self.dataset_shape
        if dtype is None:
            dtype = self.dataset_dtype

        return parent.create_dataset(
            name,
            shape=shape,
            dtype=dtype,
            **kwargs,
        )

    def test_attrs(self, file_):
        check_attrs_rw(self.dataset(file_).attrs)

    def test_simple_rw(self, file_):
        ds = self.dataset(file_)
        data = np.ones_like(ds)
        ds[...] = data
        np.testing.assert_equal(ds[...], data)
