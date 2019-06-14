from abc import ABC
from copy import deepcopy

import numpy as np

from .common import check_attrs_rw, ds_kwargs


class DatasetTestBase(ABC):
    dataset_kwargs = deepcopy(ds_kwargs)

    def dataset(self, parent, **kwargs):
        kwds = deepcopy(self.dataset_kwargs)
        kwds.update(kwargs)

        return parent.create_dataset(**kwds)

    def test_attrs(self, file_):
        check_attrs_rw(self.dataset(file_).attrs)

    def test_simple_rw(self, file_):
        ds = self.dataset(file_)
        data = np.ones_like(ds)
        ds[...] = data
        np.testing.assert_equal(ds[...], data)
