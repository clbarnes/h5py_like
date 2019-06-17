from abc import ABC
from copy import deepcopy

import numpy as np
import pytest

from .common import check_attrs_rw, ds_kwargs


random_state = np.random.RandomState(1991)


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

    @pytest.mark.parametrize(
        "slice_args",
        [
            np.index_exp[...],
            np.index_exp[::-1, ::-1, ::-1],
            np.index_exp[4:8, 1:19, -5:-1],
            np.index_exp[2, 2, 2],
            np.index_exp[:, :, np.newaxis, :],
            np.index_exp[1:5],
            np.index_exp[..., 1:5],
            np.index_exp[..., 1:5, ...],
        ],
    )
    def test_slicing(self, slice_args, file_):
        data = random_state.random_sample((20, 20, 20))
        self.dataset_kwargs = deepcopy(self.dataset_kwargs)
        for key in ("shape", "dtype"):
            del self.dataset_kwargs[key]
        ds = self.dataset(file_, data=data)

        impl_e = None
        np_e = None

        try:
            impl = ds[slice_args]
        except Exception as e:
            impl_e = e

        try:
            numpy = data[slice_args]
        except Exception as e:
            np_e = e

        if np_e or impl_e:
            assert isinstance(impl_e, type(np_e))
        else:
            np.testing.assert_allclose(impl, numpy)
