from abc import ABC
from copy import deepcopy

import numpy as np
import pytest

from .common import check_attrs_rw, ds_kwargs


random_state = np.random.RandomState(1991)


class DatasetTestBase(ABC):
    dataset_kwargs = deepcopy(ds_kwargs)

    def dataset(self, parent, data=None, **kwargs):
        kwds = deepcopy(self.dataset_kwargs)
        kwds.update(kwargs)

        if data is not None:
            for key in ("shape", "dtype"):
                del kwds[key]
        kwds["data"] = data

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

    def test_broadcast_scalar(self, file_):
        ds = self.dataset(file_)
        ds[:] = 5
        expected = np.full_like(ds, 5)
        np.testing.assert_allclose(ds, expected)

    @pytest.mark.parametrize(
        "slice_args,total",
        [(np.index_exp[...], 10 ** 3), (np.index_exp[:5, :5, :5], 5 ** 3)],
    )
    def test_setitem_total(self, slice_args, total, file_):
        ds = self.dataset(file_)
        ds[slice_args] = 1
        assert ds[:].sum() == total

    def test_setitem(self, file_):
        data = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])

        ds = self.dataset(file_, data=data, chunks=(3, 1))
        ds[:2, :2] = np.array([[5, 6], [7, 8]], dtype=data.dtype)

        np.testing.assert_allclose(
            ds[:], np.array([[5, 6, 3], [7, 8, 3], [1, 2, 3]], dtype=data.dtype)
        )
