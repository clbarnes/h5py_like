from abc import ABC
from copy import deepcopy

import numpy as np
import pytest

from h5py_like.shape_utils import to_slice
from .common import check_attrs_rw, ds_kwargs, LoggedClassMixin

random_state = np.random.RandomState(1991)


class DatasetTestBase(LoggedClassMixin, ABC):
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

    def test_asarray(self, file_):
        ds = self.dataset(file_)
        arr = np.asarray(ds)
        assert arr.shape == ds.shape
        assert arr.dtype == ds.dtype
        assert np.allclose(arr, ds[:])


chunked = deepcopy(ds_kwargs)
chunked["chunks"] = (5, 5, 5)


class ThreadedDatasetTestBase(DatasetTestBase):
    dataset_kwargs = deepcopy(chunked)

    @pytest.mark.parametrize("threads", [1, 4], ids=lambda x: f"threads{x}")
    def test_threaded_read(self, file_, threads):
        shape = (20, 20)
        chunks = (10, 10)
        data = np.arange(np.product(shape), dtype=int).reshape(shape)
        ds = self.dataset(file_, data, chunks=chunks)
        ds.threads = threads

        read_start = (5, 5)
        read_shape = (10, 10)

        slicing = to_slice(read_start, read_shape)
        assert np.array_equal(ds[slicing], data[slicing])

    @pytest.mark.parametrize("threads", [1, 4], ids=lambda x: f"threads{x}")
    def test_threaded_write(self, file_, threads):
        shape = (20, 20)
        chunks = (10, 10)

        data = np.ones(shape, dtype=int)

        ds = self.dataset(file_, data, chunks=chunks)
        ds.threads = threads

        write_start = (5, 5)
        write_shape = (10, 10)
        write_slice = to_slice(write_start, write_shape)

        data[write_slice] = 9
        ds[write_slice] = 9

        assert np.array_equal(data, ds[:])
