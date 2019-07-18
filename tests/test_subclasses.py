import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from h5py_like import Mode, FileMixin
from h5py_like.shape_utils import thread_read_fn, thread_write_fn, to_slice
from h5py_like.test_utils import (
    FileTestBase,
    DatasetTestBase,
    GroupTestBase,
    ModeTestBase,
)
from tests.h5_impl import File

logger = logging.getLogger(__name__)


class TestFile(FileTestBase):
    pass


class TestGroup(GroupTestBase):
    pass


class TestDataset(DatasetTestBase):
    pass


class TestMode(ModeTestBase):
    def setup_method(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        try:
            shutil.rmtree(self.tmp_dir)
        except FileNotFoundError:
            pass

    def factory(self, mode: Mode) -> FileMixin:
        fpath = self.tmp_dir / "test.hdf5"
        return File(fpath, mode)


@pytest.mark.parametrize("threads", [1, 4], ids=lambda x: f"threads{x}")
def test_threaded_read(file_, threads):
    shape = (20, 20)
    chunks = (10, 10)
    data = np.arange(np.product(shape), dtype=int).reshape(shape)
    ds = file_.create_dataset("ds", data=data, chunks=chunks)

    def fn(start_coord, block_shape):
        logger.warning("slicing at start %s, shape %s", start_coord, block_shape)
        slicing = to_slice(start_coord, block_shape)
        sliced = ds[slicing]
        logger.warning("sliced at start %s, got shape %s", start_coord, sliced.shape)
        return sliced

    read_start = (5, 5)
    read_shape = (10, 10)

    out = thread_read_fn(read_start, read_shape, ds.chunks, data.shape, fn, threads)
    exp_slicing = to_slice(read_start, read_shape)
    expected = data[exp_slicing]
    assert np.array_equal(expected, out)


@pytest.mark.parametrize("threads", [1, 4], ids=lambda x: f"threads{x}")
def test_threaded_write(file_, threads):
    shape = (20, 20)
    chunks = (10, 10)

    data = np.ones(shape, dtype=int)

    ds = file_.create_dataset("ds", data=data, chunks=chunks)

    def fn(offset, arr):
        logger.warning("writing array of shape %s to offset %s", arr.shape, offset)
        slicing = to_slice(offset, arr.shape)
        ds[slicing] = arr
        logger.warning("wrote array of shape %s to offset %s", arr.shape, offset)

    write_start = (5, 5)
    write_shape = (10, 10)

    write_arr = np.ones(write_shape, dtype=int) * 9

    thread_write_fn(write_start, write_arr, chunks, ds.shape, fn, threads)

    expected = data.copy()
    expected[5:15, 5:15] = write_arr
    actual = ds[:]
    assert np.array_equal(expected, actual)
