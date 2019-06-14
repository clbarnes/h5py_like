import h5py
import pytest

from .h5_impl import File


@pytest.fixture
def file_(tmp_path):
    with h5py.File(tmp_path / "test.hdf5") as f:
        yield File(f)
