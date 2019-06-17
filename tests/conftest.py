import pytest

from h5py_like import Mode
from .h5_impl import File


@pytest.fixture
def file_(tmp_path):
    with File(tmp_path / "test.hdf5", Mode.READ_WRITE_CREATE) as f:
        yield f
