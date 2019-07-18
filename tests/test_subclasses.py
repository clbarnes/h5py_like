import logging
import shutil
import tempfile
from pathlib import Path

from h5py_like import Mode, FileMixin
from h5py_like.test_utils import (
    FileTestBase,
    GroupTestBase,
    ModeTestBase,
    ThreadedDatasetTestBase,
)
from tests.h5_impl import File

logger = logging.getLogger(__name__)


class TestFile(FileTestBase):
    pass


class TestGroup(GroupTestBase):
    pass


class TestThreadedDataset(ThreadedDatasetTestBase):
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
