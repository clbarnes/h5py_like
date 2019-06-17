import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from h5py_like import Mode, FileMixin
from h5py_like.test_utils import FileTestBase, DatasetTestBase, GroupTestBase, ModeTestBase
from tests.h5_impl import File


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

    @contextmanager
    def factory(self, mode: Mode) -> FileMixin:
        fpath = self.tmp_dir / "test.hdf5"
        with File(fpath, mode) as f:
            yield f
