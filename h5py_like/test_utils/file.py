from abc import ABC, abstractmethod
from contextlib import contextmanager

import pytest

from h5py_like import Mode, FileMixin
from h5py_like.common import ReadOnlyException
from .group import GroupLikeTestsMixin


class FileTestBase(GroupLikeTestsMixin, ABC):
    def cls_ext(self, file_obj):
        return type(file_obj), file_obj.filename.suffix

    def test_attrs(self, file_):
        super().test_attrs(file_)

    def test_create_group(self, file_):
        super().test_create_group(file_)

    def test_create_dataset(self, file_):
        super().test_create_dataset(file_)

    def test_create_dataset_from_data(self, file_):
        super().test_create_dataset_from_data(file_)


class ModeTestBase(ABC):
    @abstractmethod
    @contextmanager
    def factory(self, mode: Mode) -> FileMixin:
        """Create a File at a location deterministic per test, as a context manager.

        Do not clean up the File in this method.
        """
        pass

    def ensure_exists(self, group=None):
        try:
            with self.factory(Mode.READ_WRITE_CREATE) as f:
                if group is not None:
                    g = f.create_group(group)
        except FileExistsError:
            pass

    def test_read_write_create(self):
        mode = Mode.READ_WRITE_CREATE

        with self.factory(mode) as f:
            f.create_group("group")

        with self.factory(mode) as f:
            assert "group" in f

    def test_create(self):
        mode = Mode.CREATE

        with self.factory(mode):
            pass

        with pytest.raises(FileExistsError):
            with self.factory(mode):
                pass

    def test_read_only(self):
        mode = Mode.READ_ONLY
        with pytest.raises(FileNotFoundError):
            with self.factory(mode):
                pass

        self.ensure_exists()

        with self.factory(mode) as f:
            with pytest.raises(ReadOnlyException):
                g = f.create_group("group")

    def test_create_truncate(self):
        mode = Mode.CREATE_TRUNCATE

        with self.factory(mode):
            pass

        self.ensure_exists("group")

        with self.factory(mode) as f:
            assert "group" not in f

    def test_read_write(self):
        mode = Mode.READ_WRITE

        with pytest.raises(FileNotFoundError):
            with self.factory(mode):
                pass

        self.ensure_exists()

        with self.factory(mode) as f:
            g = f.create_group("group")
