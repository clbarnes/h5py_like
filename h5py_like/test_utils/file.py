from abc import ABC

from h5py_like.test_utils.group import GroupLikeTestsMixin


class FileTestBase(GroupLikeTestsMixin, ABC):
    def test_attrs(self, file_):
        super().test_attrs(file_)

    def test_create_group(self, file_):
        super().test_attrs(file_)

    def test_create_dataset(self, file_):
        super().test_attrs(file_)

    def test_create_dataset_from_data(self, file_):
        super().test_attrs(file_)
