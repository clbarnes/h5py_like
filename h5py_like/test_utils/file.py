from abc import ABC

from .group import GroupLikeTestsMixin


class FileTestBase(GroupLikeTestsMixin, ABC):
    def test_attrs(self, file_):
        super().test_attrs(file_)

    def test_create_group(self, file_):
        super().test_create_group(file_)

    def test_create_dataset(self, file_):
        super().test_create_dataset(file_)

    def test_create_dataset_from_data(self, file_):
        super().test_create_dataset_from_data(file_)
