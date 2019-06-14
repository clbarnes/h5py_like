from abc import ABC

import numpy as np

from h5py_like import GroupBase
from .common import check_attrs_rw


class GroupLikeTestsMixin(ABC):
    def test_attrs(self, obj: GroupBase):
        check_attrs_rw(obj.attrs)

    def test_create_group(self, obj: GroupBase):
        name = "another"
        g2 = obj.create_group(name)
        assert g2.parent == obj
        assert g2.name == f"{obj.name}/{name}"
        assert name in obj

    def test_create_dataset(self, obj: GroupBase):
        name = "dataset"
        shape = (10, 10, 10)
        dtype = np.dtype('uint16')
        ds = obj.create_dataset(name, shape=shape, dtype=dtype)
        assert ds.name == f"{obj.name}/{name}"
        assert ds.shape == shape
        assert ds.dtype == dtype

    def test_create_dataset_from_data(self, obj: GroupBase):
        data = np.ones((10, 10), dtype=np.dtype('uint64'))
        ds = obj.create_dataset("test_ds", data=data)
        np.testing.assert_equal(ds[...], data)


class GroupTestBase(GroupLikeTestsMixin, ABC):
    group_name = "group"

    def group(self, parent, name=None, **kwargs):
        if name is None:
            name = self.group_name

        return parent.create_group(name, **kwargs)

    def test_attrs(self, file_: GroupBase):
        super().test_attrs(self.group(file_))

    def test_create_group(self, file_: GroupBase):
        super().test_create_group(self.group(file_))

    def test_create_dataset(self, file_: GroupBase):
        super().test_create_dataset(self.group(file_))

    def test_create_dataset_from_data(self, file_: GroupBase):
        super().test_create_dataset(self.group(file_))
