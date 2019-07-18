from abc import ABC
from copy import deepcopy

import numpy as np

from h5py_like import GroupBase, Name
from .common import check_attrs_rw, ds_kwargs, LoggedClassMixin

random_state = np.random.RandomState(2019)


class GroupLikeTestsMixin(LoggedClassMixin, ABC):
    dataset_kwargs = deepcopy(ds_kwargs)

    def test_attrs(self, obj: GroupBase):
        check_attrs_rw(obj.attrs)

    def test_create_group(self, obj: GroupBase):
        name = "another"
        g2 = obj.create_group(name)
        assert g2.parent == obj
        assert g2.name == str(Name(obj.name) / name)
        assert name in obj

    def test_create_dataset(self, obj: GroupBase):
        dsk = self.dataset_kwargs
        ds = obj.create_dataset(**dsk)
        assert ds.name == str(Name(obj.name) / dsk["name"])
        assert ds.shape == dsk["shape"]
        assert ds.dtype == dsk["dtype"]

    def test_create_dataset_from_data(self, obj: GroupBase):
        dsk = deepcopy(self.dataset_kwargs)
        data = np.ones(dsk.pop("shape"), dtype=dsk.pop("dtype"))
        del dsk["name"]
        ds = obj.create_dataset("test_ds", data=data, **dsk)
        np.testing.assert_array_equal(ds[...], data)


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
        super().test_create_dataset_from_data(self.group(file_))
