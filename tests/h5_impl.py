from typing import Iterator, Union, Tuple, Optional, Any

import h5py
import numpy as np

from h5py_like import GroupBase, FileMixin, DatasetBase, AttributeManagerBase, Mode
from h5py_like.base import H5ObjectLike
from h5py_like.shape_utils import Indexer


class AttributeManager(AttributeManagerBase):
    def __init__(self, parent, attrs: h5py.AttributeManager):
        self._impl = attrs
        super().__init__(parent.mode)

    def __setitem__(self, k, v) -> None:
        self._impl[k] = v

    def __delitem__(self, v) -> None:
        del self._impl[v]

    def __getitem__(self, k):
        return self._impl[k]

    def __len__(self) -> int:
        return len(self._impl)

    def __iter__(self) -> Iterator:
        return iter(self._impl)


class Dataset(DatasetBase):
    def __init__(self, parent, ds: h5py.Dataset):
        self._parent = parent
        self._impl = ds
        self._attrs = AttributeManager(self, self._impl.attrs)
        super().__init__(parent.mode)

    @property
    def _indexer(self):
        return Indexer(self.shape)

    @property
    def dims(self):
        return self._impl.dims

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._impl.shape

    @property
    def dtype(self) -> np.dtype:
        return self._impl.dtype

    @property
    def maxshape(self) -> Tuple[int, ...]:
        return self._impl.maxshape

    @property
    def fillvalue(self) -> Any:
        return self._impl.fillvalue

    @property
    def chunks(self) -> Optional[Tuple[int, ...]]:
        return self._impl.chunks

    def resize(self, size: Union[int, Tuple[int, ...]], axis: Optional[int] = None):
        self._impl.resize(size)

    def __getitem__(self, args) -> np.ndarray:
        def fn(offset, shape):
            slices = tuple(slice(o, o+s) for o, s in zip(offset, shape))
            return self._impl[slices]

        return self._getitem(args, fn, self._astype)

    def __setitem__(self, args, val):
        def fn(offset, array):
            slices = tuple(slice(o, o+s) for o, s in zip(offset, array.shape))
            self._impl[slices] = array

        return self._setitem(args, val, fn)

    @property
    def attrs(self) -> "AttributeManagerBase":
        return self._attrs

    @property
    def name(self) -> str:
        return self._impl.name

    @property
    def parent(self):
        return self._parent


class Group(GroupBase):
    def __init__(self, parent, group: h5py.Group):
        self._parent = parent
        self._impl = group
        self._attrs = AttributeManager(self, self._impl.attrs)
        super().__init__(parent.mode)

    def create_group(self, name) -> GroupBase:
        gr = self._impl.create_group(name)
        return Group(self, gr)

    def create_dataset(self, name, shape=None, dtype=None, data=None, **kwds) -> DatasetBase:
        ds = self._impl.create_dataset(name, shape=shape, dtype=dtype, data=data, **kwds)
        return Dataset(self, ds)

    def __getitem__(self, name) -> H5ObjectLike:
        item = self._impl[name]

        items = [item]
        while items[-1].parent != items[-1]:
            items.append(items[-1].parent)

        parent = self.file
        items.pop()  # we already know the file
        if items:
            last, *groups = items
        else:
            return parent

        for group in reversed(groups):
            parent = Group(parent, group)

        if isinstance(last, h5py.Group):
            cls = Group
        elif isinstance(last, h5py.Dataset):
            cls = Dataset
        else:
            raise TypeError

        return cls(parent, last)

    def __setitem__(self, name, obj):
        self._impl[name] = obj._impl

    def copy(self, source, dest, name=None, shallow=False, expand_soft=False, expand_external=False, expand_refs=False,
             without_attrs=False):
        self._impl.copy(source, dest, name=None, shallow=False, expand_soft=False, expand_external=False, expand_refs=False,
             without_attrs=False)

    @property
    def attrs(self) -> "AttributeManagerBase":
        return self._impl.attrs

    @property
    def name(self) -> str:
        return self._impl.name

    @property
    def parent(self):
        return self._parent

    def __delitem__(self, v) -> None:
        del self._impl[v]

    def __len__(self) -> int:
        return len(self._impl)

    def __iter__(self) -> Iterator:
        return iter(self._impl)


class File(FileMixin, Group):
    def __init__(self, f: h5py.File):
        super().__init__(f.name, mode=Mode.from_str(f.mode))
        self._impl = f
        self._attrs = AttributeManager(self, self._impl.attrs)

    def flush(self):
        return self._impl.flush()

    def close(self):
        return self._impl.close()
