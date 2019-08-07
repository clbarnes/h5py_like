import re
from contextlib import contextmanager
from typing import Iterator, Union, Tuple, Optional, Any

import h5py
import numpy as np

from h5py_like import GroupBase, FileMixin, DatasetBase, AttributeManagerBase, Mode
from h5py_like.base import H5ObjectLike
from h5py_like.shape_utils import Indexer, to_slice, thread_read_fn, thread_write_fn

errno_re = re.compile(r"errno = (\d+),")


@contextmanager
def process_oserror():
    try:
        yield
    except OSError as e:
        msg = str(e)
        match = errno_re.search(str(e))
        if match:
            raise OSError(int(match.groups()[0]), str(e))
        elif "file exists" in msg:
            raise FileExistsError(str(e))
        else:
            raise


class AttributeManager(AttributeManagerBase):
    def __init__(self, container, attrs: h5py.AttributeManager):
        self._impl = attrs
        super().__init__(container.mode)

    def __setitem__(self, k, v) -> None:
        with process_oserror():
            self._impl[k] = v

    def __delitem__(self, v) -> None:
        with process_oserror():
            del self._impl[v]

    def __getitem__(self, k):
        return self._impl[k]

    def __len__(self) -> int:
        return len(self._impl)

    def __iter__(self) -> Iterator:
        return iter(self._impl)


class Dataset(DatasetBase):
    def __init__(self, parent, ds: h5py.Dataset):
        basename = ds.name.split("/")[-1]
        super().__init__(basename, parent)
        self._impl = ds
        self._attrs = AttributeManager(self, self._impl.attrs)

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
        with process_oserror():
            return self._impl.chunks

    def resize(self, size: Union[int, Tuple[int, ...]], axis: Optional[int] = None):
        with process_oserror():
            self._impl.resize(size)

    def __getitem__(self, args) -> np.ndarray:
        def inner_fn(offset, shape):
            slices = to_slice(offset, shape)
            return self._impl[slices]

        if self.threads:

            def fn(offset, shape):
                return thread_read_fn(
                    offset,
                    shape,
                    self.chunks or self.shape,
                    self.shape,
                    inner_fn,
                    self.threads,
                )

        else:
            fn = inner_fn

        return self._getitem(args, fn, self._astype)

    def __setitem__(self, args, val):
        def inner_fn(offset, array):
            slices = to_slice(offset, array.shape)
            self._impl[slices] = array

        if self.threads:

            def fn(offset, array):
                return thread_write_fn(
                    offset,
                    array,
                    self.chunks or self.shape,
                    self.shape,
                    inner_fn,
                    self.threads,
                )

        else:
            fn = inner_fn

        with process_oserror():
            return self._setitem(args, val, fn)

    @property
    def attrs(self) -> "AttributeManagerBase":
        return self._attrs

    @property
    def name(self) -> str:
        return self._impl.name


class Group(GroupBase):
    def _create_child_group(self, name) -> GroupBase:
        with process_oserror():
            gr = self._impl.create_group(name)
        return Group(self, gr)

    def _create_child_dataset(self, name, shape=None, dtype=None, data=None, **kwds):
        with process_oserror():
            ds = self._impl.create_dataset(name, shape, dtype, data, **kwds)
        return Dataset(self, ds)

    def _get_child(self, name) -> H5ObjectLike:
        obj = self._impl[name]
        if isinstance(obj, h5py.Group):
            cls = Group
        elif isinstance(obj, h5py.Dataset):
            cls = Dataset
        else:
            raise TypeError()
        return cls(self, obj)

    def __init__(self, parent, group: h5py.Group):
        basename = group.name.split("/")[-1]
        super().__init__(basename, parent)
        self._impl = group
        self._attrs = AttributeManager(self, self._impl.attrs)

    def __setitem__(self, name, obj):
        with process_oserror():
            self._impl[name] = obj._impl

    def copy(
        self,
        source,
        dest,
        name=None,
        shallow=False,
        expand_soft=False,
        expand_external=False,
        expand_refs=False,
        without_attrs=False,
    ):
        with process_oserror():
            self._impl.copy(
                source,
                dest,
                name=None,
                shallow=False,
                expand_soft=False,
                expand_external=False,
                expand_refs=False,
                without_attrs=False,
            )

    @property
    def attrs(self) -> "AttributeManagerBase":
        return self._attrs

    @property
    def name(self) -> str:
        return self._impl.name

    def __delitem__(self, v) -> None:
        with process_oserror():
            del self._impl[v]

    def __len__(self) -> int:
        return len(self._impl)

    def __iter__(self) -> Iterator:
        return iter(self._impl)


class File(FileMixin, Group):
    def __init__(self, name, mode=Mode.default()):
        super().__init__(name, mode)
        with process_oserror():
            self._impl = h5py.File(name, str(mode))
        self._attrs = AttributeManager(self, self._impl.attrs)

    def flush(self):
        return self._impl.flush()

    def close(self):
        return self._impl.close()
