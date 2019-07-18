import sys
from abc import abstractmethod, ABC
from contextlib import contextmanager

import numpy as np
from typing import Tuple, Optional, Any, Union, Iterator

from .shape_utils import IndexableArrayLike
from .common import Mode, classname
from h5py_like.base import H5ObjectLike, mutation


class DatasetBase(H5ObjectLike, IndexableArrayLike, ABC):
    """Represents an HDF5-like dataset

    If the dataset implementation supports threaded reads and writes, the number of
    threads should be controlled by the ``threads`` attribute.
    """

    threads = None
    _is_file = False

    def __init__(self, mode: Mode = Mode.default()):
        self._astype = None
        super().__init__(mode)

    @contextmanager
    def astype(self, dtype):
        """ Get a context manager allowing you to perform reads to a
        different destination type, e.g.:
        >>> with dataset.astype('f8'):
        ...     double_precision = dataset[0:100:2]
        """
        self._astype = np.dtype(dtype)
        yield
        self._astype = None

    @property
    @abstractmethod
    def dims(self):
        """ Access dimension scales attached to this dataset. """
        raise NotImplementedError()

    @property
    def ndim(self) -> int:
        """Numpy-style attribute giving the number of dimensions"""
        return len(self.shape)

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """Numpy-style shape tuple giving dataset dimensions"""
        raise NotImplementedError()

    @shape.setter
    def shape(self, shape) -> None:
        self.resize(shape)

    @property
    def size(self) -> int:
        """Numpy-style attribute giving the total dataset size"""
        return np.prod(self.shape, dtype=np.intp).item()

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """Numpy dtype representing the datatype"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def maxshape(self) -> Tuple[int, ...]:
        """Shape up to which this dataset can be resized.  Axes with value
        None have no resize limit. """
        raise NotImplementedError()

    @property
    @abstractmethod
    def fillvalue(self) -> Any:
        """Fill value for this dataset (0 by default)"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def chunks(self) -> Optional[Tuple[int, ...]]:
        """Dataset chunks (or None)"""
        raise NotImplementedError()

    def _sanitize_resize(
        self, size: Union[int, Tuple[int, ...]], axis: Optional[int] = None
    ) -> Tuple[int, ...]:
        if self.chunks is None:
            raise TypeError("Only chunked datasets can be resized")

        if axis is not None:
            if not (0 <= axis < self.ndim):
                raise ValueError("Invalid axis (0 to %s allowed)" % (self.ndim - 1))
            try:
                newlen = int(size)
            except TypeError:
                raise TypeError("Argument must be a single int if axis is specified")
            size = list(self.shape)
            size[axis] = newlen

        return tuple(size)

    @abstractmethod
    def resize(self, size: Union[int, Tuple[int, ...]], axis: Optional[int] = None):
        """ Resize the dataset, or the specified axis.
        The dataset must be stored in chunked format; it can be resized up to
        the "maximum shape" (keyword maxshape) specified at creation time.
        The rank of the dataset cannot be changed.
        "Size" should be a shape tuple, or if an axis is specified, an integer.
        BEWARE: This functions differently than the NumPy resize() method!
        The data is not "reshuffled" to fit in the new shape; each axis is
        grown or shrunk independently.  The coordinates of existing data are
        fixed.

        self._sanitize_resize may help validate the resize operation.
        """
        # checked_size = self._sanitize_resize(size, axis)
        raise NotImplementedError()

    def __len__(self) -> int:
        """ The size of the first axis.  TypeError if scalar.
        Limited to 2**32 on 32-bit systems; Dataset.len() is preferred.
        """
        size = self.len()
        if size > sys.maxsize:
            raise OverflowError(
                "Value too big for Python's __len__; use Dataset.len() instead."
            )
        return size

    def len(self) -> int:
        """ The size of the first axis.  TypeError if scalar.
        Use of this method is preferred to len(dset), as Python's built-in
        len() cannot handle values greater then 2**32 on 32-bit systems.
        """
        shape = self.shape
        if len(shape) == 0:
            raise TypeError("Attempt to take len() of scalar dataset")
        return shape[0]

    def __iter__(self) -> Iterator[np.ndarray]:
        """ Iterate over the first axis.  TypeError if scalar.
        BEWARE: Modifications to the yielded data are *NOT* written to file.
        """
        for i in range(self.len()):
            yield self[i]

    @abstractmethod
    def __getitem__(self, args) -> np.ndarray:
        """Read a slice from the HDF5-like dataset.

        See h5py_like.shape_utils.getitem for a utility function which supports
        positive and negative integers/slices, striding, ellipses (explicit and
        implicit), reading scalars, and reading arrays with 0-length dimensions.

        Don't forget to use ``self._astype or self.dtype`` when setting the read dtype,
        to be compatible with the ``Dataset.astype`` context manager.

        The _getitem method is a helper for this.
        """

    @abstractmethod
    def __setitem__(self, args, val):
        """ Write to the HDF5-like dataset from a Numpy array.

        The _setitem method is a helper for this.
        """

    def read_direct(self, dest, source_sel=None, dest_sel=None):
        """ Read data directly from the underlying array into an existing NumPy array.
        The destination array must be C-contiguous and writable.
        Selections must be the output of numpy.s_[<args>].

        The default implementation just reads and writes as usual, and therefore has no
        speedups.
        """
        if source_sel is None:
            source_sel = Ellipsis

        if dest_sel is None:
            dest[...] = self[source_sel]
        else:
            dest[dest_sel] = self[source_sel]

    @mutation
    def write_direct(self, source, source_sel=None, dest_sel=None):
        """ Write data directly to HDF5 from a NumPy array.
        The source array must be C-contiguous.  Selections must be
        the output of numpy.s_[<args>].

        The default implementation just reads and writes as usual, and therefore has no
        speedups.
        """
        if source_sel is None:
            source_sel = Ellipsis

        if dest_sel is None:
            self[...] = source[source_sel]
        else:
            self[dest_sel] = source[source_sel]

    def __array__(self, dtype=None):
        """ Create a Numpy array containing the whole dataset.  DON'T THINK
        THIS MEANS DATASETS ARE INTERCHANGEABLE WITH ARRAYS.  For one thing,
        you have to read the whole dataset every time this method is called.
        """
        arr = self[...]
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def __str__(self):
        return (
            f"<{classname(self)}(name='{self.name}', "
            f"shape={self.shape}, dtype={self.dtype}, file={self.file})>"
        )

    def __eq__(self, other):
        return all(
            (
                isinstance(other, type(self)),
                self.name == other.name,
                self.parent == other.parent,
            )
        )
