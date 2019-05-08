import sys
from abc import abstractmethod, ABC

import numpy as np
from typing import Tuple, Optional, Any, Union, Iterator

from h5py_like.shape_utils import getitem, setitem
from .common import H5ObjectLike, ReadOnlyException


class Dataset(H5ObjectLike, ABC):
    """
        Represents an HDF5-like dataset
    """

    @abstractmethod
    def astype(self, dtype):
        """ Get a context manager allowing you to perform reads to a
        different destination type, e.g.:
        >>> with dataset.astype('f8'):
        ...     double_precision = dataset[0:100:2]
        """
        raise NotImplementedError()

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
        return np.prod(self.shape, dtype=np.intp)

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
    def _read_array(self, start, shape) -> np.ndarray:
        """Internal method which reads a contiguous array of shape ``shape`` from offset ``start``"""
        pass

    def __getitem__(self, args) -> np.ndarray:
        """Read a slice from the HDF5-like dataset.
        Supports slices, integers, negative indexing, and striding.
        Strided queries read "stepped-over" data and then stride it in a second pass.
        Does not support logical indexing.
        """
        return getitem(args, self.shape, self.dtype, self._read_array)

    @abstractmethod
    def _write_array(self, start: Tuple[int, ...], arr: np.ndarray):
        """Internal method which writes a contiguous array at offset ``start``"""
        pass

    def __setitem__(self, args, val):
        """ Write to the HDF5-like dataset from a Numpy array.
        NumPy's broadcasting rules are honored, for "simple" indexing
        (slices and integers).  For advanced indexing, the shapes must
        match.
        """
        if self.file.mode.READ_ONLY:
            raise ReadOnlyException(
                "Cannot change the attributes of a read-only object"
            )

        setitem(args, val, self.shape, self.dtype, self._write_array)

    @abstractmethod
    def read_direct(self, dest, source_sel=None, dest_sel=None):
        """ Read data directly from HDF5 into an existing NumPy array.
        The destination array must be C-contiguous and writable.
        Selections must be the output of numpy.s_[<args>].
        Broadcasting is supported for simple indexing.
        """
        raise NotImplementedError()

    @abstractmethod
    def write_direct(self, source, source_sel=None, dest_sel=None):
        """ Write data directly to HDF5 from a NumPy array.
        The source array must be C-contiguous.  Selections must be
        the output of numpy.s_[<args>].
        Broadcasting is supported for simple indexing.
        """
        raise NotImplementedError()

    def __array__(self, dtype=None):
        """ Create a Numpy array containing the whole dataset.  DON'T THINK
        THIS MEANS DATASETS ARE INTERCHANGEABLE WITH ARRAYS.  For one thing,
        you have to read the whole dataset every time this method is called.
        """
        arr = np.empty(self.shape, dtype=self.dtype if dtype is None else dtype)

        # Special case for (0,)*-shape datasets
        if np.product(self.shape, dtype=np.ulonglong) == 0:
            return arr

        self.read_direct(arr)
        return arr
