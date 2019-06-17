from __future__ import annotations

import itertools
import numbers
from abc import ABC, abstractmethod
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from math import ceil
from typing import Callable, Tuple, TypeVar, Union, NamedTuple, List

import numpy as np

DEFAULT_THREADS = 4


class NullSlicingException(ValueError):
    def __init__(self, msg="", out_shape=None):
        super().__init__(msg)
        self.out_shape = out_shape


class StartLenStride(NamedTuple):
    start: int
    len: int
    stride: int


def slice_to_start_len_stride(slice_: slice, max_len: int) -> StartLenStride:
    """
    Convert a slice object, possibly with None or negative members, into positive integers for start, length, and stride.

    Raises NullSlicingException if there would be no results returned from this indexing.

    If stride is negative, start is still the lower corner and length is still positive: the negative stride
    must be done after the data read.

    :param slice_:
    :param max_len: maximum length of the dimension
    :return: tuple of positive integer start, length, +ve or -ve stride
    :raises: NullSlicingException
    """
    start, old_stop, stride = slice_.indices(max_len)
    stop = max(old_stop, 0)
    if stop > old_stop:
        start += abs(old_stop)
    shape = stop - start

    if shape == 0 or shape * stride < 0:
        raise NullSlicingException("Requested slice is empty")

    return StartLenStride(min(start, stop), abs(shape), stride)


def int_to_start_len_stride(i: int, max_len: int) -> StartLenStride:
    """
    Convert an integer index, possibly negative, into positive integers for start, length, and stride.

    Raises NullSlicingException if there would be no results returned from this indexing

    :param i: integer index
    :param max_len: maximum length of the dimension
    :return: tuple of positive integer start, length, stride
    """
    if -max_len < i < 0:
        begin = i + max_len
    elif i >= max_len or i < -max_len:
        raise IndexError(
            "index {} is out of bounds for axis with size".format(i, max_len)
        )
    else:
        begin = i

    return StartLenStride(begin, 1, 1)


class Roi(NamedTuple):
    """ROI described by positive integers.
    """

    start: Tuple[int, ...]
    read_shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    out_shape: Tuple[int, ...]


ConcreteSliceLike = Union[slice, int]
SliceLike = Union[ConcreteSliceLike, type(Ellipsis), np.newaxis]
SliceArgs = TypeVar("SliceArgs", SliceLike, Tuple[SliceLike])


class Indexer:
    type_msg = (
        "Advanced selection inappropriate. "
        "Only numbers, slices (`:`), ellipsis (`...`), and np.newaxis (None) "
        "are valid indices (or tuples thereof); got {}"
    )
    non_indexes = (None, Ellipsis)

    def __init__(self, max_shape: Tuple[int, ...]):
        """Class for normalising some of numpy's indexing scheme.

        :param max_shape: shape of the array to be read
        """
        self.max_shape = max_shape
        self.ndim = len(max_shape)

    def _handle_newaxis_ellipses(
        self, index_tup: Tuple[SliceLike]
    ) -> Tuple[List[ConcreteSliceLike], List[int]]:
        """

        :param index_tup:
        :return: list of ints and slices the same length as the dataset's shape,
         and list of indices at which to insert newaxes (must be in inserted in order)
        """
        concrete_indices = sum(idx not in self.non_indexes for idx in index_tup)
        index_lst = []
        newaxis_at = []
        has_ellipsis = False
        int_count = 0
        for item in index_tup:
            if isinstance(item, numbers.Number):
                int_count += 1

            if item is None:
                newaxis_at.append(len(index_lst) + len(newaxis_at) - int_count)
            elif item == Ellipsis:
                if has_ellipsis:
                    raise IndexError("an index can only have a single ellipsis ('...')")
                has_ellipsis = True
                initial_len = len(index_lst)
                while len(index_lst) + (concrete_indices - initial_len) < self.ndim:
                    index_lst.append(slice(None))
            else:
                index_lst.append(item)

        if len(index_lst) > self.ndim:
            raise IndexError("too many indices for array")
        while len(index_lst) < self.ndim:
            index_lst.append(slice(None))

        return index_lst, newaxis_at

    def __getitem__(self, args: SliceArgs):
        index_tup = np.index_exp[args]
        index_lst, newaxis_at = self._handle_newaxis_ellipses(index_tup)

        start_readshape_stride = []
        out_shape = []

        for item in index_lst:
            d = len(start_readshape_stride)
            max_len = self.max_shape[d]

            if isinstance(item, slice):

                try:
                    bls = slice_to_start_len_stride(item, max_len)
                    start_readshape_stride.append(bls)
                    out_shape.append(ceil(bls.len / abs(bls.stride)))
                except NullSlicingException:
                    start_readshape_stride.append((None, None, None))
                    out_shape.append(0)

            elif isinstance(item, (int, np.integer)):

                start_readshape_stride.append(
                    int_to_start_len_stride(int(item), max_len)
                )

            else:
                raise TypeError(self.type_msg.format(type(item)))

        for newax_idx in newaxis_at:
            out_shape.insert(newax_idx, 1)

        out_shape = tuple(out_shape)
        if 0 in out_shape:
            raise NullSlicingException("Slicing has a 0-length dimension", out_shape)

        start, read_shape, stride = zip(*start_readshape_stride)
        return Roi(start, read_shape, stride, out_shape)


class IndexableArrayLike(ABC):
    @property
    @abstractmethod
    def shape(self) -> Tuple[int]:
        pass

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return np.prod(self.shape, dtype=np.intp).item()

    @property
    @abstractmethod
    def dtype(self):
        pass

    @property
    def _indexer(self):
        return Indexer(self.shape)

    def _getitem(
        self,
        args: SliceArgs,
        read_fn: Callable[[Tuple[int, ...], Tuple[int, ...]], np.ndarray],
        dtype: np.dtype = None,
    ) -> np.ndarray:
        """
        Use a given function to get data from an underlying dataset, and stride it with numpy if necessary.

        :param args: arguments as passed to __getitem__
        :param dtype: data type of the data (default whatever is returned by the internal function)
        :param read_fn: callable which takes offset and shape tuples of positive integers, and returns a numpy array
        :return: numpy array of the returned data with the requested dtype
        """
        dtype = self.dtype if dtype is None else np.dtype(dtype)
        try:
            start, read_shape, stride, out_shape = self._indexer[args]
        except NullSlicingException as e:
            return np.empty(e.out_shape, dtype=dtype)

        arr = np.asarray(read_fn(start, read_shape), dtype=dtype)

        if set(stride) != {1}:
            stride_slices = tuple(slice(None, None, s) for s in stride)
            arr = arr[stride_slices]

        if not out_shape:
            return arr.flatten()[0]
        elif arr.shape != out_shape:
            return arr.reshape(out_shape)
        else:
            return arr

    def _setitem(
        self,
        args: SliceArgs,
        array: np.ndarray,
        write_fn: Callable[[Tuple[int, ...], np.ndarray], None],
    ):
        """
        Use a given function to insert data into an underlying dataset.
        Does not support striding, or broadcasting other than scalar.

        :param args: index arguments as passed to __setitem__
        :param array: array-like data to write
        :param indexer: Indexer for the dataset
        :param dtype: data type to write
        :param write_fn: function which takes an offset as a tuple of integers, and a numpy array, and writes to an underlying dataset
        """
        try:
            start, host_shape, stride, _ = self._indexer[args]
        except NullSlicingException:
            return

        if set(stride) != {1}:
            raise NotImplementedError("Strided writes are not supported")

        if np.isscalar(array):
            item_arr = np.full(host_shape, array, dtype=self.dtype)
        else:
            try:
                item_arr = np.asarray(array, self.dtype, order="C")
            except ValueError as e:
                if any(
                    s in str(e) for s in ("invalid literal for ", "could not convert")
                ):
                    bad_dtype = np.asarray(array).dtype
                    raise TypeError("No conversion path for dtype: " + repr(bad_dtype))
                else:
                    raise
            except TypeError as e:
                if "argument must be" in str(e):
                    raise OSError(
                        "Can't prepare for writing data (no appropriate function for conversion path)"
                    )
                else:
                    raise

        if host_shape != item_arr.shape:
            raise ValueError("Shape of array does not match shape of index")

        return write_fn(start, item_arr)


def threaded_block_read(
    start: Tuple[int, ...],
    shape: Tuple[int, ...],
    stride: Tuple[int, ...],
    chunks: Tuple[int, ...],
    read_fn: Callable[[Tuple[int, ...]], np.ndarray],
    threads=DEFAULT_THREADS,
) -> np.ndarray:
    """
    For a blocked dataset, read complete blocks using python threads and then stitch and stride it in numpy.

    :param start: offset from 0 corner
    :param shape: unstrided shape of block to be read
    :param stride: strides
    :param chunks: block shape inside the dataset
    :param read_fn: function which takes a block index as a tuple of ints and returns a numpy array
    :param threads: number of threads to use
    :return: numpy array
    """
    # todo: check for off-by-one
    start = np.asarray(start, dtype=int)
    shape = np.asarray(shape, dtype=int)
    chunks = np.asarray(chunks, dtype=int)

    end = start + shape

    start_block_idx, start_internal_offset = divmod(start, chunks)

    # todo: handle ragged edges
    stop_block_idx, stop_internal_offset = divmod(end, chunks)
    if np.count_nonzero(stop_internal_offset):
        stop_block_idx += 1
        stop_internal_offset -= chunks

    block_blocks_shape: np.ndarray = stop_block_idx - start_block_idx

    blocks = np.empty(shape=block_blocks_shape, dtype=object)

    with ThreadPoolExecutor(max_workers=threads) as exe:
        futures = dict()
        for idx in itertools.product(*(range(i) for i in block_blocks_shape)):
            fut = exe.submit(read_fn, start_block_idx + idx)
            futures[fut] = idx

        for fut in as_completed(futures):
            idx = futures[fut]
            blocks[idx] = fut.result()

    big_arr = np.block(blocks.tolist())
    return big_arr[
        tuple(
            slice(*sss)
            for sss in zip(start_internal_offset, stop_internal_offset, stride)
        )
    ]
