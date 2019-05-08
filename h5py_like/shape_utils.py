import numbers
from collections import deque
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor

from typing import Iterable, Callable, Tuple, Any, Iterator, TypeVar, Union, Optional
from itertools import islice, product

import numpy as np

from .common import DEFAULT_THREADS


class NullSlicingException(ValueError):
    pass


def slice_to_begin_len_stride(slice_: slice, max_len: int) -> Tuple[int, int, int]:
    """
    Convert a slice object, possibly with None or negative members, into positive integers for start, length, and stride.

    Raises NullSlicingException if there would be no results returned from this indexing.

    :param slice_:
    :param max_len: maximum length of the dimension
    :return: tuple of positive integer start, length, stride
    :raises: NullSlicingException
    """
    """For a single dimension with a given size, turn a slice object into a (start_idx, length)
     pair. Returns (None, 0) if slice is invalid."""
    stride = 1 if slice_.step is None else slice_.step

    if slice_.start is None:
        start = 0
    elif -max_len <= slice_.start < 0:
        start = max_len + slice_.start
    elif slice_.start < -max_len or slice_.start >= max_len:
        raise NullSlicingException("Requested slice is empty")
    else:
        start = slice_.start

    if slice_.stop is None or slice_.stop > max_len:
        stop = max_len
    elif slice_.stop < 0:
        stop = max_len + slice_.stop
    else:
        stop = slice_.stop

    if stride < 0:
        stop, start = start, stop

    if stop <= start:
        raise NullSlicingException("Requested slice is empty")

    return start, stop - start, stride


def int_to_begin_len_stride(i: int, max_len: int) -> Tuple[int, int, int]:
    """
    Convert an integer index, possibly negative, into positive integers for start, length, and stride.

    Raises NullSlicingException if there would be no results returned from this indexing

    :param i: integer index
    :param max_len: maximum length of the dimension
    :return: tuple of positive integer start, length, stride
    :raises: NullSlicingException
    """
    if -max_len < i < 0:
        begin = i + max_len
    elif i >= max_len or i < -max_len:
        raise IndexError("Index ({}) out of range (0-{})".format(i, max_len - 1))
    else:
        begin = i

    return begin, 1, 1


def sliding_window(iterable: Iterable, wsize: int) -> Iterator[Tuple[Any, ...]]:
    """
    Move a sliding window over the iterable.
    If the window is longer than the iterable, yields 0 results.

    :param iterable:
    :param wsize: length of the sliding window
    :return: iterator of wsize-length tuples
    """
    """Yield a wsize-length tuple of items from iterable arr as a sliding window"""
    q = deque(islice(iterable, wsize), wsize)
    if len(q) != wsize:
        return
    yield tuple(q)
    for item in iterable:
        q.append(item)
        yield tuple(q)


def rectify_shape(arr: np.ndarray, required_shape: Tuple[int, ...]) -> np.ndarray:
    """Reshape arr into the required shape while keeping neighbouring non-singleton dimensions together
    e.g. shape (1, 2, 1) -> (2, 1, 1, 1) is fine
    shape (1, 2, 1, 2, 1, 1, 1) -> (1, 2, 2, 1) is not
    """
    if arr.shape == required_shape:
        return arr

    important_shape = list(arr.shape)
    while len(important_shape) > 1:
        if important_shape[0] == 1:
            important_shape.pop(0)
        elif important_shape[-1] == 1:
            important_shape.pop()
        else:
            break

    important_shape = tuple(important_shape)

    msg = (
        "could not broadcast input array from shape {} into shape {}; "
        "complicated broadcasting not supported"
    ).format(arr.shape, required_shape)

    if len(important_shape) > len(required_shape):
        raise ValueError(msg)

    is_leading = True
    for window in sliding_window(required_shape, len(important_shape)):
        if window == important_shape:
            if is_leading:
                is_leading = False
                continue
            else:
                break
        if is_leading:
            if window[0] != 1:
                break
        else:
            if window[-1] != 1:
                break
    else:
        return arr.reshape(required_shape)

    raise ValueError(msg)


SliceLike = Union[ellipsis, slice, int]
SliceArgs = TypeVar("SliceArgs", SliceLike, Tuple[SliceLike])


def sanitize_indices(
    args: SliceArgs, max_shape: Tuple[int, ...]
) -> Tuple[
    Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Optional[Tuple[int, ...]]
]:
    """
    Given arguments as usually passed to __getitem__,
    convert them into tuples of positive integers describing the
    offset, shape (total, as if unstrided), and striding of the query.
    Also returns a tuple of dimensions to squeeze (i.e. the slice argument was an integer).

    :param args: arguments as passed to __getitem__
    :param max_shape: maximum shape of the array-like dataset
    :return: start, shape, and strides over each dimension, and a tuple of which dimensions should be squeezed (None if none of them)
    """
    type_msg = (
        "Advanced selection inappropriate. "
        "Only numbers, slices (`:`), and ellipsis (`...`) are valid indices (or tuples thereof); got {}"
    )
    ndim = len(max_shape)

    if isinstance(args, tuple):
        index_lst = list(args)
    elif isinstance(args, (numbers.Number, slice, type(Ellipsis))):
        index_lst = [args]
    else:
        raise TypeError(type_msg.format(type(args)))

    if len([item for item in index_lst if item != Ellipsis]) > ndim:
        raise TypeError("Argument sequence too long")
    elif len(index_lst) < ndim and Ellipsis not in index_lst:
        index_lst.append(Ellipsis)

    begin_len_stride = []
    found_ellipsis = False
    squeeze = []
    dim = 0
    for item in index_lst:
        d = len(begin_len_stride)
        if isinstance(item, slice):
            begin_len_stride.append(slice_to_begin_len_stride(item, max_shape[d]))
        elif isinstance(item, numbers.Number):
            begin_len_stride.append(int_to_begin_len_stride(int(item), max_shape[d]))
            squeeze.append(dim)
        elif isinstance(item, type(Ellipsis)):
            if found_ellipsis:
                raise ValueError("Only one ellipsis may be used")
            found_ellipsis = True
            while len(begin_len_stride) + (len(index_lst) - d - 1) < ndim:
                begin_len_stride.append((0, max_shape[len(begin_len_stride)], 1))
        else:
            raise TypeError(type_msg.format(type(item)))
        dim = len(begin_len_stride)

    roi_begin, roi_shape, stride = zip(*begin_len_stride)
    squeeze = tuple(squeeze) if squeeze else None
    return roi_begin, roi_shape, stride, squeeze


def getitem(
    args: SliceArgs,
    max_shape: Tuple[int, ...],
    dtype: np.dtype,
    read_fn: Callable[[Tuple[int, ...], Tuple[int, ...]], np.ndarray],
) -> np.ndarray:
    """
    Use a given function to get data from an underlying dataset, and stride it with numpy if necessary.

    :param args: arguments as passed to __getitem__
    :param max_shape: maximum shape of the array-like dataset
    :param dtype: data type of the data (default whatever is returned by the internal function)
    :param read_fn: callable which takes offset and shape tuples of positive integers, and returns a numpy array
    :return: numpy array of the returned data with the requested dtype
    """
    try:
        begin, shape, stride, squeeze = sanitize_indices(args, max_shape)
    except NullSlicingException:
        # todo: could be mixture of 0 and non-zero lengths
        return np.empty((0,) * len(max_shape), dtype=dtype)

    arr = np.asarray(read_fn(begin, shape), dtype=dtype)
    if (
        isinstance(args, tuple)
        and len(args) == len(max_shape)
        and all(isinstance(i, int) for i in args)
    ):
        return arr.item()

    if set(stride) == {1}:
        return arr

    stride_slices = tuple(slice(None, None, s) for s in stride)
    return arr[stride_slices].squeeze(squeeze)


def setitem(
    args: SliceArgs,
    array: np.ndarray,
    max_shape: Tuple[int, ...],
    dtype,
    write_fn: Callable[[Tuple[int, ...], np.ndarray], np.ndarray],
):
    """
    Use a given function to insert data into an underlying dataset.
    Does not support striding or broadcasting.

    :param args: index arguments as passed to __setitem__
    :param array: array-like data to write
    :param max_shape: shape of the target dataset
    :param dtype: data type to write
    :param write_fn: function which takes an offset as a tuple of integers, and a numpy array, and writes to an underlying dataset
    """
    begin, shape, stride, _ = sanitize_indices(args, max_shape)
    if set(stride) != {1}:
        raise NotImplementedError("Strided writes are not supported")

    if 0 in shape:
        return
    if shape != array.shape:
        raise ValueError("Extents of requested write go beyond array bounds")

    try:
        item_arr = np.asarray(array, dtype, order="C")
    except ValueError as e:
        if any(s in str(e) for s in ("invalid literal for ", "could not convert")):
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

    # item_arr = rectify_shape(item_arr, shape)

    return write_fn(begin, item_arr)


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
        for idx in product(*(range(i) for i in block_blocks_shape)):
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
