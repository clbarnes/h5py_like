import numpy as np
import pytest

from h5py_like.shape_utils import (
    Roi,
    NullSlicingException,
    slice_to_start_len_stride,
    int_to_start_len_stride,
    Indexer,
)

LEN = 10


@pytest.mark.parametrize(
    ["sl", "expected"],
    [
        (slice(None), (0, LEN, 1)),
        (slice(1, 5, 2), (1, 4, 2)),
        (slice(-5, -1), (5, 4, 1)),
        (slice(5, 1, -2), (1, 4, -2)),
        (slice(None, None, -5), (0, LEN, -5)),
    ],
)
def test_slice_to_start_len_stride(sl, expected):
    test = slice_to_start_len_stride(sl, LEN)
    assert test == expected


@pytest.mark.parametrize(
    "sl", [slice(0, 0), slice(1, 5, -1), slice(5, 1), slice(5, -5)]
)
def test_slice_to_start_len_stride_error(sl):
    with pytest.raises(NullSlicingException):
        slice_to_start_len_stride(sl, LEN)


@pytest.mark.parametrize(["idx", "expected"], [(1, (1, 1, 1)), (-1, (9, 1, 1))])
def test_int_to_start_len_stride(idx, expected):
    test = int_to_start_len_stride(idx, LEN)
    assert test == expected


@pytest.mark.parametrize("idx", [LEN + 1, -LEN - 1])
def test_int_to_start_len_stride_fails(idx):
    with pytest.raises(IndexError):
        int_to_start_len_stride(idx, LEN)


SHAPE = (10, 20, 30)


def test_indexer_simple():
    indexer = Indexer(SHAPE)
    test = indexer[1:2, 1:2, 1:2]
    assert test == Roi(
        start=(1, 1, 1), read_shape=(1, 1, 1), stride=(1, 1, 1), out_shape=(1, 1, 1)
    )


def test_indexer_squeeze_1():
    indexer = Indexer(SHAPE)
    test = indexer[1:2, 1:2, 1]
    assert test == Roi(
        start=(1, 1, 1), read_shape=(1, 1, 1), stride=(1, 1, 1), out_shape=(1, 1)
    )


def test_indexer_squeeze_all():
    indexer = Indexer(SHAPE)
    test = indexer[1, 1, 1]
    assert test == Roi(
        start=(1, 1, 1), read_shape=(1, 1, 1), stride=(1, 1, 1), out_shape=()
    )


def test_indexer_ellipsis():
    indexer = Indexer(SHAPE)
    test = indexer[...]
    assert test == Roi(
        start=(0, 0, 0), read_shape=SHAPE, stride=(1, 1, 1), out_shape=SHAPE
    )


def test_implied_ellipsis():
    indexer = Indexer(SHAPE)
    test = indexer[1:2]
    assert test == Roi(
        start=(1, 0, 0), read_shape=(1, 20, 30), stride=(1, 1, 1), out_shape=(1, 20, 30)
    )


def test_indexer_ellipsis_and_int():
    indexer = Indexer(SHAPE)
    test0 = indexer[..., 1, 1]
    assert test0 == Roi(
        start=(0, 1, 1), read_shape=(10, 1, 1), stride=(1, 1, 1), out_shape=(10,)
    )

    test1 = indexer[1, ..., 1]
    assert test1 == Roi(
        start=(1, 0, 1), read_shape=(1, 20, 1), stride=(1, 1, 1), out_shape=(20,)
    )

    test2 = indexer[1, 1, ...]
    assert test2 == Roi(
        start=(1, 1, 0), read_shape=(1, 1, 30), stride=(1, 1, 1), out_shape=(30,)
    )


def test_indexer_newaxis():
    indexer = Indexer(SHAPE)
    test = indexer[1:2, 1:2, np.newaxis, 1:2]
    assert test == Roi(
        start=(1, 1, 1), read_shape=(1, 1, 1), stride=(1, 1, 1), out_shape=(1, 1, 1, 1)
    )


def test_indexer_stride():
    indexer = Indexer(SHAPE)
    test = indexer[::5, ::-5, :]
    assert test == Roi(
        start=(0, 0, 0), read_shape=SHAPE, stride=(5, -5, 1), out_shape=(2, 4, 30)
    )
