import numpy as np
import pytest

from h5py_like.shape_utils import (
    Roi,
    NullSlicingException,
    slice_to_start_len_stride,
    int_to_start_len_stride,
    Indexer,
    guess_chunks,
    CHUNK_MAX,
    chunk_roi,
    thread_read_fn,
    to_slice,
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


@pytest.mark.parametrize(
    ["start", "shape", "expected"],
    [[(10, 10), (15, 15), (slice(10, 25), slice(10, 25))]],
)
def test_to_slice(start, shape, expected):
    assert to_slice(start, shape) == expected


@pytest.mark.parametrize(
    "shape",
    [
        (100,),
        (100, 100),
        (1000000,),
        (1000000000,),
        (10000000000000000000000,),
        (10000, 10000),
        (10000000, 1000),
        (1000, 10000000),
        (10000000, 1000, 2),
        (1000, 10000000, 2),
        (10000, 10000, 10000),
        (100000, 100000, 100000),
        (1000000000, 1000000000, 1000000000),
        (0,),
        (0, 0),
        (10, 0),
        (0, 10),
        (1, 2, 0, 4, 5),
    ],
    ids=str,
)
@pytest.mark.parametrize("typesize", [1, 2, 4, 8])
def test_guess_chunks(shape, typesize):
    chunks = guess_chunks(shape, typesize)
    assert isinstance(chunks, tuple)
    assert len(chunks) == len(shape)
    assert all([0 < c <= max(s, 1) for c, s in zip(chunks, shape)])
    bytes_per_chunk = typesize * np.product(chunks)
    assert bytes_per_chunk < CHUNK_MAX


def idfn(val):
    if isinstance(val, tuple):
        return str(val)
    elif isinstance(val, set):
        return ""


@pytest.mark.parametrize(
    ["start", "shape", "expected"],
    [
        [(0, 0), (10, 10), {((0, 0), (10, 10))}],  # simple
        [(0, 0), (8, 8), {((0, 0), (8, 8))}],  # high offset
        [(2, 2), (8, 8), {((2, 2), (8, 8))}],  # low offset
        [
            (0, 0),
            (20, 20),
            {
                ((0, 0), (10, 10)),
                ((10, 0), (10, 10)),
                ((0, 10), (10, 10)),
                ((10, 10), (10, 10)),
            },
        ],  # multi-chunk
        [
            (5, 5),
            (20, 20),
            {
                ((5, 5), (5, 5)),
                ((5, 10), (5, 10)),
                ((5, 20), (5, 5)),
                ((10, 5), (10, 5)),
                ((10, 10), (10, 10)),
                ((10, 20), (10, 5)),
                ((20, 5), (5, 5)),
                ((20, 10), (5, 10)),
                ((20, 20), (5, 5)),
            },
        ],  # complicated multi-chunk
    ],
    ids=idfn,
)
def test_chunk_roi(start, shape, expected):
    chunks = (10, 10)

    chunk_coords_set = set(chunk_roi(start, shape, chunks, (30, 30)))

    assert chunk_coords_set == expected


@pytest.mark.parametrize(
    ["start", "shape"],
    [
        [(0, 0), (10, 10)],  # simple
        [(0, 0), (8, 8)],  # high offset
        [(2, 2), (8, 8)],  # low offset
        [(0, 0), (20, 20)],  # multi-chunk
        [(5, 5), (20, 20)],  # complicated multi-chunk
    ],
    ids=str,
)
def test_thread_read_fn(start, shape):
    random = np.random.RandomState(1991)
    data = random.random_sample((30, 30))

    chunks = (10, 10)

    def fn(start_coord, block_shape):
        stop_coord = np.array(start_coord) + block_shape
        slicing = tuple(slice(sta, sto) for sta, sto in zip(start_coord, stop_coord))
        return data[slicing]

    out = thread_read_fn(start, shape, chunks, data.shape, fn, threads=1)
    slicing = to_slice(start, shape)
    expected = data[slicing]
    assert np.allclose(expected, out)
