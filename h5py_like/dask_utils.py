from .group import GroupBase
from .dataset import DatasetBase


def read_as_dask(obj, path=None, **kwargs):
    """Read h5py-like dataset into a dask array with ``dask.array.from_array``.

    :param obj: Object to read.
        If a Dataset, read this dataset and ignore ``path``.
        If a Group (or File), get the dataset at ``path``, relative to the group.
    :param path: If ``obj`` is a Group, get the dataset with this (relative) name
    :param kwargs: passed to dask.array.from_array
    :return: dask.array.Array
    """
    if isinstance(obj, GroupBase):
        obj = obj[path]
    if not isinstance(obj, DatasetBase):
        raise ValueError("Object is not a dataset: {}".format(obj))

    import dask.array as da

    kw = {"fancy": False, "name": False}
    kw.update(kwargs)

    return da.from_array(obj, **kw)


def write_dask(array, obj, path=None, ds_kwargs=None, **kwargs):
    """Write dask array to h5py-like dataset with ``dask.array.store``.

    :param array: dask.array.Array to write
    :param obj: Object to write to.
        If a Dataset, write to this dataset and ignore ``path``.
        If a Group, require the dataset at ``path``, relative to the group.
        ``require``s a dataset with compatible metadata
        (see ``GroupBase.require_dataset``).
    :param path: If ``obj`` is a group, get the dataset with this (relative) name
    :param ds_kwargs: passed to ``GroupBase.require_dataset`` if ``obj`` is a Group
    :param kwargs: passed to ``dask.array.store``
    :return: ``dask.array.store`` return value.
    """
    if isinstance(obj, GroupBase):
        defaults = {"shape": array.shape, "dtype": array.dtype, "chunks": array.chunks}
        defaults.update(ds_kwargs or dict())
        obj = obj.require_dataset(path, **defaults)
    if not isinstance(obj, DatasetBase):
        raise ValueError("Object is not a dataset: {}".format(obj))

    import dask.array as da

    return da.store(array, obj, **kwargs)
