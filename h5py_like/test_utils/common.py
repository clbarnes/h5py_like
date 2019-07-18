import logging
import numpy as np

from h5py_like.common import classname

ds_kwargs = {"name": "dataset", "shape": (10, 10, 10), "dtype": np.dtype("uint16")}


def check_attrs_rw(attrs):
    assert "key" not in attrs
    attrs["key"] = "value"
    assert "key" in attrs
    assert attrs["key"] == "value"
    del attrs["key"]
    assert "key" not in attrs


class LoggedClassMixin:
    @property
    def logger(self):
        return logging.getLogger(classname(self))
