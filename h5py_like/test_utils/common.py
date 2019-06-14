import numpy as np

ds_kwargs = {"name": "dataset", "shape": (10, 10, 10), "dtype": np.dtype("uint16")}


def check_attrs_rw(attrs):
    assert "key" not in attrs
    attrs["key"] = "value"
    assert "key" in attrs
    assert attrs["key"] == "value"
    del attrs["key"]
    assert "key" not in attrs
