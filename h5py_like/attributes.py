from abc import ABC
from collections.abc import MutableMapping

import numpy as np

from .common import Mode
from h5py_like.base import WriteModeMixin, mutation


class AttributeManagerBase(MutableMapping, WriteModeMixin, ABC):
    """
    Allows dictionary-style access to an HDF5-like object's attributes.
    These are created exclusively by the library and are available as
    a Python attribute at <object>.attrs
    Like Group objects, attributes provide a minimal dictionary-
    style interface.  Anything which can be reasonably converted to a
    Numpy array or Numpy scalar can be stored.
    Attributes are automatically created on assignment with the
    syntax <obj>.attrs[name] = value, with the type automatically
    deduced from the value.  Existing attributes are overwritten.
    To modify an existing attribute while preserving its (numpy) shape and type, use the
    method modify().  To specify an attribute of a particular (numpy) type and
    shape, use create().
    """

    def __init__(self, mode: Mode = Mode.default()):
        self._mode = Mode.from_str(mode)

    @mutation
    def create(self, name, data, shape=None, dtype=None):
        """ Set a new attribute, overwriting any existing attribute.
        The type and shape of the attribute are determined from the data.  To
        use a specific (numpy) type or shape, or to preserve the type of an attribute,
        use the methods create() and modify().
        """
        if dtype is None:
            dtype = getattr(data, "dtype", None)
        else:
            dtype = np.dtype(dtype)

        arr = np.asarray(data, dtype=dtype)

        if shape is not None and shape != arr.shape:
            arr = arr.reshape(shape)

        self[name] = arr

    @mutation
    def modify(self, name, value):
        """ Change the value of an attribute while preserving its (numpy) type.
        Differs from __setitem__ in that if the attribute already exists, its
        type is preserved.  This can be very useful for interacting with
        externally generated files.
        If the attribute doesn't exist, it will be automatically created.
        """
        existing = self.get(name)
        if not existing:
            self.create(name, value)
        else:
            self.create(
                name,
                value,
                getattr(existing, "shape", None),
                getattr(existing, "dtype", None),
            )
