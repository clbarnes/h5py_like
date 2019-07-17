from abc import ABC, abstractmethod
from collections.abc import MutableMapping

import numpy as np
from typing import Any, Optional, Callable

from .base import H5ObjectLike, mutation
from .dataset import DatasetBase
from .common import classname


class GroupBase(H5ObjectLike, MutableMapping, ABC):
    """ Represents an HDF5-like group.
    """

    _is_file = False

    @abstractmethod
    def _create_child_group(self, name) -> "GroupBase":
        """Create a group which is a direct child of this object with the given name.

        Should raise a TypeError if a dataset exists there,
        or a FileExistsError if a group exists there.
        """
        pass

    def _require_descendant_groups(self, *names):
        this = self
        for name in names:
            try:
                this = this._get_child(name)
                if not isinstance(this, GroupBase):
                    raise TypeError("Not a group")
            except KeyError:
                this = this._create_child_group(name)
        return this

    @mutation
    def create_group(self, name) -> "GroupBase":
        """ Create and return a new subgroup.
        Name may be absolute or relative.  Fails if the target name already
        exists.
        """
        ancestor, group_names, last_name = self._descend(name)
        parent = ancestor._require_descendant_groups(*group_names)
        if last_name in parent:
            raise FileExistsError(f"Group or dataset found at '{name}'")
        return parent._create_child_group(last_name)

    @abstractmethod
    def _create_child_dataset(self, name, shape=None, dtype=None, data=None, **kwds):
        """Create a dataset which is a direct child of this object with the given name.

        Should raise a TypeError if a group exists there,
        or a FileExistsError if a dataset exists there.
        """
        pass

    @mutation
    def create_dataset(
        self, name, shape=None, dtype=None, data=None, **kwds
    ) -> DatasetBase:
        """ Create a new HDF5-like dataset
        name
            Name of the dataset (absolute or relative).
        shape
            Dataset shape.  Use "()" for scalar datasets.  Required if "data"
            isn't provided.
        dtype
            Numpy dtype or string.  If omitted, dtype('f') will be used.
            Required if "data" isn't provided; otherwise, overrides data
            array's dtype.
        data
            Provide data to initialize the dataset.  If used, you can omit
            shape and dtype arguments.
        Keyword-only arguments:
        """
        ancestor, group_names, last_name = self._descend(name)
        parent = ancestor._require_descendant_groups(*group_names)
        if last_name in parent:
            raise FileExistsError(f"Group or dataset found at '{name}'")
        return parent._create_child_dataset(name, shape, dtype, data, **kwds)

    def require_dataset(self, name, shape, dtype, exact=False, **kwds) -> DatasetBase:
        """ Open a dataset, creating it if it doesn't exist.
        If keyword "exact" is False (default), an existing dataset must have
        the same shape and a conversion-compatible dtype to be returned.  If
        True, the shape and dtype must match exactly.
        Other dataset keywords (see create_dataset) may be provided, but are
        only used if a new dataset is to be created.
        Raises TypeError if an incompatible object already exists, or if the
        shape or dtype don't match according to the above rules.
        """
        if name not in self:
            return self.create_dataset(name, *(shape, dtype), **kwds)

        dset = self[name]
        if not isinstance(dset, DatasetBase):
            raise TypeError(
                "Incompatible object (%s) already exists" % dset.__class__.__name__
            )

        if not shape == dset.shape:
            raise TypeError(
                "Shapes do not match (existing %s vs new %s)" % (dset.shape, shape)
            )

        if exact:
            if not dtype == dset.dtype:
                raise TypeError(
                    "Datatypes do not exactly match (existing %s vs new %s)"
                    % (dset.dtype, dtype)
                )
        elif not np.can_cast(dtype, dset.dtype):
            raise TypeError(
                "Datatypes cannot be safely cast (existing %s vs new %s)"
                % (dset.dtype, dtype)
            )

        return dset

    def create_dataset_like(self, name, other, **kwupdate) -> DatasetBase:
        """ Create a dataset similar to `other`.
        name
            Name of the dataset (absolute or relative).  Provide None to make
            an anonymous dataset.
        other
            The dataset which the new dataset should mimic. All properties, such
            as shape, dtype, chunking, ... will be taken from it, but no data
            or attributes are being copied.
        Any dataset keywords (see create_dataset) may be provided, including
        shape and dtype, in which case the provided values take precedence over
        those from `other`.
        """
        for k in ("shape", "dtype", "chunks", "fillvalue"):
            kwupdate.setdefault(k, getattr(other, k))

        # Special case: the maxshape property always exists, but if we pass it
        # to create_dataset, the new dataset will automatically get chunked
        # layout. So we copy it only if it is different from shape.
        if other.maxshape != other.shape:
            kwupdate.setdefault("maxshape", other.maxshape)

        return self.create_dataset(name, **kwupdate)

    def require_group(self, name) -> "GroupBase":
        """Return a group, creating it if it doesn't exist.
        TypeError is raised if something with that name already exists that
        isn't a group.
        """
        if name not in self:
            return self.create_group(name)

        group = self[name]
        if not isinstance(group, GroupBase):
            raise TypeError(f"Incompatible object ({type(group)}) already exists")

        return group

    @abstractmethod
    def _get_child(self, name) -> H5ObjectLike:
        """Get an object (a concrete Dataset or Group) which is a direct child of this object.

        Raise a KeyError if it's not found.
        """
        pass

    def __getitem__(self, name) -> H5ObjectLike:
        """ Open an object in the file """
        ancestor, group_names, last_name = self._descend(name)
        for group_name in group_names:
            ancestor = ancestor._get_child(group_name)
            if not isinstance(ancestor, GroupBase):
                raise TypeError(f"Expected Group, got {type(ancestor)}")
        return ancestor._get_child(last_name)

    @abstractmethod
    def __setitem__(self, name, obj):
        """ Add an object to the group.  The name must not already be in use.
        The action taken depends on the type of object assigned:
        Named HDF5 object (Dataset, Group, Datatype)
            A hard link is created at "name" which points to the
            given object.
        SoftLink or ExternalLink
            Create the corresponding link.
        Numpy ndarray
            The array is converted to a dataset object, with default
            settings (contiguous storage, etc.).
        Numpy dtype
            Commit a copy of the datatype as a named datatype in the file.
        Anything else
            Attempt to convert it to an ndarray and store it.  Scalar
            values are stored as scalar datasets. Raise ValueError if we
            can't understand the resulting array dtype.
        """

    @abstractmethod
    def copy(
        self,
        source,
        dest,
        name=None,
        shallow=False,
        expand_soft=False,
        expand_external=False,
        expand_refs=False,
        without_attrs=False,
    ):
        """Copy an object or group.
        The source can be a path, Group, Dataset, or Datatype object.  The
        destination can be either a path or a Group object.  The source and
        destinations need not be in the same file.
        If the source is a Group object, all objects contained in that group
        will be copied recursively.
        When the destination is a Group object, by default the target will
        be created in that group with its current name (basename of obj.name).
        You can override that by setting "name" to a string.
        There are various options which all default to "False":
         - shallow: copy only immediate members of a group.
         - expand_soft: expand soft links into new objects.
         - expand_external: expand external links into new objects.
         - expand_refs: copy objects that are pointed to by references.
         - without_attrs: copy object without copying attributes.
       Example:
        >>> f = File('myfile.hdf5')
        >>> f.listnames()
        ['MyGroup']
        >>> f.copy('MyGroup', 'MyCopy')
        >>> f.listnames()
        ['MyGroup', 'MyCopy']
        """

    @mutation
    def move(self, source, dest):
        """ Move a link to a new location in the file.
        If "source" is a hard link, this effectively renames the object.  If
        "source" is a soft or external link, the link itself is moved, with its
        value unmodified.
        """
        self.copy(source, dest)
        del self[source]

    def _recurse(self):
        """Depth-first search"""
        for k, v in self.items():
            yield k, v
            try:
                yield from v._recurse()
            except AttributeError:
                pass

    def visit(self, func: Callable[[str], Optional[Any]]) -> Optional[Any]:
        """ Recursively visit all names in this group and subgroups (HDF5 1.8).
        You supply a callable (function, method or callable object); it
        will be called exactly once for each link in this group and every
        group below it. Your callable must conform to the signature:
            func(<member name>) => <None or return value>
        Returning None continues iteration, returning anything else stops
        and immediately returns that value from the visit method.  No
        particular order of iteration within groups is guaranteed.
        Example:
        >>> # List the entire contents of the file
        >>> f = File("foo.hdf5")
        >>> list_of_names = []
        >>> f.visit(list_of_names.append)
        """
        for key, _ in self._recurse():
            result = func(key)
            if result is not None:
                return result

    def visititems(
        self, func: Callable[[str, H5ObjectLike], Optional[Any]]
    ) -> Optional[Any]:
        """ Recursively visit names and objects in this group (HDF5 1.8).
        You supply a callable (function, method or callable object); it
        will be called exactly once for each link in this group and every
        group below it. Your callable must conform to the signature:
            func(<member name>, <object>) => <None or return value>
        Returning None continues iteration, returning anything else stops
        and immediately returns that value from the visit method.  No
        particular order of iteration within groups is guaranteed.
        Example:
        # Get a list of all datasets in the file
        >>> mylist = []
        >>> def func(name, obj):
        ...     if isinstance(obj, DatasetBase):
        ...         mylist.append(name)
        ...
        >>> f = File('foo.hdf5')
        >>> f.visititems(func)
        """
        for key, val in self._recurse():
            result = func(key, val)
            if result is not None:
                return result

    def __str__(self):
        return f"<{classname(self)}(name='{self.name}', file={self.file})>"

    def __eq__(self, other):
        try:
            return all(
                (
                    isinstance(other, GroupBase),
                    not other._is_file,
                    self.name == other.name,
                    self.parent == other.parent,
                )
            )
        except AttributeError:
            return False
