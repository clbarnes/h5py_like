from abc import ABC, abstractmethod
from functools import wraps
from typing import Tuple, List, Iterator, Optional

from .common import ReadOnlyException, Mode, Name


class WriteModeMixin:
    _mode = Mode.READ_ONLY

    @property
    def mode(self):
        return self._mode

    def raise_on_readonly(self):
        if not self.mode.writable:
            raise ReadOnlyException(f"Cannot write to readonly {type(self)}")


def mutation(fn):
    @wraps(fn)
    def wrapped(obj: WriteModeMixin, *args, **kwargs):
        obj.raise_on_readonly()
        return fn(obj, *args, **kwargs)

    return wrapped


class H5ObjectLike(WriteModeMixin, ABC):
    _is_file = False

    def __init__(self, mode: Mode = Mode.default()):
        self._mode = Mode.from_str(mode)

    @property
    @abstractmethod
    def attrs(self) -> "AttributeManagerBase":  # noqa
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def file(self) -> "FileMixin":  # noqa
        p = self
        for p in self._ancestors():
            pass
        return p

    @property
    @abstractmethod
    def parent(self):
        pass

    @property
    def mode(self):
        return self.parent.mode

    def _ancestors(self) -> Iterator["H5ObjectLike"]:
        """Iterate through ancestors (including this object) until the root"""
        ancestor = self
        while not ancestor._is_file:
            yield ancestor
            ancestor = ancestor.parent
        yield ancestor

    def _absolute_name(self, other: str) -> str:
        """Get the absolute name of the object at the given absolute or relative name"""
        name = Name(self.name)
        return str(name.joinpath(other))

    def _relative_name(self, other: str) -> Optional[str]:
        """Get the relative name of the other object from this one.

        Raises ValueError if they cannot be reached; return None if it is this object.
        """
        name = Name(self.name)
        other = self._absolute_name(other)
        out = str(Name(other).relative_to(name))
        return None if out == "." else out

    def _ancestor_and_relative_name(
        self, other: str
    ) -> Tuple["H5ObjectLike", Optional[str]]:
        """Get the most recent common ancestor with the other name,
        and the relative path from that ancestor to the other name"""
        for ancestor in self._ancestors():
            try:
                return ancestor, ancestor._relative_name(other)
            except ValueError as e:
                if "does not start with" not in str(e):
                    raise

        raise RuntimeError(f"Name '{other}' is not relative to root file")

    def _descend(self, other: str) -> Tuple["H5ObjectLike", List[str], Optional[str]]:
        """Get most recent common ancestor and intermediate names to other.

        :param other:
        :return: most recent common ancestor group, list of intermediate names,
            name of other (or None)
        """
        ancestor, relative_name = self._ancestor_and_relative_name(other)
        if not relative_name:
            return ancestor, [], relative_name

        *mids, final = Name(relative_name).parts
        return ancestor, list(mids), final
