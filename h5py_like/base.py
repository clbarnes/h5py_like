from abc import ABC, abstractmethod
from functools import wraps

from .common import ReadOnlyException, Mode


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
        return fn(*args, **kwargs)
    return wrapped


class H5ObjectLike(WriteModeMixin, ABC):
    def __init__(self, mode: Mode = Mode.default()):
        self._mode = Mode.from_str(mode)

    @property
    @abstractmethod
    def attrs(self) -> "AttributeManagerBase":
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def file(self) -> "FileMixin":
        current = self
        while True:
            parent = current.parent
            if parent is current:
                return current

    @property
    @abstractmethod
    def parent(self):
        pass

    @property
    def mode(self):
        return self.parent.mode