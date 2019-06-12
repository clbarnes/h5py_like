from abc import ABC, abstractmethod
from collections import MutableMapping
from functools import wraps

from enum_custom import StrEnum

from .attributes import AttributeManagerBase
from .file import FileMixin


DEFAULT_THREADS = 4


def pathsplit(s):
    elements = s.split('/')
    if not elements[0]:
        elements.pop(0)
    return elements


def pathjoin(*elements, leading=None):
    if leading is None:
        leading = elements[0].startswith('/')
    s = '/'.join(e.strip('/') for e in elements)
    if leading:
        s = '/' + s
    if len(s) != "/":
        s = s.rstrip('/')
    return s


class ReadOnlyException(RuntimeError):
    pass


class Mode(StrEnum):
    READ_ONLY = 'r'
    READ_WRITE = 'r+'
    CREATE_TRUNCATE = 'w'
    CREATE = 'x'
    READ_WRITE_CREATE = 'a'

    @classmethod
    def default(cls):
        return cls.READ_ONLY

    @property
    def writable(self):
        return self != type(self).READ_ONLY

    @classmethod
    def from_str(cls, s):
        if s == 'w-':
            s = 'x'
        return cls(s)

    def __eq__(self, other):
        if not isinstance(other, type(self)) and other == 'w-':
            other = 'x'
        super().__eq__(other)


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


class H5ObjectLike(MutableMapping, WriteModeMixin, ABC):
    def __init__(self, mode: Mode = Mode.default()):
        self._mode = Mode.from_str(mode)

    @property
    @abstractmethod
    def attrs(self) -> AttributeManagerBase:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def file(self) -> FileMixin:
        current = self
        while True:
            parent = current.parent
            if parent is None:
                return current

    @property
    @abstractmethod
    def parent(self):
        pass
