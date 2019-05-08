from abc import ABC, abstractmethod
from collections import MutableMapping

from enum_custom import StrEnum

from .attributes import AttributeManager
from .file import File


DEFAULT_THREADS = 4


class ReadOnlyException(RuntimeError):
    pass


class H5ObjectLike(MutableMapping, ABC):
    @property
    @abstractmethod
    def attrs(self) -> AttributeManager:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def file(self) -> File:
        current = self
        while True:
            parent = current.parent
            if parent is None:
                return current

    @property
    @abstractmethod
    def parent(self):
        pass


class Mode(StrEnum):
    READ_ONLY = "r"
    READ_WRITE = "r+"
    CREATE_TRUNCATE = "w"
    CREATE = "x"
    READ_WRITE_CREATE = "a"

    @property
    def writable(self):
        return self != type(self).READ_ONLY

    @classmethod
    def from_str(cls, s):
        if s == "w-":
            s = "x"
        return cls(s)
