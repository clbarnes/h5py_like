from enum import Enum
from pathlib import PurePosixPath


class ReadOnlyException(RuntimeError):
    pass


class StrEnum(str, Enum):
    """Enum subclass which members are also instances of str
    and directly comparable to strings. str type is forced at declaration.
    """

    def __new__(cls, *args):
        for arg in args:
            if not isinstance(arg, str):
                raise TypeError("Not text %s:" % arg)
        return super(StrEnum, cls).__new__(cls, *args)

    def __str__(self):
        return self.value

    def __hash__(self):
        return hash(str(self))


class Mode(StrEnum):
    READ_ONLY = "r"
    READ_WRITE = "r+"
    CREATE_TRUNCATE = "w"
    CREATE = "x"
    READ_WRITE_CREATE = "a"

    @classmethod
    def default(cls):
        return cls.READ_ONLY

    @property
    def writable(self):
        return self != type(self).READ_ONLY

    @classmethod
    def from_str(cls, s):
        if s == "w-":
            s = "x"
        return cls(s)

    def __eq__(self, other):
        if not isinstance(other, type(self)) and other == "w-":
            other = "x"
        return super().__eq__(other)


def classname(obj):
    cls = type(obj)
    module = cls.__module__
    name = cls.__qualname__
    if module is not None and module != "__builtin__":
        name = module + "." + name
    return name


Name = PurePosixPath
