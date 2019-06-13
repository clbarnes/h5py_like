from enum import Enum


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


class StrEnum(str, Enum):
    """Enum subclass which members are also instances of str
    and directly comparable to strings. str type is forced at declaration.
    """
    def __new__(cls, *args):
        for arg in args:
            if not isinstance(arg, str):
                raise TypeError('Not text %s:' % arg)
        return super(StrEnum, cls).__new__(cls, *args)


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
