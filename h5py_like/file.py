from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path

from .common import Mode


class FileMixin(ABC):
    @abstractmethod
    def __init__(self, name, mode=Mode.READ_WRITE_CREATE):
        self.filename: Path = Path(name).absolute()
        self._mode: Mode = Mode.from_str(mode)

    @property
    def parent(self) -> FileMixin:
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        self.close()

    def flush(self):
        pass

    def close(self):
        pass

    @property
    def mode(self):
        return self._mode

    @property
    def name(self):
        return '/'

    def __eq__(self, other):
        return all((
            isinstance(other, type(self)),
            self.filename == other.filename,
            self.mode == other.mode
        ))
