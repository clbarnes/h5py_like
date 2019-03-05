import os
from abc import ABC, abstractmethod
from pathlib import Path

from .common import Mode
from .group import Group


class File(Group, ABC):
    @abstractmethod
    def __init__(self, name, mode=Mode.READ_WRITE_CREATE):
        self.filename: Path = Path(name).absolute()
        self.mode: Mode = Mode.from_str(mode)

    @property
    def parent(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        self.close()

    def flush(self):
        pass

    def close(self):
        pass
