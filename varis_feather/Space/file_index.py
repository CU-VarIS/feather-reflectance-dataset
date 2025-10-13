

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class FileIndexEntry:
    name: str
    path_local: Path

    # If this has been measured or authored, is_source is False for caches
    is_source: bool

    present_local: bool = False
    date_local: datetime | None = None
    
    # None means we have not checked
    present_remote: bool | None = None
    date_remote: datetime | None = None

    # Size in bytes
    size: int = -1

class FileIndex:
    def __init__(self):
        self.entries: dict[str, FileIndexEntry] = {}

    def add(self, path_local: Path, name: str|None=None, is_source=True):
        if path_local is None:
            return
        
        name = name or path_local.name

        present_local = path_local.is_file()
        stat_local = self.stat(path_local) if present_local else None

        self.entries[name] = FileIndexEntry(
            name=name,
            path_local=path_local,
            is_source=is_source,
            present_local=present_local,
            date_local=datetime.fromtimestamp(stat_local.st_mtime) if stat_local else None,
            size=stat_local.st_size if stat_local else -1,
        )

    def items(self):
        return self.entries.items()

    def __iter__(self):
        return iter(self.entries.values())

    def name_to_srcpath(self) -> dict[str, Path]:
        return {e.name: e.path_local for e in self.entries.values() if e.present_local}
    
    def update(self, file_index: 'FileIndex'):
        self.entries.update(file_index.entries)

    _stat_cache: dict[Path, os.stat_result] = {}

    @classmethod
    def stat(cls, path: Path)-> os.stat_result:
        """Stat a file, but use os.scandir to cache the directory listing for speedup.
        Stat appears to be slow on my mount.
        """
        path = path.expanduser().resolve()

        if path not in cls._stat_cache:
            print("stat ", path.parent)
            for entry in os.scandir(path.parent):
                cls._stat_cache[Path(entry.path)] = entry.stat()
        
        return cls._stat_cache[path]
