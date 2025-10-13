

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import requests

import aiofiles
from aiobotocore.session import get_session
from pytz import UTC
from ..Paths import SCENES, STORAGE_BUCKET, STORAGE_ID_AND_KEY, STORAGE_URL


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

    differs: bool | None = None

    # @property
    # def path_remote(self) -> str:
    #     return f"{STORAGE_URL}/{STORAGE_BUCKET}/{self.name}"

class FileIndex:
    def __init__(self, name: str = "", dir_src: Path | None = None):
        self.name = name
        self.dir_src = dir_src
        self.entries: dict[str, FileIndexEntry] = {}

    @property
    def storage_prefix(self) -> str:
        return f"{self.name}/{self.dir_src.name}"

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
            date_local=datetime.fromtimestamp(stat_local.st_mtime, tz = UTC) if stat_local else None,
            size=stat_local.st_size if stat_local else -1,
        )

    def items(self):
        return self.entries.items()

    def __iter__(self):
        return iter(self.entries.values())

    def name_to_srcpath(self) -> dict[str, Path]:
        return {e.name: e.path_local for e in self.entries.values() if e.present_local}
    
    def update(self, file_index: 'FileIndex'):
        assert file_index.name == self.name, "Can only merge FileIndex of the same name"
        assert file_index.dir_src == self.dir_src, "Can only merge FileIndex with the same source dir"
        self.entries.update(file_index.entries)


    @classmethod
    def download_simple(cls, remote_name: str, local_dir: Path, overwrite: bool = False):
        url = f"{STORAGE_URL}/{STORAGE_BUCKET}/{remote_name}"
        local_path = local_dir / Path(remote_name).name

        if local_path.exists() and not overwrite:
            return local_path

        response = requests.get(url)
        response.raise_for_status()

        local_dir.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(response.content)
        return local_path

    @classmethod
    def download_starter(cls, cap_name: str, cap_dir: Path):
        """Ensure the starter files are included"""

        cap_dir = Path(cap_dir)
        variant = cap_dir.name
        prefix = f"{cap_name}/{variant}"

        starter_files = [
            "index_frames.csv",
            "000_subcrops_choice.svg",
        ]

        for fname in starter_files:
            try:
                cls.download_simple(f"{prefix}/{fname}", cap_dir, overwrite=False)
            except Exception as e:
                print(f"Failed to download {fname} for {cap_name} in {variant}: {e}")


    # @classmethod
    # def download_simple(self, name: str):
        # url = f"{STORAGE_URL}/{STORAGE_BUCKET}/{self.storage_prefix}/{name}"


    _stat_cache: dict[Path, os.stat_result] = {}

    @classmethod
    def stat(cls, path: Path)-> os.stat_result:
        """Stat a file, but use os.scandir to cache the directory listing for speedup.
        Stat appears to be slow on my mount.
        """
        path = path.expanduser().resolve()

        if path not in cls._stat_cache:
            for entry in os.scandir(path.parent):
                cls._stat_cache[Path(entry.path)] = entry.stat()
        
        return cls._stat_cache[path]


    async def check_remote_status(self, client, bucket_name: str = STORAGE_BUCKET):
        """Scan local and remote files to find cases where the local counterpart is newer.
        Returns list of entries:
            local path - absolute
            bucket
            remote path
            size in bytes
        """

        prefix = self.storage_prefix

        # List objects in target prefix, paginate to make sure we get all objects
        async for response in client.get_paginator('list_objects_v2').paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in response.get("Contents", []):
                name = obj["Key"].removeprefix(f"{prefix}/")

                if (entry := self.entries.get(name)):
                    remote_size = obj["Size"]
                    entry.present_remote = True
                    entry.date_remote = obj["LastModified"]

                    if not entry.present_local:
                        entry.differs = True
                        entry.size = remote_size

                    else:
                        local_newer = entry.date_remote is None or entry.date_local > entry.date_remote
                        size_different = entry.size != remote_size

                        entry.differs = local_newer or size_different

