import asyncio
from datetime import datetime
import os
from pathlib import Path

import aiofiles
from aiobotocore.session import get_session
from pytz import UTC
from tqdm import tqdm

from .. import load_standard_capture
from ..Paths import SCENES, STORAGE_BUCKET, STORAGE_ID_AND_KEY, STORAGE_URL
from ..Space.file_index import FileIndex




async def s3_sync_filter_uploads(client, bucket_name: str, prefix: str, remote_to_local: dict[str, Path]) -> list[tuple[str, str, str, int]]:
    """Scan local and remote files to find cases where the local counterpart is newer.
    Returns list of entries:
        local path - absolute
        bucket
        remote path
        size in bytes
    """

    # List objects in target prefix, paginate to make sure we get all objects
    # Get modified date and size
    existing_objects : dict[str, tuple[ datetime, int]] = {}
    async for response in client.get_paginator('list_objects_v2').paginate(Bucket=bucket_name, Prefix=prefix):
        existing_objects.update({
            obj["Key"]: (obj["LastModified"], obj["Size"]) 
            for obj in response.get("Contents", [])
        })

    to_upload = []
    for remote_path, local_path in remote_to_local.items():
        if not local_path.exists():
            continue

        local_stat = FileIndex.stat(local_path)
        local_mod_date = datetime.fromtimestamp(local_stat.st_mtime, tz = UTC)
        local_size = local_stat.st_size

        remote_full_path = f"{prefix}/{remote_path}"
        remote_mod_date, remote_size = existing_objects.get(remote_full_path, (None, -1))

        local_newer = remote_mod_date is None or local_mod_date > remote_mod_date
        size_different = local_size != remote_size

        if local_newer or size_different:
            to_upload.append((str(local_path.resolve()), bucket_name, remote_full_path, local_size))

    return to_upload

async def s3_sync_perform_uploads(client, to_upload: list[tuple[str, str, str, int]], num_concurrent_uploads: int = 4):
    """
    Perform uploads of files to S3 bucket.
    to_upload is a list of tuples:
        local path - absolute
        bucket
        remote path
        size in bytes
    """

    # Semaphore to limit concurrent uploads
    semaphore = asyncio.Semaphore(num_concurrent_uploads)

    bytes_total = sum(size for _, _, _, size in to_upload)
    print(f"Uploading {len(to_upload)} changed files, total size {bytes_total / (1024*1024):.2f} MB")


    msg = "START"
    bytes_done = 0
    pbar = None

    def msg_file_done(local_path, size):
        nonlocal bytes_done, msg, pbar
        bytes_done += size
        msg = f"Last: {Path(local_path).name} - {bytes_done / (1024*1024):.2f} / {bytes_total / (1024*1024):.2f} MB - {(bytes_done/bytes_total)*100:.1f}%"

        if pbar:
            pbar.set_postfix_str(msg)

        return msg

    async def upload_file(local_path, bucket, remote_path, size):
        async with semaphore:
            print(f"Uploading {local_path} {size / (1024*1024):.2f} MB")
            async with aiofiles.open(local_path, 'rb') as f:
                content = await f.read()
                # Upload the file
                try:
                    await client.put_object(Bucket=bucket, Key=remote_path, Body=content)

                    msg_file_done(Path(local_path).name, size)

                except Exception as e:
                    print(f"Failed to upload {local_path} to {remote_path}: {e}")
        
    futures = [
        upload_file(local_path, bucket, remote_path, size)
        for local_path, bucket, remote_path, size in to_upload
    ]

    # Use tqdm to show progress
    for future in (pbar := tqdm(asyncio.as_completed(futures), total=len(futures), desc="Uploading")):
        await future
        # pbar.set_postfix_str(msg)



async def upload(num_concurrent_uploads: int = 4, first: int = 0):


    to_upload = []
    num_local = 0

    load_kw = dict(
        drop_outliers=False,
        use_index=False,
    )

    async with get_session().create_client(
        "s3",
        endpoint_url=STORAGE_URL,
        aws_secret_access_key=STORAGE_ID_AND_KEY[1],
        aws_access_key_id=STORAGE_ID_AND_KEY[0]
    ) as client:

        for cap_name in SCENES:
            retro, olat = load_standard_capture(cap_name, olat_kwargs=load_kw)
            t = "rectified" if "stereo" not in cap_name.lower() else "stereov1"

            olat_index = olat.file_index()
            to_upload += await s3_sync_filter_uploads(
                client,
                bucket_name=STORAGE_BUCKET,
                prefix=f"{olat.name}/olat_iso_{t}", 
                remote_to_local=olat_index,
            )
            num_local += len(olat_index)

            if retro:
                retro_index = retro.file_index()
                to_upload += await s3_sync_filter_uploads(
                    client,
                    bucket_name=STORAGE_BUCKET,
                    prefix=f"{retro.name}/retro_iso_{t}",
                    remote_to_local=retro_index,
                )

                num_local += len(retro_index)

        print(f"By directory:")
        num_per_dir: dict[Path, int] = {}
        for local_path, _, _, _ in to_upload:
            p = Path(local_path).parent
            num_per_dir[p] = num_per_dir.get(p, 0) + 1

        for p, n in num_per_dir.items():
            print(f"  {n} in {p}")

        print(f"Total local files: {num_local}, cached {num_local - len(to_upload)} to upload: {len(to_upload)}")

        # Confirmation
        ans = input(f"Proceed with upload of {len(to_upload)} files? (y/N) ")
        if ans.lower() != "y":
            return

        await s3_sync_perform_uploads(client, to_upload, num_concurrent_uploads=num_concurrent_uploads)



