
from pathlib import Path

from ..Paths import DIR_DATASET



def rename_captures(dir_category: Path, fmt_full: str, fmt_rectified: str):
    renames = []
    for dir_capture in dir_category.iterdir():
        if dir_capture.name.endswith("Stereo"):
            continue

        dir_rectified = dir_capture / "rectified"
        if dir_rectified.is_symlink():
            print("Warning symlink: ", dir_rectified)
            continue

        if dir_rectified.is_dir():
            renames.append((dir_rectified, Path(dir_capture.name) / fmt_rectified))

        if any(f.is_file() for f in dir_capture.iterdir()):
            renames.append((dir_capture, Path(dir_capture.name) / fmt_full))

    return renames

def main(dir_src: Path, dir_dest: Path = DIR_DATASET / "captures"):
    """Rename capture directories.
    Examples:
        * OLAT rectified
            FullScattering/FeatherName/rectified
            ->
            captures/FeatherName/olat_iso_rectified
        
        * OLAT unrectified
            FullScattering/FeatherName/
            ->
            captures/FeatherName/olat_iso_original

        * Retroreflection rectified
            Retroreflection/128x1/FeatherName/rectified
            ->
            captures/FeatherName/retro_128x1_rectified

        * Retroreflection unrectified
            Retroreflection/128x1/FeatherName/
            ->
            captures/FeatherName/retro_128x1_original

    """

    dir_src = Path(dir_src)
    dir_dest = Path(dir_dest)

    rs = []
    rs += rename_captures(dir_src / "Retroreflection" / "128x1", "retro_128x1_original", "retro_128x1_rectified")
    rs += rename_captures(dir_src / "FullScattering", "olat_iso_original", "olat_iso_rectified")

    rs = [(src, dir_dest / dst) for src, dst in rs]

    # print(len(rs))
    # print("\n".join(f'mv {src} -> {dst}' for src, dst in rs))


    # Summary: all directories to be moved, plus one line for a group of files from the same dir
    summary : dict[Path, tuple[Path, int]] = {}
    for src, dst in rs:
        if src.is_dir():
            summary[src] = (dst, -1)
        else:
            summary[src.parent] = (dst.parent, summary.get(src.parent, (None, 0))[1] + 1)

    print(f"Total moves: {len(rs)} in {len(summary)} source directories")
    for src_path, (dst_path, num_files) in summary.items():
        if num_files >= 0:
            print(f"\t{num_files} files: {src_path} -> {dst_path}")
        else:
            print(f"\tDIR {src_path}: -> {dst_path}")

    # Ask for confirmation
    ans = input("Proceed with moves? (y/N) ")
    if ans.lower() == "y":
        for src, dst in rs:
            if dst.exists():
                print(f"Warning exists {dst}")
            else:
                print(f"Ok to move to {dst}")
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.rename(dst)


