# Directory scanning with configurable depth control

from pathlib import Path
from enum import Enum
from app.utils.image import is_supported_format


class DirectoryWalkError(Exception):
    """Raised when a directory cannot be scanned."""
    pass


class ImageSortOrder(Enum):
    """
    Controls the order in which image files are returned.

    FILESYSTEM: preserves natural filesystem order (default).
                Use when images were scanned or saved in front-back sequence.
    ALPHABETICAL: sorts by filename alphabetically.
                  Use when filenames reflect a meaningful sequence.
    CREATION_TIME: sorts by file creation timestamp.
                   Use when creation time reflects scan order.
    """
    FILESYSTEM = "filesystem"
    ALPHABETICAL = "alphabetical"
    CREATION_TIME = "creation_time"


def _walk(
    current_dir: Path,
    max_depth: int
) -> list[Path]:
    """
    Internal recursive helper that collects image files up to max_depth.

    Args:
        current_dir: The directory currently being scanned
        max_depth: Remaining depth levels to recurse into.
                   0 means flat scan only.
                  -1 means unlimited recursion.
    """
    images = [
        f for f in current_dir.glob("*")
        if f.is_file() and is_supported_format(f)
    ]

    if max_depth > 0 or max_depth == -1:
        for subdir in current_dir.glob("*"):
            if subdir.is_dir():
                next_depth = max_depth - 1 if max_depth != -1 else -1
                images.extend(_walk(subdir, next_depth))

    return images


def get_image_files(
    directory: Path,
    max_depth: int = 0,
    sort_order: ImageSortOrder = ImageSortOrder.FILESYSTEM
) -> list[Path]:
    """
    Returns a list of supported image files in a directory.

    Args:
        directory: Path to the directory to scan
        max_depth: How many levels of subdirectories to recurse into.
                   0 = flat scan (top-level only, default)
                   1 = one level of subdirectories
                  -1 = unlimited recursion
        sort_order: Controls the order images are returned in.
                    Defaults to FILESYSTEM to preserve natural scan order,
                    which is important for front-back pairing logic.

    Raises DirectoryWalkError if the directory does not exist or is not
    accessible.
    """
    _validate_directory(directory)

    try:
        images = _walk(directory, max_depth)
    except OSError as e:
        raise DirectoryWalkError(
            f"Failed to scan directory {directory}: {e}"
        )

    if sort_order == ImageSortOrder.ALPHABETICAL:
        return sorted(images)
    elif sort_order == ImageSortOrder.CREATION_TIME:
        return sorted(images, key=lambda f: f.stat().st_ctime)
    else:
        return images


def get_subdirectories(
    directory: Path,
    max_depth: int = 1,
    sort_order: ImageSortOrder = ImageSortOrder.FILESYSTEM
) -> list[Path]:
    """
    Returns a list of subdirectories within a directory.
    Used by Mixed mode to identify set subdirectories.

    Args:
        directory: Path to the directory to scan
        max_depth: How many levels deep to look for subdirectories.
                   1 = immediate subdirectories only (default)
                  -1 = unlimited recursion
        sort_order: Controls the order subdirectories are returned in.

    Raises DirectoryWalkError if the directory does not exist or is not
    accessible.
    """
    _validate_directory(directory)

    try:
        if max_depth == -1:
            all_items = directory.rglob("*")
        else:
            all_items = _walk_dirs(directory, max_depth)

        subdirs = [item for item in all_items if item.is_dir()]

    except OSError as e:
        raise DirectoryWalkError(
            f"Failed to scan directory {directory}: {e}"
        )

    if sort_order == ImageSortOrder.ALPHABETICAL:
        return sorted(subdirs)
    elif sort_order == ImageSortOrder.CREATION_TIME:
        return sorted(subdirs, key=lambda f: f.stat().st_ctime)
    else:
        return subdirs


def _walk_dirs(
    current_dir: Path,
    max_depth: int
) -> list[Path]:
    """
    Internal recursive helper that collects subdirectories up to max_depth.
    """
    subdirs = [
        item for item in current_dir.glob("*")
        if item.is_dir()
    ]

    if max_depth > 1:
        for subdir in subdirs:
            subdirs.extend(_walk_dirs(subdir, max_depth - 1))

    return subdirs


def count_image_files(
    directory: Path,
    max_depth: int = 0
) -> int:
    """
    Returns the count of supported image files in a directory.
    Useful for progress tracking before processing begins.
    """
    return len(get_image_files(directory, max_depth))


def _validate_directory(directory: Path) -> None:
    """
    Validates that a path exists and is a directory.
    Raises DirectoryWalkError if validation fails.
    """
    if not directory.exists():
        raise DirectoryWalkError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise DirectoryWalkError(f"Path is not a directory: {directory}")