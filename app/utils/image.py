# Image loading, base64 encoding, hashing and format validation utilities

# Image loading, base64 encoding, hashing, and format validation utilities

import base64
import hashlib
from pathlib import Path

# Supported image formats and their MIME types
SUPPORTED_FORMATS = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


class ImageLoadError(Exception):
    """Raised when an image cannot be loaded or validated."""
    pass


def is_supported_format(image_path: Path) -> bool:
    """Returns True if the file extension is a supported image format."""
    return image_path.suffix.lower() in SUPPORTED_FORMATS


def get_mime_type(image_path: Path) -> str:
    """
    Returns the MIME type for a given image path.
    Raises ImageLoadError if the format is not supported.
    """
    mime_type = SUPPORTED_FORMATS.get(image_path.suffix.lower())
    if not mime_type:
        raise ImageLoadError(
            f"Unsupported image format: {image_path.suffix}. "
            f"Supported formats: {list(SUPPORTED_FORMATS.keys())}"
        )
    return mime_type


def load_image_as_base64(image_path: Path) -> str:
    """
    Loads an image file and returns it as a base64 encoded string.
    Raises ImageLoadError if the file cannot be read.
    """
    if not image_path.exists():
        raise ImageLoadError(f"Image file not found: {image_path}")

    if not is_supported_format(image_path):
        raise ImageLoadError(
            f"Unsupported image format: {image_path.suffix}"
        )

    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except OSError as e:
        raise ImageLoadError(f"Failed to read image file {image_path}: {e}")


def hash_image(image_path: Path) -> str:
    """
    Returns an MD5 hash of the image file contents.
    Used for duplicate detection before processing.
    Raises ImageLoadError if the file cannot be read.
    """
    try:
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except OSError as e:
        raise ImageLoadError(f"Failed to hash image file {image_path}: {e}")


def validate_image(image_path: Path) -> None:
    """
    Validates that an image file exists, is a supported format,
    and can be read without errors.
    Raises ImageLoadError with a descriptive message if validation fails.
    """
    if not image_path.exists():
        raise ImageLoadError(f"Image file not found: {image_path}")

    if not image_path.is_file():
        raise ImageLoadError(f"Path is not a file: {image_path}")

    if not is_supported_format(image_path):
        raise ImageLoadError(
            f"Unsupported image format '{image_path.suffix}' "
            f"for file: {image_path.name}"
        )

    try:
        with open(image_path, "rb") as f:
            f.read(1)  # attempt to read one byte to confirm file is readable
    except OSError as e:
        raise ImageLoadError(f"Image file is unreadable: {image_path}: {e}")