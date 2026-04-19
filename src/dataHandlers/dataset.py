"""Image loading and directory listing utilities."""

from pathlib import Path

import cv2

from config import settings


def load_image(path, color=True):
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def list_images(directory):
    directory = Path(directory)
    paths = []
    for ext in settings.IMAGE_EXTENSIONS:
        paths.extend(directory.rglob(f"*{ext}"))
    return sorted(paths)


def get_db_images():
    return list_images(settings.DB_DIR)


def get_query_images():
    return list_images(settings.QUERY_DIR)
