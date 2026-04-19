import cv2
import numpy as np

from config import settings


def detect_edges(image_gray, low_thresh=None, high_thresh=None):
    low_thresh = low_thresh or settings.CANNY_LOW_THRESH
    high_thresh = high_thresh or settings.CANNY_HIGH_THRESH
    return cv2.Canny(image_gray, low_thresh, high_thresh)


def sample_edge_points(edge_map, spacing=None):
    spacing = spacing or settings.PATCH_SPACING
    ys, xs = np.where(edge_map > 0)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=int)

    points = np.stack([xs, ys], axis=1)

    if spacing > 1 and len(points) > 0:
        indices = np.arange(0, len(points), spacing)
        points = points[indices]

    return points


def extract_patches(image, points, patch_size=None):
    patch_size = patch_size or settings.PATCH_SIZE
    half = patch_size // 2
    h, w = image.shape[:2]
    patches = []
    valid_points = []

    for x, y in points:
        x1, y1 = x - half, y - half
        x2, y2 = x + half, y + half

        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            continue

        patch = image[y1:y2, x1:x2].copy()
        patches.append(patch)
        valid_points.append((x, y))

    return patches, valid_points


def detect_edge_keypoints(gray, edge_map, max_corners=None,
                          quality_level=None, min_distance=None,
                          edge_proximity=None):
    """Detect Shi-Tomasi corners near edges for repeatable keypoints.

    Inspired by SuperPoint-style interest point detection: find the most
    distinctive locations (corners, junctions) that lie on or near Canny
    edges — keeping SelMA's edge-based identity while improving
    repeatability across viewpoints.
    """
    max_corners = max_corners or settings.KEYPOINT_MAX_CORNERS
    quality_level = quality_level or settings.KEYPOINT_QUALITY_LEVEL
    min_distance = min_distance or settings.KEYPOINT_MIN_DISTANCE
    edge_proximity = edge_proximity or settings.KEYPOINT_EDGE_PROXIMITY

    corners = cv2.goodFeaturesToTrack(
        gray, maxCorners=max_corners * 2,
        qualityLevel=quality_level,
        minDistance=min_distance,
    )

    if corners is None or len(corners) == 0:
        return np.empty((0, 2), dtype=int)

    # Dilate edge map so corners within `edge_proximity` pixels count
    kernel_size = edge_proximity * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    edge_dilated = cv2.dilate(edge_map, kernel)

    points = corners.reshape(-1, 2)
    valid = []
    for x, y in points:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < edge_dilated.shape[0] and 0 <= xi < edge_dilated.shape[1]:
            if edge_dilated[yi, xi] > 0:
                valid.append((xi, yi))

    if len(valid) > max_corners:
        valid = valid[:max_corners]

    return np.array(valid, dtype=int) if valid else np.empty((0, 2), dtype=int)


def get_edge_patches(image, patch_size=None, spacing=None,
                     low_thresh=None, high_thresh=None, max_patches=None):
    patch_size = patch_size or settings.PATCH_SIZE
    spacing = spacing or settings.PATCH_SPACING
    low_thresh = low_thresh or settings.CANNY_LOW_THRESH
    high_thresh = high_thresh or settings.CANNY_HIGH_THRESH
    max_patches = max_patches or settings.MAX_PATCHES

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    edge_map = detect_edges(gray, low_thresh, high_thresh)

    # Keypoint detection: Shi-Tomasi corners near edges (repeatable)
    # Falls back to uniform edge sampling if too few keypoints found
    if settings.USE_EDGE_KEYPOINTS:
        points = detect_edge_keypoints(gray, edge_map)
        if len(points) < 10:
            points = sample_edge_points(edge_map, spacing)
    else:
        points = sample_edge_points(edge_map, spacing)

    patches, valid_points = extract_patches(image, points, patch_size)

    if len(patches) > max_patches:
        indices = np.linspace(0, len(patches) - 1, max_patches, dtype=int)
        patches = [patches[i] for i in indices]
        valid_points = [valid_points[i] for i in indices]

    return patches, valid_points
