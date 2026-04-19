"""Parse COLMAP binary model files (cameras.bin, images.bin).

Extracts per-image intrinsics (K), rotation (R), and translation (T)
from COLMAP SfM reconstructions as used by the phototourism benchmark.

Reference:
  https://colmap.github.io/format.html#binary-file-format
"""

import struct
from pathlib import Path

import numpy as np


# ── camera model id → (model_name, num_params) ────────────────────────

CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),    # f, cx, cy
    1: ("PINHOLE", 4),           # fx, fy, cx, cy
    2: ("SIMPLE_RADIAL", 4),     # f, cx, cy, k
    3: ("RADIAL", 5),            # f, cx, cy, k1, k2
    4: ("OPENCV", 8),            # fx, fy, cx, cy, k1, k2, p1, p2
    5: ("OPENCV_FISHEYE", 8),    # fx, fy, cx, cy, k1, k2, k3, k4
    6: ("FULL_OPENCV", 12),      # fx, fy, cx, cy, k1-k6, p1, p2
    7: ("FOV", 5),               # fx, fy, cx, cy, omega
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def _read_next(f, fmt):
    size = struct.calcsize(fmt)
    data = f.read(size)
    return struct.unpack(fmt, data)


def qvec_to_rotmat(qvec):
    """Convert COLMAP quaternion (w, x, y, z) to 3×3 rotation matrix."""
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*y*y - 2*z*z,   2*x*y - 2*w*z,       2*x*z + 2*w*y],
        [2*x*y + 2*w*z,       1 - 2*x*x - 2*z*z,   2*y*z - 2*w*x],
        [2*x*z - 2*w*y,       2*y*z + 2*w*x,       1 - 2*x*x - 2*y*y],
    ])
    return R


def params_to_K(model_id, params, width, height):
    """Build 3×3 intrinsic matrix from COLMAP camera parameters."""
    model_name = CAMERA_MODELS.get(model_id, ("UNKNOWN", 0))[0]

    if model_name == "SIMPLE_PINHOLE":
        f, cx, cy = params[0], params[1], params[2]
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

    elif model_name == "PINHOLE":
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    elif model_name in ("SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        f, cx, cy = params[0], params[1], params[2]
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

    elif model_name in ("RADIAL", "RADIAL_FISHEYE"):
        f, cx, cy = params[0], params[1], params[2]
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

    elif model_name in ("OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV",
                         "THIN_PRISM_FISHEYE"):
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    elif model_name == "FOV":
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    else:
        raise ValueError(f"Unsupported COLMAP camera model: {model_name} (id={model_id})")


# ── binary readers ─────────────────────────────────────────────────────

def read_cameras_binary(path):
    """Parse cameras.bin → dict {camera_id: {model_id, width, height, params, K}}"""
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = _read_next(f, "<Q")[0]
        for _ in range(num_cameras):
            camera_id = _read_next(f, "<i")[0]
            model_id = _read_next(f, "<i")[0]
            width = _read_next(f, "<Q")[0]
            height = _read_next(f, "<Q")[0]
            num_params = CAMERA_MODELS.get(model_id, ("UNKNOWN", 0))[1]
            params = np.array(_read_next(f, f"<{num_params}d"))
            K = params_to_K(model_id, params, width, height)
            cameras[camera_id] = {
                "model_id": model_id,
                "width": int(width),
                "height": int(height),
                "params": params,
                "K": K,
            }
    return cameras


def read_images_binary(path):
    """Parse images.bin → dict {image_name: {camera_id, R, T, qvec, tvec}}"""
    images = {}
    with open(path, "rb") as f:
        num_images = _read_next(f, "<Q")[0]
        for _ in range(num_images):
            image_id = _read_next(f, "<i")[0]
            qvec = np.array(_read_next(f, "<4d"))
            tvec = np.array(_read_next(f, "<3d"))
            camera_id = _read_next(f, "<i")[0]

            # image name (null-terminated string)
            name_chars = []
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_chars.append(c.decode("utf-8"))
            name = "".join(name_chars)

            # 2D points (skip for our purposes)
            num_points2D = _read_next(f, "<Q")[0]
            # Each point: x, y (double), point3D_id (long long)
            f.read(num_points2D * (8 + 8 + 8))

            R = qvec_to_rotmat(qvec)
            images[name] = {
                "image_id": image_id,
                "camera_id": camera_id,
                "R": R,
                "T": tvec,
                "qvec": qvec,
                "tvec": tvec,
            }
    return images


# ── high-level loader ──────────────────────────────────────────────────

def load_colmap_calibration(sparse_dir):
    """Load COLMAP binary model and return per-image calibration dict.

    Parameters
    ----------
    sparse_dir : str | Path
        Directory containing cameras.bin and images.bin.

    Returns
    -------
    calibration : dict
        {image_name: {"K": 3x3, "R": 3x3, "T": 3-vec, "imsize": (w, h)}}
    """
    sparse_dir = Path(sparse_dir)
    cameras_path = sparse_dir / "cameras.bin"
    images_path = sparse_dir / "images.bin"

    if not cameras_path.is_file():
        raise FileNotFoundError(f"cameras.bin not found in {sparse_dir}")
    if not images_path.is_file():
        raise FileNotFoundError(f"images.bin not found in {sparse_dir}")

    cameras = read_cameras_binary(cameras_path)
    images = read_images_binary(images_path)

    calibration = {}
    for name, img_data in images.items():
        cam = cameras[img_data["camera_id"]]
        calibration[name] = {
            "K": cam["K"],
            "R": img_data["R"],
            "T": img_data["T"],
            "imsize": (cam["width"], cam["height"]),
        }

    return calibration
