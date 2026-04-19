"""Generate a small synthetic benchmark scene for testing.

Creates a scene directory with:
  - images/ — checkerboard images rendered from different viewpoints
  - calibration.json — per-image K, R, T
  - pairs.txt — image pairs to evaluate

Usage:
    python scripts/create_test_scene.py [--output data/phototourism/sacre_coeur]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


def rotation_matrix(axis, angle_deg):
    """Rodrigues rotation matrix around a unit axis by angle_deg degrees."""
    angle = math.radians(angle_deg)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def render_checkerboard(width, height, K, R, T, board_size=8, square_size=0.5):
    """Render a 3D checkerboard onto a virtual camera at pose (R, T).

    The checkerboard sits at z=2 in world coords, spanning ±board_extent.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    board_extent = board_size * square_size / 2.0

    # Project grid of 3D world points → 2D
    step = square_size / 4.0
    xs = np.arange(-board_extent, board_extent, step)
    ys = np.arange(-board_extent, board_extent, step)

    for wx in xs:
        for wy in ys:
            wz = 2.0  # board sits at z=2 in world
            pw = np.array([wx, wy, wz])
            pc = R @ pw + T  # camera coords
            if pc[2] <= 0.1:
                continue
            px = K @ pc
            u, v = int(px[0] / px[2]), int(px[1] / px[2])
            if 0 <= u < width and 0 <= v < height:
                # Determine checker colour
                ci = int(math.floor((wx + board_extent) / square_size))
                cj = int(math.floor((wy + board_extent) / square_size))
                if (ci + cj) % 2 == 0:
                    color = (240, 240, 240)
                else:
                    color = (40, 40, 40)
                cv2.circle(img, (u, v), 1, color, -1)

    # Add some texture / noise to make edges detectable
    noise = np.random.randint(0, 15, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)

    # Draw some edges — horizontal and vertical lines on the board
    for i in range(board_size + 1):
        coord = -board_extent + i * square_size
        # horizontal line at y=coord
        pts_3d = np.array([[x, coord, 2.0] for x in np.linspace(-board_extent, board_extent, 200)])
        _draw_3d_line(img, pts_3d, K, R, T, width, height)
        # vertical line at x=coord
        pts_3d = np.array([[coord, y, 2.0] for y in np.linspace(-board_extent, board_extent, 200)])
        _draw_3d_line(img, pts_3d, K, R, T, width, height)

    return img


def _draw_3d_line(img, pts_3d, K, R, T, width, height):
    prev = None
    for pw in pts_3d:
        pc = R @ pw + T
        if pc[2] <= 0.1:
            prev = None
            continue
        px = K @ pc
        u, v = int(px[0] / px[2]), int(px[1] / px[2])
        cur = (u, v)
        if prev is not None and 0 <= u < width and 0 <= v < height:
            cv2.line(img, prev, cur, (128, 128, 128), 1)
        prev = cur if (0 <= u < width and 0 <= v < height) else None


def generate_scene(output_dir, n_images=10, width=640, height=480):
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Intrinsics (same for all images)
    fx, fy = 500.0, 500.0
    cx, cy = width / 2.0, height / 2.0
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ])

    calibration = {}
    image_names = []

    for i in range(n_images):
        # Vary camera position in a circular arc looking at the checkerboard
        angle = (i / n_images) * 60 - 30  # -30° to +30°
        y_offset = (i / n_images) * 1.0 - 0.5  # slight vertical shift

        R = rotation_matrix(np.array([0, 1, 0]), angle)
        # Camera positioned at ~z=-1 looking toward z=+2 (the board)
        T = np.array([0.0, y_offset, 0.0])  # translation in camera frame

        name = f"img_{i:04d}.jpg"
        image_names.append(name)

        img = render_checkerboard(width, height, K, R, T)
        cv2.imwrite(str(images_dir / name), img)

        calibration[name] = {
            "K": K.tolist(),
            "R": R.tolist(),
            "T": T.tolist(),
            "imsize": [width, height],
        }

    # Save calibration
    calib_path = output_dir / "calibration.json"
    with open(calib_path, "w") as f:
        json.dump(calibration, f, indent=2)

    # Generate all pairs
    pairs_path = output_dir / "pairs.txt"
    with open(pairs_path, "w") as f:
        for i in range(len(image_names)):
            for j in range(i + 1, len(image_names)):
                f.write(f"{image_names[i]} {image_names[j]}\n")

    print(f"Created synthetic scene in {output_dir}")
    print(f"  {n_images} images ({width}x{height})")
    print(f"  {len(image_names) * (len(image_names)-1) // 2} pairs")
    print(f"  calibration.json + pairs.txt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str,
                   default="data/phototourism/sacre_coeur")
    p.add_argument("--n-images", type=int, default=10)
    args = p.parse_args()
    generate_scene(args.output, n_images=args.n_images)
