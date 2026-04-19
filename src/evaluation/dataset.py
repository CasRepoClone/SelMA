"""Load phototourism-style benchmark scenes with ground truth poses.

Supported data layouts
──────────────────────
1. COLMAP binary (phototourism training format):

       <scene>/
           dense/images/           ← photographs
           dense/sparse/           ← cameras.bin, images.bin, points3D.bin

2. Phototourism HDF5 (requires ``h5py``):

       <scene>/
           dense/images/  (or images/)
           calibration.h5          ← per-image K, R, T, imsize

3. Generic JSON:

       <scene>/
           images/
           calibration.json        ← {"img.jpg": {"K":…, "R":…, "T":…}}
           pairs.txt               ← optional "img1.jpg img2.jpg" per line
"""

import json
import itertools
from pathlib import Path

import numpy as np

from config import settings


class BenchmarkScene:
    """Represents a single benchmark scene with images and ground truth."""

    def __init__(self, scene_path):
        self.scene_path = Path(scene_path)
        if not self.scene_path.is_dir():
            raise FileNotFoundError(f"Scene directory not found: {self.scene_path}")

        self.images_dir = self._find_images_dir()
        self.calibration = self._load_calibration()
        self.image_names = sorted(self.calibration.keys())
        print(f"  Loaded scene with {len(self.image_names)} calibrated images")

    # ── directory discovery ────────────────────────────────────────────

    def _find_images_dir(self):
        candidates = [
            self.scene_path / "dense" / "images",
            self.scene_path / "images",
            self.scene_path,
        ]
        for d in candidates:
            if d.is_dir() and any(d.rglob(f"*{ext}") for ext in settings.IMAGE_EXTENSIONS):
                return d
        raise FileNotFoundError(
            f"No image directory found in {self.scene_path}. "
            f"Expected dense/images/ or images/ sub-folder."
        )

    # ── calibration loading ────────────────────────────────────────────

    def _load_calibration(self):
        # Priority: COLMAP binary → HDF5 → JSON
        colmap_dir = self.scene_path / "dense" / "sparse"
        h5_path = self.scene_path / "calibration.h5"
        json_path = self.scene_path / "calibration.json"

        if colmap_dir.is_dir() and (colmap_dir / "cameras.bin").is_file():
            return self._load_colmap_calibration(colmap_dir)
        if h5_path.is_file():
            return self._load_h5_calibration(h5_path)
        if json_path.is_file():
            return self._load_json_calibration(json_path)

        raise FileNotFoundError(
            f"No calibration found in {self.scene_path}. "
            f"Provide dense/sparse/ (COLMAP), calibration.h5, or calibration.json."
        )

    @staticmethod
    def _load_colmap_calibration(sparse_dir):
        from calibration.colmap_parser import load_colmap_calibration
        return load_colmap_calibration(sparse_dir)

    @staticmethod
    def _load_h5_calibration(path):
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required to read calibration.h5. "
                "Install it with: pip install h5py"
            )

        calib = {}
        with h5py.File(str(path), "r") as f:
            for name in f.keys():
                grp = f[name]
                K = np.array(grp["K"])
                R = np.array(grp["R"])
                T = np.array(grp["T"]).ravel()
                imsize = tuple(np.array(grp["imsize"]).ravel().astype(int)) if "imsize" in grp else None
                calib[name] = {"K": K, "R": R, "T": T, "imsize": imsize}
        return calib

    @staticmethod
    def _load_json_calibration(path):
        with open(path, "r") as f:
            raw = json.load(f)

        calib = {}
        for name, entry in raw.items():
            K = np.array(entry["K"], dtype=np.float64)
            R = np.array(entry["R"], dtype=np.float64)
            T = np.array(entry["T"], dtype=np.float64).ravel()
            imsize = tuple(entry["imsize"]) if "imsize" in entry else None
            calib[name] = {"K": K, "R": R, "T": T, "imsize": imsize}
        return calib

    # ── image access ───────────────────────────────────────────────────

    def get_image_path(self, name):
        path = self.images_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        return path

    # ── pair generation ────────────────────────────────────────────────

    def get_pairs(self, max_pairs=None, covisibility_threshold=None):
        """Return list of (img1_name, img2_name) pairs.

        Tries loading pair files in this order:
          1. new-vis-pairs/keys-th-<threshold>.npy  (phototourism format)
          2. pairs.txt  (one pair per line, space-separated)
          3. Covisibility-based pairs from camera geometry
        """
        pairs = self._try_load_vis_pairs(covisibility_threshold)
        if pairs is None:
            pairs = self._try_load_pairs_txt()
        if pairs is None:
            pairs = self._generate_covisibility_pairs()

        if max_pairs is not None and len(pairs) > max_pairs:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(pairs), size=max_pairs, replace=False)
            pairs = [pairs[i] for i in sorted(indices)]

        return pairs

    def _generate_covisibility_pairs(self, max_angle_deg=60.0):
        """Generate pairs of images likely to share visual content.

        Filters by:
          - Camera center distance (within median distance × 3)
          - Viewing direction angular difference < max_angle_deg
        """
        names = self.image_names
        n = len(names)

        # Compute camera centers and viewing directions
        centers = {}
        view_dirs = {}
        for name in names:
            c = self.calibration[name]
            R, T = c["R"], c["T"]
            # Camera center in world coords: C = -R^T @ T
            centers[name] = -R.T @ T
            # Viewing direction (z-axis of camera in world coords)
            view_dirs[name] = R.T @ np.array([0.0, 0.0, 1.0])

        # Compute all pairwise distances
        all_dists = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[names[i]] - centers[names[j]])
                all_dists.append(d)

        if not all_dists:
            return list(itertools.combinations(names, 2))

        median_dist = np.median(all_dists)
        max_dist = median_dist * 3.0

        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                n1, n2 = names[i], names[j]
                dist = np.linalg.norm(centers[n1] - centers[n2])
                if dist > max_dist:
                    continue

                # Viewing direction similarity
                cos_angle = np.dot(view_dirs[n1], view_dirs[n2])
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_angle))
                if angle > max_angle_deg:
                    continue

                pairs.append((n1, n2))

        print(f"  Generated {len(pairs)} covisibility pairs "
              f"(from {n*(n-1)//2} total, "
              f"dist<{max_dist:.1f}, angle<{max_angle_deg:.0f}°)")

        if not pairs:
            # Fallback: use all pairs
            pairs = list(itertools.combinations(names, 2))
            print(f"  Fallback: using all {len(pairs)} pairs")

        return pairs

    def _try_load_vis_pairs(self, threshold=None):
        vis_dir = self.scene_path / "new-vis-pairs"
        if not vis_dir.is_dir():
            return None

        threshold = threshold or 0.1
        npy_path = vis_dir / f"keys-th-{threshold}.npy"
        if not npy_path.is_file():
            # try first .npy found
            npy_files = sorted(vis_dir.glob("*.npy"))
            if not npy_files:
                return None
            npy_path = npy_files[0]

        raw = np.load(str(npy_path))
        pairs = []
        for row in raw:
            n1, n2 = str(row[0]), str(row[1])
            if n1 in self.calibration and n2 in self.calibration:
                pairs.append((n1, n2))
        return pairs if pairs else None

    def _try_load_pairs_txt(self):
        pairs_path = self.scene_path / "pairs.txt"
        if not pairs_path.is_file():
            return None

        pairs = []
        for line in pairs_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                n1, n2 = parts[0], parts[1]
                if n1 in self.calibration and n2 in self.calibration:
                    pairs.append((n1, n2))
        return pairs if pairs else None

    # ── ground truth geometry ──────────────────────────────────────────

    def get_relative_pose(self, img1_name, img2_name):
        """Return ground truth relative rotation R and translation t.

        R_rel, t_rel such that  p_2 = R_rel @ p_1 + t_rel  (in camera coords).
        Translation is unit-normalised (direction only).
        """
        c1 = self.calibration[img1_name]
        c2 = self.calibration[img2_name]

        R1, T1 = c1["R"], c1["T"]
        R2, T2 = c2["R"], c2["T"]

        R_rel = R2 @ R1.T
        t_rel = T2 - R_rel @ T1

        t_norm = np.linalg.norm(t_rel)
        if t_norm > 1e-8:
            t_rel = t_rel / t_norm

        return R_rel, t_rel

    def get_intrinsics(self, img_name):
        """Return 3×3 intrinsic matrix K for an image."""
        return self.calibration[img_name]["K"]

    def get_fundamental_matrix(self, img1_name, img2_name):
        """Compute ground truth fundamental matrix from calibration.

        F = K2^{-T} [t]_x R K1^{-1}   where R, t are relative pose.
        """
        K1 = self.get_intrinsics(img1_name)
        K2 = self.get_intrinsics(img2_name)
        R_rel, t_rel = self.get_relative_pose(img1_name, img2_name)

        tx = np.array([
            [0, -t_rel[2], t_rel[1]],
            [t_rel[2], 0, -t_rel[0]],
            [-t_rel[1], t_rel[0], 0],
        ])

        E = tx @ R_rel
        F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
        return F
