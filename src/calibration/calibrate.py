"""Camera calibration from checkerboard images.

Uses OpenCV's standard checkerboard corner detection pipeline:
  1. Find chessboard corners
  2. Refine to sub-pixel accuracy
  3. Compute intrinsic matrix K, distortion coefficients, and per-image poses

Usage:
    from calibration.calibrate import CameraCalibrator
    cal = CameraCalibrator(board_size=(9, 6), square_size=25.0)
    cal.add_images(["img1.jpg", "img2.jpg", ...])
    K, dist, per_image = cal.calibrate()
"""

from pathlib import Path

import cv2
import numpy as np


class CameraCalibrator:
    """Calibrate a camera from multiple checkerboard views.

    Parameters
    ----------
    board_size : tuple (cols, rows)
        Number of *inner corners* per row and column (e.g. (9, 6) for a
        10×7 squares board).
    square_size : float
        Physical size of one square in your chosen unit (mm, cm, …).
        Determines the scale of the extrinsics.
    """

    def __init__(self, board_size=(9, 6), square_size=25.0):
        self.board_size = board_size
        self.square_size = square_size

        # 3-D object points for one board view (z=0 plane)
        objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[
            0:board_size[0], 0:board_size[1]
        ].T.reshape(-1, 2) * square_size
        self._objp_template = objp

        self._obj_points = []   # list of Nx3 arrays  (one per accepted image)
        self._img_points = []   # list of Nx2 arrays  (detected corners)
        self._image_paths = []  # for bookkeeping
        self._image_size = None # (width, height)

    # ── corner detection ───────────────────────────────────────────────

    def add_image(self, image_or_path):
        """Detect checkerboard corners in a single image.

        Parameters
        ----------
        image_or_path : str | Path | ndarray
            File path or pre-loaded BGR image.

        Returns
        -------
        found : bool
            True if corners were detected.
        corners : ndarray (N, 1, 2) | None
            Refined corners in pixel coordinates, or None.
        """
        if isinstance(image_or_path, (str, Path)):
            img = cv2.imread(str(image_or_path))
            path = str(image_or_path)
        else:
            img = image_or_path
            path = "<array>"

        if img is None:
            return False, None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        h, w = gray.shape[:2]

        if self._image_size is None:
            self._image_size = (w, h)

        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
                 | cv2.CALIB_CB_NORMALIZE_IMAGE
                 | cv2.CALIB_CB_FAST_CHECK)
        found, corners = cv2.findChessboardCorners(gray, self.board_size, flags)

        if not found:
            return False, None

        # Sub-pixel refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        self._obj_points.append(self._objp_template.copy())
        self._img_points.append(corners)
        self._image_paths.append(path)

        return True, corners

    def add_images(self, paths):
        """Batch-add images.  Returns number of successful detections."""
        count = 0
        for p in paths:
            ok, _ = self.add_image(p)
            if ok:
                count += 1
        return count

    # ── calibration ────────────────────────────────────────────────────

    def calibrate(self):
        """Run camera calibration.

        Returns
        -------
        K : ndarray (3, 3)
            Intrinsic camera matrix.
        dist_coeffs : ndarray (1, 5)
            Distortion coefficients [k1, k2, p1, p2, k3].
        per_image : list[dict]
            Per-image results with keys:
              path, R (3×3), T (3×1), rvec, tvec, reproj_error
        """
        if len(self._obj_points) < 3:
            raise RuntimeError(
                f"Need at least 3 accepted images for calibration, "
                f"got {len(self._obj_points)}."
            )

        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            self._obj_points, self._img_points,
            self._image_size, None, None,
        )

        per_image = []
        for i in range(len(rvecs)):
            R, _ = cv2.Rodrigues(rvecs[i])

            # Per-image reprojection error
            projected, _ = cv2.projectPoints(
                self._obj_points[i], rvecs[i], tvecs[i], K, dist
            )
            err = cv2.norm(self._img_points[i], projected, cv2.NORM_L2)
            err /= len(projected)

            per_image.append({
                "path": self._image_paths[i],
                "R": R,
                "T": tvecs[i].ravel(),
                "rvec": rvecs[i].ravel(),
                "tvec": tvecs[i].ravel(),
                "reproj_error": float(err),
            })

        print(f"  Calibration RMS reprojection error: {rms:.4f} px")
        print(f"  Images used: {len(per_image)}")
        return K, dist, per_image

    # ── undistortion ───────────────────────────────────────────────────

    @staticmethod
    def undistort(image, K, dist_coeffs, alpha=1.0):
        """Undistort an image using calibration results.

        Parameters
        ----------
        alpha : float
            0 = crop all black borders, 1 = keep all pixels.
        """
        h, w = image.shape[:2]
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), alpha)
        undistorted = cv2.undistort(image, K, dist_coeffs, None, new_K)
        return undistorted, new_K

    # ── visualisation helper ───────────────────────────────────────────

    @staticmethod
    def draw_corners(image, board_size, corners, found):
        """Draw detected corners on image (modifies in-place, returns it)."""
        vis = image.copy()
        cv2.drawChessboardCorners(vis, board_size, corners, found)
        return vis

    # ── serialisation ──────────────────────────────────────────────────

    def save_calibration(self, path, K, dist_coeffs, per_image):
        """Save calibration to a JSON file (benchmark-compatible)."""
        import json
        data = {
            "camera_matrix": K.tolist(),
            "dist_coeffs": dist_coeffs.tolist(),
            "rms_error": None,
            "images": {},
        }
        for entry in per_image:
            name = Path(entry["path"]).name
            data["images"][name] = {
                "K": K.tolist(),
                "R": entry["R"].tolist(),
                "T": entry["T"].tolist(),
                "reproj_error": entry["reproj_error"],
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Calibration saved to {path}")

    @staticmethod
    def load_calibration(path):
        """Load calibration from JSON. Returns K, dist_coeffs, per_image dict."""
        import json
        with open(path, "r") as f:
            data = json.load(f)
        K = np.array(data["camera_matrix"], dtype=np.float64)
        dist = np.array(data["dist_coeffs"], dtype=np.float64)
        return K, dist, data.get("images", {})
