"""RANSAC geometric verification filter.

Wraps OpenCV's findFundamentalMat, findHomography, and estimateAffine2D
to verify spatial consistency of matched keypoints. Supports fundamental
matrix, homography, and affine transform models.
"""

import cv2
import numpy as np

from config import settings


class RANSACFilter:
    def __init__(self, method=None, reproj_thresh=None, max_iters=None,
                 confidence=None, min_inliers=None):
        self.method = method or settings.RANSAC_METHOD
        self.reproj_thresh = reproj_thresh or settings.RANSAC_REPROJ_THRESH
        self.max_iters = max_iters or settings.RANSAC_MAX_ITERS
        self.confidence = confidence or settings.RANSAC_CONFIDENCE
        self.min_inliers = min_inliers or settings.RANSAC_MIN_INLIERS

    def filter_matches(self, query_points, db_points):
        query_pts = np.float32(query_points).reshape(-1, 1, 2)
        db_pts = np.float32(db_points).reshape(-1, 1, 2)

        if len(query_pts) < self.min_inliers:
            return np.array([]), None, np.zeros(len(query_points), dtype=bool)

        if self.method == "fundamental":
            model, mask = self._fit_fundamental(query_pts, db_pts)
        elif self.method == "homography":
            model, mask = self._fit_homography(query_pts, db_pts)
        elif self.method == "affine":
            model, mask = self._fit_affine(query_pts, db_pts)
        else:
            raise ValueError(f"Unknown RANSAC method: {self.method}")

        if mask is None:
            return np.array([]), None, np.zeros(len(query_points), dtype=bool)

        inlier_mask = mask.ravel().astype(bool)
        return inlier_mask, model, inlier_mask

    def _fit_fundamental(self, pts1, pts2):
        F, mask = cv2.findFundamentalMat(
            pts1, pts2,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=self.reproj_thresh,
            confidence=self.confidence,
            maxIters=self.max_iters
        )
        return F, mask

    def _fit_homography(self, pts1, pts2):
        H, mask = cv2.findHomography(
            pts1, pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.reproj_thresh,
            maxIters=self.max_iters,
            confidence=self.confidence
        )
        return H, mask

    def _fit_affine(self, pts1, pts2):
        M, mask = cv2.estimateAffine2D(
            pts1, pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.reproj_thresh,
            maxIters=self.max_iters,
            confidence=self.confidence
        )
        return M, mask

    def count_inliers(self, inlier_mask):
        if isinstance(inlier_mask, np.ndarray) and inlier_mask.size > 0:
            return int(np.sum(inlier_mask))
        return 0

    def passes_threshold(self, inlier_mask):
        return self.count_inliers(inlier_mask) >= self.min_inliers
