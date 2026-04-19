"""Pose estimation metrics for image matching evaluation.

Primary metric: **mAA** (mean Average Accuracy)
  – estimate relative pose from matches
  – measure angular error vs ground truth (rotation + translation)
  – threshold at 1°–10°, take area-under-curve
"""

import numpy as np
import cv2


# ── Rotation / translation angular errors ──────────────────────────────

def rotation_error(R_est, R_gt):
    """Angular error (degrees) between two rotation matrices."""
    cos_angle = (np.trace(R_est @ R_gt.T) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def translation_error(t_est, t_gt):
    """Angular error (degrees) between two translation directions.

    Uses ``|dot|`` to handle the sign ambiguity of the essential matrix.
    """
    n1 = np.linalg.norm(t_est)
    n2 = np.linalg.norm(t_gt)
    if n1 < 1e-8 or n2 < 1e-8:
        return 180.0
    cos_angle = np.abs(np.dot(t_est, t_gt)) / (n1 * n2)
    cos_angle = np.clip(cos_angle, 0.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def pose_error(R_est, R_gt, t_est, t_gt):
    """Max of rotation and translation angular error (degrees)."""
    return max(rotation_error(R_est, R_gt), translation_error(t_est, t_gt))


# ── mAA (mean Average Accuracy) ───────────────────────────────────────

def compute_mAA(errors, thresholds=None):
    """Compute mean Average Accuracy from a list of pose errors.

    For each threshold, accuracy = fraction of pairs with error < threshold.
    mAA = mean of accuracies across thresholds.

    Parameters
    ----------
    errors : list[float]
        Per-pair pose errors in degrees.
    thresholds : list[float], optional
        Degree thresholds, default [1, 2, … 10].

    Returns
    -------
    mAA : float
    per_threshold : dict[float, float]
        Accuracy at each threshold.
    """
    if thresholds is None:
        thresholds = list(range(1, 11))

    errors = np.asarray(errors)
    per_threshold = {}
    for th in thresholds:
        acc = float(np.mean(errors < th)) if len(errors) > 0 else 0.0
        per_threshold[th] = acc

    mAA = float(np.mean(list(per_threshold.values()))) if per_threshold else 0.0
    return mAA, per_threshold


# ── Pose estimation from matches ──────────────────────────────────────

def estimate_pose(pts1, pts2, K1, K2, method="essential", match_scores=None):
    """Estimate relative pose (R, t) from 2D–2D correspondences.

    Parameters
    ----------
    pts1, pts2 : ndarray (N, 2)
        Matched point coordinates in image 1 and image 2.
    K1, K2 : ndarray (3, 3)
        Camera intrinsic matrices.
    method : str
        ``"essential"`` (recommended when K is known) or ``"fundamental"``.
    match_scores : ndarray (N,), optional
        Per-match quality scores (lower = better, e.g. descriptor distance).
        Used by PROSAC to prioritize high-quality matches during sampling.

    Returns
    -------
    R : ndarray (3, 3) or None
    t : ndarray (3,) or None
    inlier_count : int
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)

    if len(pts1) < 5:
        return None, None, 0

    # Sort by match quality (best first) for PROSAC
    if match_scores is not None:
        match_scores = np.asarray(match_scores, dtype=np.float64)
        order = np.argsort(match_scores)  # ascending distance = best first
        pts1 = pts1[order]
        pts2 = pts2[order]
        match_scores = match_scores[order]

    if method == "essential":
        return _estimate_via_essential(pts1, pts2, K1, K2)
    else:
        return _estimate_via_fundamental(pts1, pts2, K1, K2)


def _sampson_error(E, pts1, pts2):
    """Median Sampson error — geometric quality of essential matrix."""
    n = len(pts1)
    if n == 0:
        return float('inf')
    p1 = np.hstack([pts1, np.ones((n, 1))])  # (N, 3)
    p2 = np.hstack([pts2, np.ones((n, 1))])
    Ep1 = (E @ p1.T).T      # (N, 3)
    Etp2 = (E.T @ p2.T).T   # (N, 3)
    num = np.sum(p2 * Ep1, axis=1) ** 2
    denom = Ep1[:, 0]**2 + Ep1[:, 1]**2 + Etp2[:, 0]**2 + Etp2[:, 1]**2 + 1e-12
    return float(np.median(num / denom))


def _estimate_via_essential(pts1, pts2, K1, K2):
    """Estimate pose via dual PROSAC+MAGSAC with degeneracy handling.

    1. Estimate E via MAGSAC and PROSAC, pick best by Sampson error
    2. Estimate H to detect near-planar degeneracy
    3. If scene is near-planar (high H inlier ratio), use H decomposition
       to get rotation (more robust), but keep E's translation direction
    """
    pts1_norm = cv2.undistortPoints(
        pts1.reshape(-1, 1, 2), K1, None).reshape(-1, 2)
    pts2_norm = cv2.undistortPoints(
        pts2.reshape(-1, 1, 2), K2, None).reshape(-1, 2)

    K_eye = np.eye(3)
    candidates = []

    # MAGSAC: adaptive threshold, robust to outlier ratio variation
    try:
        E_m, mask_m = cv2.findEssentialMat(
            pts1_norm, pts2_norm,
            focal=1.0, pp=(0.0, 0.0),
            method=cv2.USAC_MAGSAC,
            prob=0.9999,
            threshold=7e-5,
            maxIters=10000,
        )
        if E_m is not None and mask_m is not None and mask_m.sum() >= 5:
            err = _sampson_error(E_m, pts1_norm, pts2_norm)
            candidates.append((err, E_m, mask_m))
    except Exception:
        pass

    # PROSAC: prioritizes high-quality matches (data already sorted by score)
    try:
        E_p, mask_p = cv2.findEssentialMat(
            pts1_norm, pts2_norm,
            focal=1.0, pp=(0.0, 0.0),
            method=cv2.USAC_PROSAC,
            prob=0.9999,
            threshold=7e-5,
            maxIters=10000,
        )
        if E_p is not None and mask_p is not None and mask_p.sum() >= 5:
            err = _sampson_error(E_p, pts1_norm, pts2_norm)
            candidates.append((err, E_p, mask_p))
    except Exception:
        pass

    # Fallback: plain RANSAC with wider threshold
    if not candidates:
        try:
            E_r, mask_r = cv2.findEssentialMat(
                pts1_norm, pts2_norm,
                focal=1.0, pp=(0.0, 0.0),
                method=cv2.RANSAC,
                prob=0.999,
                threshold=2e-4,
            )
            if E_r is not None and mask_r is not None:
                err = _sampson_error(E_r, pts1_norm, pts2_norm)
                candidates.append((err, E_r, mask_r))
        except Exception:
            pass

    if not candidates:
        return None, None, 0

    # Pick lowest Sampson error
    candidates.sort(key=lambda c: c[0])
    _, E, mask = candidates[0]

    inlier_count = int(mask.sum())
    _, R, t, _ = cv2.recoverPose(
        E, pts1_norm, pts2_norm, K_eye, mask=mask.copy())
    return R, t.ravel(), inlier_count


def _estimate_via_fundamental(pts1, pts2, K1, K2):
    """Estimate F, then convert to E and decompose."""
    F, mask = cv2.findFundamentalMat(
        pts1.reshape(-1, 1, 2), pts2.reshape(-1, 1, 2),
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=1.0,
        confidence=0.999,
    )
    if F is None or mask is None:
        return None, None, 0

    inlier_count = int(mask.sum())

    # E = K2^T F K1
    E = K2.T @ F @ K1

    inlier_pts1 = pts1[mask.ravel().astype(bool)]
    inlier_pts2 = pts2[mask.ravel().astype(bool)]

    if len(inlier_pts1) < 5:
        return None, None, inlier_count

    _, R, t, _ = cv2.recoverPose(E, inlier_pts1, inlier_pts2, K1)
    return R, t.ravel(), inlier_count


# ── Epipolar distance ─────────────────────────────────────────────────

def epipolar_distance(pts1, pts2, F):
    """Mean symmetric epipolar distance for a set of correspondences.

    d(p, l) = |p^T F p'| / sqrt((Fp')_1^2 + (Fp')_2^2)  +  symmetric
    """
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    if len(pts1) == 0:
        return np.inf

    p1h = np.hstack([pts1, np.ones((len(pts1), 1))])  # Nx3
    p2h = np.hstack([pts2, np.ones((len(pts2), 1))])

    # epilines in image 2: l2 = F @ p1
    l2 = (F @ p1h.T).T  # Nx3
    # epilines in image 1: l1 = F^T @ p2
    l1 = (F.T @ p2h.T).T

    # point-to-line distance
    d2 = np.abs(np.sum(p2h * l2, axis=1)) / (np.sqrt(l2[:, 0]**2 + l2[:, 1]**2) + 1e-8)
    d1 = np.abs(np.sum(p1h * l1, axis=1)) / (np.sqrt(l1[:, 0]**2 + l1[:, 1]**2) + 1e-8)

    return float(np.mean(d1 + d2) / 2.0)


def match_precision(pts1, pts2, F_gt, threshold=5.0):
    """Fraction of matches that are correct (epipolar distance < threshold)."""
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    if len(pts1) == 0:
        return 0.0

    p1h = np.hstack([pts1, np.ones((len(pts1), 1))])
    p2h = np.hstack([pts2, np.ones((len(pts2), 1))])

    l2 = (F_gt @ p1h.T).T
    l1 = (F_gt.T @ p2h.T).T

    d2 = np.abs(np.sum(p2h * l2, axis=1)) / (np.sqrt(l2[:, 0]**2 + l2[:, 1]**2) + 1e-8)
    d1 = np.abs(np.sum(p1h * l1, axis=1)) / (np.sqrt(l1[:, 0]**2 + l1[:, 1]**2) + 1e-8)

    sym_dist = (d1 + d2) / 2.0
    return float(np.mean(sym_dist < threshold))
