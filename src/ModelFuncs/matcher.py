import numpy as np
import cv2

from config import settings


def cosine_similarity_matrix(features_a, features_b):
    a_norm = features_a / (np.linalg.norm(features_a, axis=1, keepdims=True) + 1e-8)
    b_norm = features_b / (np.linalg.norm(features_b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


def match_features(query_features, db_features, top_k=None,
                   use_mnn=None, ratio_thresh=None, min_score=None):
    """Match query features to database features with robust filtering.

    Implements three complementary filters from the matching literature:
      1. **Minimum score** — discard obviously bad matches (baseline quality)
      2. **Lowe's ratio test** (SIFT/SuperPoint) — reject ambiguous matches
         where the 2nd-best is too close to the best
      3. **Mutual nearest neighbor** (SuperGlue/LightGlue) — keep only
         matches where both sides agree on the correspondence
    """
    top_k = top_k or settings.TOP_K_PATCHES
    use_mnn = use_mnn if use_mnn is not None else settings.MATCH_USE_MNN
    ratio_thresh = (ratio_thresh if ratio_thresh is not None
                    else settings.MATCH_RATIO_THRESH)
    min_score = (min_score if min_score is not None
                 else settings.MATCH_MIN_SCORE)

    if query_features.shape[0] == 0 or db_features.shape[0] == 0:
        return [], 0.0, 0.0

    sim_matrix = cosine_similarity_matrix(query_features, db_features)

    # Forward matching: each query → its best database feature
    best_db_idx = np.argmax(sim_matrix, axis=1)
    best_scores = sim_matrix[np.arange(len(query_features)), best_db_idx]

    avg_similarity = float(np.mean(best_scores))

    # ── Filter 1: minimum cosine similarity ──
    valid = best_scores >= min_score

    # ── Filter 2: Lowe's ratio test ──
    if ratio_thresh < 1.0 and sim_matrix.shape[1] >= 2:
        # Get 2nd-best similarity efficiently via partial sort
        partitioned = np.partition(sim_matrix, -2, axis=1)
        second_best = partitioned[:, -2]
        ratio = second_best / (best_scores + 1e-8)
        valid &= (ratio < ratio_thresh)

    # ── Filter 3: mutual nearest neighbor cross-check ──
    if use_mnn:
        best_query_idx = np.argmax(sim_matrix, axis=0)  # reverse: db → query
        mutual = best_query_idx[best_db_idx] == np.arange(len(query_features))
        valid &= mutual

    # Collect surviving matches, sorted by score
    valid_indices = np.where(valid)[0]
    if len(valid_indices) == 0:
        return [], avg_similarity, 0.0

    scores = best_scores[valid_indices]
    sorted_order = np.argsort(scores)[::-1]
    k = min(top_k, len(sorted_order))
    selected = sorted_order[:k]
    final_indices = valid_indices[selected]

    top_k_avg = float(np.mean(scores[selected]))

    pairs = [(int(qi), int(best_db_idx[qi]), float(best_scores[qi]))
             for qi in final_indices]

    return pairs, avg_similarity, top_k_avg


def _rootsift(desc):
    """RootSIFT normalization (Arandjelovic & Zisserman, CVPR 2012).

    L1-normalize then sqrt — equivalent to Hellinger kernel on L2 distance.
    """
    desc = np.asarray(desc, dtype=np.float32)
    desc = desc / (np.sum(np.abs(desc), axis=1, keepdims=True) + 1e-7)
    desc = np.sqrt(np.clip(desc, 0, None))
    return desc


def match_sift(desc1, desc2, ratio_thresh=0.75, cross_check=False,
               rootsift=True, bidirectional=True):
    """Match SIFT descriptors with Lowe's ratio test.

    Parameters
    ----------
    desc1, desc2 : ndarray (N, 128), (M, 128)
        SIFT descriptors.
    ratio_thresh : float
        Lowe's ratio test threshold (2nd / 1st best distance).
    cross_check : bool
        If True, use OpenCV crossCheck instead of ratio test.
    rootsift : bool
        Apply RootSIFT normalization before matching.
    bidirectional : bool
        Require both forward and backward ratio test to pass.

    Returns
    -------
    pairs : list of (query_idx, db_idx, distance)
    avg_dist : float
    best_dist : float
    """
    if desc1 is None or desc2 is None:
        return [], float('inf'), float('inf')
    if len(desc1) == 0 or len(desc2) == 0:
        return [], float('inf'), float('inf')

    desc1 = np.asarray(desc1, dtype=np.float32)
    desc2 = np.asarray(desc2, dtype=np.float32)

    # RootSIFT: more discriminative descriptor space
    if rootsift:
        desc1 = _rootsift(desc1)
        desc2 = _rootsift(desc2)

    if cross_check:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda m: m.distance)
        pairs = [(m.queryIdx, m.trainIdx, m.distance) for m in matches]
    else:
        # Use FLANN for large feature sets, BF for small
        if len(desc1) > 500 or len(desc2) > 500:
            index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
            search_params = dict(checks=100)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2)

        raw_fwd = matcher.knnMatch(desc1, desc2, k=2)

        # Forward ratio test
        pairs = []
        for match_pair in raw_fwd:
            if len(match_pair) < 2:
                continue
            m, n = match_pair
            if m.distance < ratio_thresh * n.distance:
                pairs.append((m.queryIdx, m.trainIdx, m.distance))

        # Bidirectional: also require reverse ratio test
        if bidirectional and len(pairs) > 0:
            raw_bwd = matcher.knnMatch(desc2, desc1, k=2)
            bwd_pass = set()
            for match_pair in raw_bwd:
                if len(match_pair) < 2:
                    continue
                m, n = match_pair
                if m.distance < ratio_thresh * n.distance:
                    bwd_pass.add((m.trainIdx, m.queryIdx))
            # Keep only mutually-confirmed matches
            pairs = [(qi, di, d) for qi, di, d in pairs
                     if (qi, di) in bwd_pass]

    if not pairs:
        return [], float('inf'), float('inf')

    dists = [d for _, _, d in pairs]
    return pairs, float(np.mean(dists)), float(np.min(dists))


def rank_db_images(query_features, db_entries, top_k_patches=None):
    top_k_patches = top_k_patches or settings.TOP_K_PATCHES
    scores = []
    for db_path, db_features in db_entries:
        _, avg_sim, top_k_avg = match_features(
            query_features, db_features, top_k=top_k_patches
        )
        scores.append((db_path, avg_sim, top_k_avg))

    scores.sort(key=lambda x: x[2], reverse=True)
    return scores


def heuristic_score(top_k_avg, num_pairs, ransac_inliers, ransac_passed):
    """Compute a combined heuristic score from similarity and geometry.

    Score = w_sim * top_k_avg
          + w_inlier * (ransac_inliers / max(num_pairs, 1))
          + w_pass  * ransac_passed
    """
    w_sim = settings.HEURISTIC_W_SIM
    w_inlier = settings.HEURISTIC_W_INLIER_RATIO
    w_pass = settings.HEURISTIC_W_RANSAC_PASS

    inlier_ratio = ransac_inliers / max(num_pairs, 1)
    return (w_sim * top_k_avg
            + w_inlier * inlier_ratio
            + w_pass * float(ransac_passed))
