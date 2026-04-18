import numpy as np

from config import settings


def cosine_similarity_matrix(features_a, features_b):
    a_norm = features_a / (np.linalg.norm(features_a, axis=1, keepdims=True) + 1e-8)
    b_norm = features_b / (np.linalg.norm(features_b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


def match_features(query_features, db_features, top_k=None):
    top_k = top_k or settings.TOP_K_PATCHES

    if query_features.shape[0] == 0 or db_features.shape[0] == 0:
        return [], 0.0, 0.0

    sim_matrix = cosine_similarity_matrix(query_features, db_features)

    best_db_idx = np.argmax(sim_matrix, axis=1)
    best_scores = sim_matrix[np.arange(len(best_db_idx)), best_db_idx]

    avg_similarity = float(np.mean(best_scores))

    k = min(top_k, len(best_scores))
    top_indices = np.argsort(best_scores)[::-1][:k]
    top_k_avg = float(np.mean(best_scores[top_indices]))

    pairs = [(int(qi), int(best_db_idx[qi]), float(best_scores[qi]))
             for qi in top_indices]

    return pairs, avg_similarity, top_k_avg


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
