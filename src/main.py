"""SelMA pipeline entry point.

Runs the two-stage visual place recognition pipeline: coarse retrieval
via DINOv2 patch features followed by RANSAC geometric re-ranking.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import settings
from config.cli import parse_args, apply_overrides
from dataHandlers.dataset import load_image, get_db_images, get_query_images
from dataHandlers.output import create_run_folder, save_match_result
from dataHandlers.visualization import (
    draw_match_lines, draw_ransac_matches, draw_spatial_features,
    save_visualizations, draw_top_candidates, save_top_candidates,
)
from GeometryFuncs.edges import get_edge_patches
from GeometryFuncs.denoise import denoise_patches
from ModelFuncs.feature_extractor import DINOv2Extractor
from ModelFuncs.matcher import match_features, rank_db_images, heuristic_score
from ransac.geometric_filter import RANSACFilter


def process_image(image_path, extractor):
    image = load_image(image_path, color=True)
    patches, points = get_edge_patches(image)

    if len(patches) == 0:
        return None, None, 0

    patches = denoise_patches(patches)
    features = extractor.extract(patches)

    return features, points, len(patches)


def main():
    args = parse_args()
    apply_overrides(args)

    # ── Benchmark mode ────────────────────────────────────────────────
    if getattr(args, "benchmark", False):
        from evaluation.evaluate import BenchmarkEvaluator

        scene = settings.BENCHMARK_SCENE
        if scene is None:
            print("Error: --benchmark-scene is required in benchmark mode.")
            sys.exit(1)

        evaluator = BenchmarkEvaluator(
            scene_path=scene,
            output_dir=settings.OUTPUT_DIR,
            max_pairs=settings.BENCHMARK_MAX_PAIRS,
            pose_method=settings.BENCHMARK_POSE_METHOD,
            descriptor=getattr(args, "descriptor", "sift"),
        )
        evaluator.evaluate()
        return

    # ── Normal matching mode ──────────────────────────────────────────
    t_start = time.time()
    print("Loading DINOv2 model...")
    extractor = DINOv2Extractor()
    ransac = RANSACFilter()

    print("Indexing database images...")
    db_paths = get_db_images()
    print(f"  Found {len(db_paths)} database images")

    db_entries = []
    for i, db_path in enumerate(db_paths):
        if (i + 1) % 100 == 0 or (i + 1) == 1 or (i + 1) == len(db_paths):
            elapsed = time.time() - t_start
            print(f"  [{elapsed:.0f}s] DB image {i + 1}/{len(db_paths)}: {db_path.name}")
        features, points, n_patches = process_image(db_path, extractor)
        if features is not None:
            db_entries.append((db_path, features, points))

    print(f"  Indexed {len(db_entries)} DB images with valid patches")

    # Build lookup dict for O(1) candidate retrieval (replaces linear scan)
    db_lookup = {str(p): (f, pts) for p, f, pts in db_entries}

    print("Processing query images...")
    query_paths = get_query_images()
    print(f"  Found {len(query_paths)} query images")

    run_dir = create_run_folder()
    print(f"  Output folder: {run_dir}")

    for qi, query_path in enumerate(query_paths):
        elapsed = time.time() - t_start
        print(f"\n[{elapsed:.0f}s] Query {qi + 1}/{len(query_paths)}: {query_path.name}")

        query_features, query_points, n_query_patches = process_image(
            query_path, extractor
        )
        if query_features is None:
            print("  No edge patches found, skipping.")
            continue

        # Stage 1: Coarse ranking using top-K discriminative patches
        db_feat_only = [(p, f) for p, f, _ in db_entries]
        rankings = rank_db_images(query_features, db_feat_only)

        if len(rankings) == 0:
            print("  No matches found.")
            continue

        # Stage 2: RANSAC re-ranking on wider shortlist
        shortlist_n = min(settings.SHORTLIST_SIZE, len(rankings))
        candidates = []

        for rank_idx in range(shortlist_n):
            cand_path, avg_sim, top_k_avg = rankings[rank_idx]

            # O(1) dict lookup instead of linear scan
            cand_features, cand_points = db_lookup[str(cand_path)]

            # Use ALL patches for detailed matching on shortlisted candidates
            pairs, _, _ = match_features(
                query_features, cand_features, top_k=len(query_features))

            # RANSAC geometric verification
            ransac_inliers = 0
            ransac_passed = False
            inlier_mask = np.array([])
            if len(pairs) >= settings.RANSAC_MIN_INLIERS:
                q_pts = [query_points[qi_idx] for qi_idx, _, _ in pairs]
                d_pts = [cand_points[di_idx] for _, di_idx, _ in pairs]
                inlier_mask, model, _ = ransac.filter_matches(q_pts, d_pts)
                ransac_inliers = ransac.count_inliers(inlier_mask)
                ransac_passed = ransac.passes_threshold(inlier_mask)

            h_score = heuristic_score(
                top_k_avg, len(pairs), ransac_inliers, ransac_passed)

            candidates.append({
                'path': cand_path,
                'features': cand_features,
                'points': cand_points,
                'pairs': pairs,
                'inlier_mask': inlier_mask,
                'avg_sim': avg_sim,
                'top_k_avg': top_k_avg,
                'ransac_inliers': ransac_inliers,
                'ransac_passed': ransac_passed,
                'heuristic_score': h_score,
            })

        # Re-rank by heuristic score and keep top-K for visualization
        candidates.sort(key=lambda c: c['heuristic_score'], reverse=True)
        candidates = candidates[:settings.TOP_K_VISUALIZE]
        for i, c in enumerate(candidates):
            c['rank'] = i + 1

        best = candidates[0]
        print(f"  Best match: {best['path'].name} "
              f"(H={best['heuristic_score']:.4f}, sim={best['top_k_avg']:.4f}, "
              f"ransac_inliers={best['ransac_inliers']}, "
              f"passed={best['ransac_passed']})")

        # Detailed visualizations for the #1 candidate
        query_img = load_image(query_path, color=True)
        db_img = load_image(best['path'], color=True)

        match_canvas = draw_match_lines(
            query_img, db_img, query_points, best['points'], best['pairs'])
        ransac_canvas = draw_ransac_matches(
            query_img, db_img, query_points, best['points'],
            best['pairs'], best['inlier_mask'])
        spatial_query = draw_spatial_features(
            query_img, query_points, query_features,
            title=f"Query: {query_path.name}")
        spatial_db = draw_spatial_features(
            db_img, best['points'], best['features'],
            title=f"DB: {best['path'].name}")

        save_visualizations(run_dir, query_path.name,
                            match_canvas, ransac_canvas,
                            spatial_query, spatial_db)

        # Top-10 candidates grid visualization
        cand_viz = []
        for c in candidates:
            cand_viz.append({
                'image': load_image(c['path'], color=True),
                'name': c['path'].name,
                'rank': c['rank'],
                'heuristic_score': c['heuristic_score'],
                'top_k_avg': c['top_k_avg'],
                'ransac_inliers': c['ransac_inliers'],
                'ransac_passed': c['ransac_passed'],
            })
        top_canvas = draw_top_candidates(query_img, query_path.name, cand_viz)
        save_top_candidates(run_dir, query_path.name, top_canvas)

        match_data = {
            "num_query_patches": n_query_patches,
            "num_match_patches": best['features'].shape[0] if best['features'] is not None else 0,
            "avg_similarity": f"{best['avg_sim']:.6f}",
            "top_k_avg_similarity": f"{best['top_k_avg']:.6f}",
            "num_matched_pairs": len(best['pairs']),
            "ransac_method": settings.RANSAC_METHOD,
            "ransac_inliers": best['ransac_inliers'],
            "ransac_passed": best['ransac_passed'],
            "heuristic_score": f"{best['heuristic_score']:.6f}",
        }

        save_match_result(run_dir, query_path, best['path'], match_data)

    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.0f}s. Results saved to {run_dir}")


if __name__ == "__main__":
    main()
