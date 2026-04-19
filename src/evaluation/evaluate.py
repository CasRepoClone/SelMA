"""Benchmark evaluation runner.

Runs SelMA on image pairs from a phototourism-style dataset, compares
estimated poses against ground truth, and reports standard metrics.

Usage (standalone)::

    python -m evaluation.evaluate --scene data/phototourism/sacre_coeur

Or via the main entry point::

    python main.py --benchmark --benchmark-scene data/phototourism/sacre_coeur
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import cv2

from evaluation.dataset import BenchmarkScene
from evaluation.metrics import (
    estimate_pose,
    pose_error,
    compute_mAA,
    epipolar_distance,
    match_precision,
)
from config import settings
from dataHandlers.dataset import load_image
from dataHandlers.visualization import draw_top_candidates
from GeometryFuncs.edges import get_edge_patches, detect_edges, detect_edge_keypoints
from GeometryFuncs.denoise import denoise_patches
from ModelFuncs.feature_extractor import DINOv2Extractor, SIFTAtEdgeKeypoints
from ModelFuncs.matcher import match_features, match_sift
from ModelFuncs.match_filter import filter_matches


class BenchmarkEvaluator:
    """Evaluate SelMA matching quality against ground truth poses."""

    def __init__(self, scene_path, output_dir=None, max_pairs=None,
                 pose_method="essential", descriptor="sift"):
        self.scene = BenchmarkScene(scene_path)
        self.output_dir = Path(output_dir or settings.OUTPUT_DIR) / "benchmark"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_pairs = max_pairs
        self.pose_method = pose_method
        self.descriptor = descriptor  # "sift" or "dinov2"

        if descriptor == "dinov2":
            self.extractor = DINOv2Extractor()
        else:
            self.sift_extractor = SIFTAtEdgeKeypoints(
                contrast_thresh=settings.SIFT_CONTRAST_THRESH,
            )

        self._feature_cache = {}
        self._image_cache = {}

        # Visualization output dirs
        self.kpmatches_dir = self.output_dir / "kpmatches"
        self.topk_dir = self.output_dir / "topk"
        self.kpmatches_dir.mkdir(parents=True, exist_ok=True)
        self.topk_dir.mkdir(parents=True, exist_ok=True)

        self._kp_images_saved = set()  # track which images already have keypoint vis

    # ── feature extraction (cached) ────────────────────────────────────

    def _get_features(self, img_name):
        if img_name in self._feature_cache:
            return self._feature_cache[img_name]

        path = self.scene.get_image_path(img_name)
        image = load_image(path, color=True)
        self._image_cache[img_name] = image

        if self.descriptor == "sift":
            # SelMA pipeline: SIFT detection filtered to edge locations
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            edge_map = detect_edges(gray)

            descriptors, valid_points = self.sift_extractor.detect_and_compute(
                image, edge_map
            )

            if descriptors is None or len(valid_points) == 0:
                self._feature_cache[img_name] = (None, None)
                return None, None

            self._feature_cache[img_name] = (descriptors, valid_points)

            # Save keypoint overlay image once per image
            if img_name not in self._kp_images_saved and valid_points is not None:
                self._save_keypoint_image(img_name, image, valid_points)

            return descriptors, valid_points
        else:
            # DINOv2 pipeline (original)
            patches, points = get_edge_patches(image)

            if len(patches) == 0:
                self._feature_cache[img_name] = (None, None)
                return None, None

            patches = denoise_patches(patches)
            features = self.extractor.extract(patches)
            self._feature_cache[img_name] = (features, points)

            if img_name not in self._kp_images_saved and len(points) > 0:
                self._save_keypoint_image(img_name, image, points)

            return features, points

    # ── evaluate a single pair ─────────────────────────────────────────

    def _evaluate_pair(self, img1_name, img2_name):
        feat1, pts1 = self._get_features(img1_name)
        feat2, pts2 = self._get_features(img2_name)

        result = {
            "img1": img1_name,
            "img2": img2_name,
            "num_features_1": 0,
            "num_features_2": 0,
            "num_matches": 0,
            "inlier_ratio": 0.0,
            "pose_error": 180.0,
            "rotation_error": 180.0,
            "translation_error": 180.0,
            "epipolar_dist": float("inf"),
            "match_precision": 0.0,
            "status": "ok",
        }

        if feat1 is None or feat2 is None:
            result["status"] = "no_features"
            return result

        result["num_features_1"] = len(feat1)
        result["num_features_2"] = len(feat2)

        # Match features
        if self.descriptor == "sift":
            pairs, _, _ = match_sift(
                feat1, feat2, ratio_thresh=settings.SIFT_RATIO_THRESH,
            )
        else:
            pairs, avg_sim, top_k_avg = match_features(
                feat1, feat2, top_k=len(feat1)
            )
        result["num_matches"] = len(pairs)

        if len(pairs) < 5:
            result["status"] = "too_few_matches"
            return result

        # Extract matched coordinates and scores
        matched_pts1 = np.array([pts1[qi] for qi, _, _ in pairs], dtype=np.float64)
        matched_pts2 = np.array([pts2[di] for _, di, _ in pairs], dtype=np.float64)
        match_scores = np.array([d for _, _, d in pairs], dtype=np.float64)

        # Ground truth
        K1 = self.scene.get_intrinsics(img1_name)
        K2 = self.scene.get_intrinsics(img2_name)
        R_gt, t_gt = self.scene.get_relative_pose(img1_name, img2_name)
        F_gt = self.scene.get_fundamental_matrix(img1_name, img2_name)

        # Match quality vs ground truth fundamental matrix
        result["epipolar_dist"] = epipolar_distance(matched_pts1, matched_pts2, F_gt)
        result["match_precision"] = match_precision(matched_pts1, matched_pts2, F_gt)

        # Estimate pose from matches (scores enable PROSAC sampling)
        R_est, t_est, n_inliers = estimate_pose(
            matched_pts1, matched_pts2, K1, K2,
            method=self.pose_method, match_scores=match_scores,
        )

        result["inlier_ratio"] = n_inliers / max(len(pairs), 1)

        if R_est is not None and t_est is not None:
            from evaluation.metrics import rotation_error, translation_error
            result["rotation_error"] = rotation_error(R_est, R_gt)
            result["translation_error"] = translation_error(t_est, t_gt)
            result["pose_error"] = max(result["rotation_error"],
                                       result["translation_error"])
        else:
            result["status"] = "pose_estimation_failed"

        return result

    def _save_keypoint_image(self, img_name, image, points):
        """Draw all postprocessed keypoints on the image and save to kpmatches/."""
        canvas = image.copy()
        if len(canvas.shape) == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        for pt in points:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(canvas, (x, y), 3, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 3, (0, 0, 0), 1, cv2.LINE_AA)
        # Label
        label = f"{len(points)} keypoints"
        cv2.putText(canvas, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
        stem = Path(img_name).stem
        cv2.imwrite(str(self.kpmatches_dir / f"{stem}.jpg"), canvas)
        self._kp_images_saved.add(img_name)

    def _save_topk_grids(self, results, top_k=5):
        """For each cached image, match against all others and show top-k."""
        # Gather all images that have cached features
        img_names = [n for n, (f, p) in self._feature_cache.items()
                     if f is not None and p is not None]
        if len(img_names) < 2:
            return

        print(f"  Matching {len(img_names)} images all-vs-all for top-{top_k} grids ...")

        for qi, q_name in enumerate(img_names):
            query_img = self._image_cache.get(q_name)
            if query_img is None:
                continue
            q_feat, q_pts = self._feature_cache[q_name]

            # Match against every other image, collecting real scores
            match_info = []
            for di, d_name in enumerate(img_names):
                if di == qi:
                    continue
                d_feat, d_pts = self._feature_cache[d_name]

                if self.descriptor == "sift":
                    pairs, avg_dist, best_dist = match_sift(
                        q_feat, d_feat,
                        ratio_thresh=settings.SIFT_RATIO_THRESH,
                    )
                else:
                    pairs, avg_dist, best_dist = match_features(
                        q_feat, d_feat, top_k=len(q_feat),
                    )

                n_matches = len(pairs)
                # Compute inlier ratio via fundamental matrix
                inlier_ratio = 0.0
                if n_matches >= 8:
                    mpts1 = np.array([q_pts[qi_] for qi_, _, _ in pairs], dtype=np.float64)
                    mpts2 = np.array([d_pts[di_] for _, di_, _ in pairs], dtype=np.float64)
                    _, fmask = cv2.findFundamentalMat(
                        mpts1.reshape(-1, 1, 2), mpts2.reshape(-1, 1, 2),
                        cv2.FM_RANSAC, 3.0, 0.999,
                    )
                    if fmask is not None:
                        inlier_ratio = float(fmask.ravel().sum()) / n_matches

                # Similarity: invert avg_dist so lower distance = higher score
                sim_score = 1.0 / (1.0 + avg_dist) if n_matches > 0 else 0.0

                match_info.append((d_name, n_matches, inlier_ratio, sim_score))

            # Sort by match count descending, take top_k
            match_info.sort(key=lambda x: x[1], reverse=True)
            top = match_info[:top_k]

            candidates = []
            for rank, (d_name, n_matches, inlier_ratio, sim_score) in enumerate(top, 1):
                d_img = self._image_cache.get(d_name)
                if d_img is None:
                    continue
                candidates.append({
                    "image": d_img,
                    "name": d_name,
                    "rank": rank,
                    "heuristic_score": inlier_ratio,
                    "top_k_avg": sim_score,
                    "ransac_inliers": n_matches,
                    "ransac_passed": inlier_ratio > 0.3,
                })

            if not candidates:
                continue

            canvas = draw_top_candidates(query_img, q_name, candidates)
            stem = Path(q_name).stem
            cv2.imwrite(str(self.topk_dir / f"{stem}.jpg"), canvas)

            if (qi + 1) % 10 == 0 or (qi + 1) == len(img_names):
                print(f"    {qi+1}/{len(img_names)} images done")

    # ── main evaluation loop ───────────────────────────────────────────

    def evaluate(self):
        """Run full benchmark and return aggregated results."""
        pairs = self.scene.get_pairs(max_pairs=self.max_pairs)
        print(f"\n{'='*60}")
        print(f"  SelMA Benchmark Evaluation")
        print(f"  Scene : {self.scene.scene_path.name}")
        print(f"  Images: {len(self.scene.image_names)}")
        print(f"  Pairs : {len(pairs)}")
        print(f"{'='*60}\n")

        results = []
        t_start = time.time()

        for i, (n1, n2) in enumerate(pairs):
            elapsed = time.time() - t_start
            if (i + 1) % 10 == 0 or i == 0 or (i + 1) == len(pairs):
                print(f"  [{elapsed:6.0f}s] Pair {i+1:4d}/{len(pairs)} : "
                      f"{n1} ↔ {n2}")

            result = self._evaluate_pair(n1, n2)
            results.append(result)

            # Progress peek
            if (i + 1) % 50 == 0:
                errors_so_far = [r["pose_error"] for r in results if r["status"] == "ok"]
                if errors_so_far:
                    maa_so_far, _ = compute_mAA(errors_so_far)
                    print(f"         interim mAA = {maa_so_far:.4f}  "
                          f"({len(errors_so_far)} valid pairs)")

        total_time = time.time() - t_start

        # Aggregate
        summary = self._aggregate(results, total_time)
        self._print_summary(summary)
        self._save_results(results, summary)

        # Top-5 matching image grids (across images involved in benchmark pairs)
        self._save_topk_grids(results, top_k=5)

        return summary

    # ── aggregation ────────────────────────────────────────────────────

    @staticmethod
    def _aggregate(results, total_time):
        ok_results = [r for r in results if r["status"] == "ok"]
        pose_errors = [r["pose_error"] for r in ok_results]
        rot_errors = [r["rotation_error"] for r in ok_results]
        trans_errors = [r["translation_error"] for r in ok_results]

        mAA, per_threshold = compute_mAA(pose_errors)

        return {
            "total_pairs": len(results),
            "valid_pairs": len(ok_results),
            "failed_pairs": len(results) - len(ok_results),
            "mAA": mAA,
            "mAA_per_threshold": per_threshold,
            "mean_pose_error": float(np.mean(pose_errors)) if pose_errors else 180.0,
            "median_pose_error": float(np.median(pose_errors)) if pose_errors else 180.0,
            "mean_rotation_error": float(np.mean(rot_errors)) if rot_errors else 180.0,
            "mean_translation_error": float(np.mean(trans_errors)) if trans_errors else 180.0,
            "mean_matches": float(np.mean([r["num_matches"] for r in results])),
            "mean_inlier_ratio": float(np.mean([r["inlier_ratio"] for r in ok_results])) if ok_results else 0.0,
            "mean_match_precision": float(np.mean([r["match_precision"] for r in ok_results])) if ok_results else 0.0,
            "total_time_s": total_time,
        }

    # ── reporting ──────────────────────────────────────────────────────

    @staticmethod
    def _print_summary(s):
        print(f"\n{'='*60}")
        print(f"  BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"  Pairs evaluated      : {s['total_pairs']}")
        print(f"  Valid / failed        : {s['valid_pairs']} / {s['failed_pairs']}")
        print(f"  ──────────────────────────────────────────")
        print(f"  mAA (1°–10°)         : {s['mAA']:.4f}")
        for th, acc in sorted(s["mAA_per_threshold"].items()):
            print(f"    Accuracy @{th:2.0f}°      : {acc:.4f}")
        print(f"  ──────────────────────────────────────────")
        print(f"  Mean pose error      : {s['mean_pose_error']:.2f}°")
        print(f"  Median pose error    : {s['median_pose_error']:.2f}°")
        print(f"  Mean rotation error  : {s['mean_rotation_error']:.2f}°")
        print(f"  Mean translation err : {s['mean_translation_error']:.2f}°")
        print(f"  ──────────────────────────────────────────")
        print(f"  Mean matches/pair    : {s['mean_matches']:.1f}")
        print(f"  Mean inlier ratio    : {s['mean_inlier_ratio']:.4f}")
        print(f"  Mean match precision : {s['mean_match_precision']:.4f}")
        print(f"  ──────────────────────────────────────────")
        print(f"  Total time           : {s['total_time_s']:.0f}s")
        print(f"{'='*60}\n")

    def _save_results(self, results, summary):
        # Per-pair CSV
        csv_path = self.output_dir / f"benchmark_{self.scene.scene_path.name}.csv"
        fieldnames = [
            "img1", "img2", "num_features_1", "num_features_2",
            "num_matches", "inlier_ratio", "pose_error",
            "rotation_error", "translation_error",
            "epipolar_dist", "match_precision", "status",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k])
                                 for k in fieldnames})

        # Summary text
        txt_path = self.output_dir / f"summary_{self.scene.scene_path.name}.txt"
        with open(txt_path, "w") as f:
            f.write(f"SelMA Benchmark — {self.scene.scene_path.name}\n")
            f.write(f"{'='*50}\n")
            for k, v in summary.items():
                if k == "mAA_per_threshold":
                    for th, acc in sorted(v.items()):
                        f.write(f"  accuracy@{th:.0f}deg = {acc:.4f}\n")
                else:
                    f.write(f"  {k} = {v}\n")

        print(f"  Results saved to {csv_path}")
        print(f"  Summary saved to {txt_path}")
        print(f"  Keypoint images in  {self.kpmatches_dir}")
        print(f"  Top-k image grids in {self.topk_dir}")


# ── standalone entry point ─────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser(description="SelMA benchmark evaluation")
    p.add_argument("--scene", type=Path, required=True,
                   help="Path to benchmark scene directory")
    p.add_argument("--output", type=Path, default=None,
                   help="Output directory")
    p.add_argument("--max-pairs", type=int, default=None,
                   help="Limit number of evaluated pairs")
    p.add_argument("--pose-method", choices=["essential", "fundamental"],
                   default="essential",
                   help="Pose estimation method (default: essential)")
    args = p.parse_args()

    evaluator = BenchmarkEvaluator(
        scene_path=args.scene,
        output_dir=args.output,
        max_pairs=args.max_pairs,
        pose_method=args.pose_method,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()
