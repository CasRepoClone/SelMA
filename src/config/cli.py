import argparse
from pathlib import Path

from config import settings


RANSAC_METHODS = ("fundamental", "homography", "affine")


def parse_args(argv=None):
    """Parse CLI arguments and return the namespace."""
    p = argparse.ArgumentParser(
        description="SelMA — Selective Edge Location Matching & Assessment",
    )

    # ── directories ────────────────────────────────────────────────────
    p.add_argument(
        "-q", "--query-dir",
        type=Path,
        default=None,
        help=f"Query images directory (default: {settings.QUERY_DIR})",
    )
    p.add_argument(
        "-d", "--db-dir",
        type=Path,
        default=None,
        help=f"Database images directory (default: {settings.DB_DIR})",
    )
    p.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help=f"Output root directory (default: {settings.OUTPUT_DIR})",
    )

    # ── RANSAC ─────────────────────────────────────────────────────────
    p.add_argument(
        "--ransac-method",
        choices=RANSAC_METHODS,
        default=None,
        help=f"RANSAC model type (default: {settings.RANSAC_METHOD})",
    )
    p.add_argument(
        "--ransac-reproj",
        type=float,
        default=None,
        help=f"RANSAC reprojection threshold (default: {settings.RANSAC_REPROJ_THRESH})",
    )
    p.add_argument(
        "--ransac-iters",
        type=int,
        default=None,
        help=f"RANSAC max iterations (default: {settings.RANSAC_MAX_ITERS})",
    )
    p.add_argument(
        "--ransac-confidence",
        type=float,
        default=None,
        help=f"RANSAC confidence (default: {settings.RANSAC_CONFIDENCE})",
    )
    p.add_argument(
        "--ransac-min-inliers",
        type=int,
        default=None,
        help=f"Minimum RANSAC inliers (default: {settings.RANSAC_MIN_INLIERS})",
    )

    return p.parse_args(argv)


def apply_overrides(args):
    """Write non-None CLI values back into the settings module."""
    if args.query_dir is not None:
        settings.QUERY_DIR = args.query_dir.resolve()
    if args.db_dir is not None:
        settings.DB_DIR = args.db_dir.resolve()
    if args.output_dir is not None:
        settings.OUTPUT_DIR = args.output_dir.resolve()

    if args.ransac_method is not None:
        settings.RANSAC_METHOD = args.ransac_method
    if args.ransac_reproj is not None:
        settings.RANSAC_REPROJ_THRESH = args.ransac_reproj
    if args.ransac_iters is not None:
        settings.RANSAC_MAX_ITERS = args.ransac_iters
    if args.ransac_confidence is not None:
        settings.RANSAC_CONFIDENCE = args.ransac_confidence
    if args.ransac_min_inliers is not None:
        settings.RANSAC_MIN_INLIERS = args.ransac_min_inliers
