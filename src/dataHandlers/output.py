import csv
import shutil
from datetime import datetime
from pathlib import Path

from config import settings


def create_run_folder():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = settings.OUTPUT_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


CSV_FIELDS = [
    "query_image", "match_image", "num_query_patches",
    "num_match_patches", "avg_similarity", "top_k_avg_similarity",
    "num_matched_pairs", "ransac_method", "ransac_inliers",
    "ransac_passed", "heuristic_score"
]


def save_match_result(run_dir, query_path, match_path, match_data):
    query_path = Path(query_path)
    match_path = Path(match_path)

    shutil.copy2(query_path, run_dir / f"query_{query_path.name}")
    shutil.copy2(match_path, run_dir / f"match_{match_path.name}")

    csv_path = run_dir / "results.csv"
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "query_image": query_path.name,
            "match_image": match_path.name,
            **match_data
        })
