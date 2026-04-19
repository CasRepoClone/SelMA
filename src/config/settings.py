"""Default configuration parameters for the SelMA pipeline.

All values can be overridden via CLI flags (see cli.py) or by modifying
this module at runtime before pipeline execution.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
IMAGES_DIR = BASE_DIR / "images_upright"
DB_DIR = IMAGES_DIR / "db"
QUERY_DIR = IMAGES_DIR / "query"
OUTPUT_DIR = BASE_DIR / "output"

# ── Image extensions ───────────────────────────────────────────────────
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# ── Edge detection ─────────────────────────────────────────────────────
CANNY_LOW_THRESH = 50
CANNY_HIGH_THRESH = 150

# ── Patch extraction ───────────────────────────────────────────────────
PATCH_SIZE = 64                   # larger patches for more context
PATCH_SPACING = 15                # denser fallback sampling
MAX_PATCHES = 2000                # need volume for MNN filtering

# ── Denoising ──────────────────────────────────────────────────────────
DENOISE_METHOD = "gaussian"   # "gaussian" (fast) or "nlmeans" (slow, higher quality)
DENOISE_GAUSSIAN_KERNEL = 3
DENOISE_H = 10
DENOISE_TEMPLATE_WINDOW = 7
DENOISE_SEARCH_WINDOW = 21

# ── DINOv2 ─────────────────────────────────────────────────────────────
DINO_MODEL_NAME = "dinov2_vits14"
DINO_INPUT_SIZE = 112  # 8×14 tokens — good speed/quality balance
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]
DINO_BATCH_SIZE = 128
DEVICE = None  # auto-detect: "cuda" if available else "cpu"

# ── Matching ───────────────────────────────────────────────────────────
TOP_K_PATCHES = 50              # top-K patches for coarse ranking (discriminative)
SHORTLIST_SIZE = 100            # how many DB candidates survive to RANSAC re-ranking
TOP_K_VISUALIZE = 10            # how many top DB candidates to visualize per query

# ── Heuristic scoring weights ──────────────────────────────────────
HEURISTIC_W_SIM = 0.5          # weight for top-k avg cosine similarity
HEURISTIC_W_INLIER_RATIO = 0.35  # weight for RANSAC inlier ratio
HEURISTIC_W_RANSAC_PASS = 0.15   # bonus when RANSAC passes threshold

# ── Keypoint detection (Shi-Tomasi corners near edges) ─────────────────
USE_EDGE_KEYPOINTS = True         # repeatable keypoints vs uniform sampling
KEYPOINT_MAX_CORNERS = 3000
KEYPOINT_QUALITY_LEVEL = 0.01
KEYPOINT_MIN_DISTANCE = 8
KEYPOINT_EDGE_PROXIMITY = 15      # max pixel distance from nearest edge
SIFT_CONTRAST_THRESH = 0.02       # lower = more features detected
SIFT_RATIO_THRESH = 0.75          # Lowe's ratio test for SIFT matching

# ── Robust matching filters ───────────────────────────────────────────
MATCH_USE_MNN = True              # mutual nearest neighbor (primary filter)
MATCH_RATIO_THRESH = 1.0          # disabled — ratio test kills edge features
MATCH_MIN_SCORE = 0.15            # minimum cosine similarity

# ── RANSAC ─────────────────────────────────────────────────────────────
RANSAC_METHOD = "fundamental"   # "fundamental", "homography", "affine"
RANSAC_REPROJ_THRESH = 5.0
RANSAC_MAX_ITERS = 2000
RANSAC_CONFIDENCE = 0.999
RANSAC_MIN_INLIERS = 8

# ── Benchmark evaluation ──────────────────────────────────────────────
BENCHMARK_SCENE = None          # path to phototourism scene directory
BENCHMARK_MAX_PAIRS = None      # limit pairs (None = all)
BENCHMARK_POSE_METHOD = "essential"  # "essential" or "fundamental"
