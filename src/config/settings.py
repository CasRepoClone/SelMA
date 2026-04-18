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
PATCH_SIZE = 20
PATCH_SPACING = 30
MAX_PATCHES = 500

# ── Denoising ──────────────────────────────────────────────────────────
DENOISE_METHOD = "gaussian"   # "gaussian" (fast) or "nlmeans" (slow, higher quality)
DENOISE_GAUSSIAN_KERNEL = 3
DENOISE_H = 10
DENOISE_TEMPLATE_WINDOW = 7
DENOISE_SEARCH_WINDOW = 21

# ── DINOv2 ─────────────────────────────────────────────────────────────
DINO_MODEL_NAME = "dinov2_vits14"
DINO_INPUT_SIZE = 56  # smallest viable for ViT-S/14 (4×14)
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

# ── RANSAC ─────────────────────────────────────────────────────────────
RANSAC_METHOD = "fundamental"   # "fundamental", "homography", "affine"
RANSAC_REPROJ_THRESH = 5.0
RANSAC_MAX_ITERS = 2000
RANSAC_CONFIDENCE = 0.999
RANSAC_MIN_INLIERS = 8
