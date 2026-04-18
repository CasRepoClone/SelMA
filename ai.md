# AI & Machine Learning Implementation Details

This document describes every AI and ML component used in the SelMA (Selective Edge Location Matching & Assessment) pipeline, including architectural decisions, implementation specifics, and inference details.

---

## 1. DINOv2 — Self-Supervised Visual Feature Extraction

### Model Selection

- **Model**: `dinov2_vits14` (ViT-Small with patch size 14)
- **Parameters**: ~21 million
- **Embedding dimension**: 384
- **Source**: `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")`
- **Training data**: LVD-142M (142 million curated images)
- **Training objective**: Self-distillation (student-teacher framework without labels)

### Why DINOv2

DINOv2 was chosen because:
1. **No task-specific fine-tuning required** — frozen features generalize across domains.
2. **Robust to illumination changes** — critical for day-night matching.
3. **Spatial awareness** — the `[CLS]` token aggregates information from all patch tokens via self-attention, encoding both local texture and global layout.
4. **Computational efficiency** — ViT-S/14 is the smallest DINOv2 variant, enabling fast inference on edge patches.

### Implementation

**File**: `src/ModelFuncs/feature_extractor.py`

```python
class DINOv2Extractor:
    def __init__(self, model_name=None, device=None):
        # Loads pretrained model via torch.hub
        # Auto-detects CUDA availability
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(self.device)
```

**Preprocessing pipeline** (per patch):
1. Convert NumPy array → PIL Image via `ToPILImage()`
2. Resize from 20×20 → 56×56 (bilinear interpolation)
3. Convert to tensor, scale to [0, 1]
4. Normalize: $\hat{x}_c = (x_c - \mu_c) / \sigma_c$ with ImageNet statistics

**Batched inference**:
- Patches are processed in batches of 128 (configurable via `DINO_BATCH_SIZE`)
- `torch.no_grad()` disables gradient computation for inference
- GPU tensors are moved back to CPU after each batch

**Grayscale handling**: Single-channel patches are converted to 3-channel by stacking: `np.stack([patch] * 3, axis=-1)`

**Output**: Each patch produces a 384-dimensional float32 feature vector (the `[CLS]` token from the final transformer layer).

---

## 2. Feature Matching — Cosine Similarity

### Implementation

**File**: `src/ModelFuncs/matcher.py`

**Similarity computation**: Given query features $\mathbf{Q} \in \mathbb{R}^{m \times 384}$ and database features $\mathbf{D} \in \mathbb{R}^{n \times 384}$:

```python
def cosine_similarity_matrix(features_a, features_b):
    a_norm = features_a / (np.linalg.norm(features_a, axis=1, keepdims=True) + 1e-8)
    b_norm = features_b / (np.linalg.norm(features_b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T
```

- L2-normalizes each feature vector (with epsilon 1e-8 for numerical stability)
- Computes the full $m \times n$ similarity matrix via matrix multiplication
- Values range from -1 (opposite) to +1 (identical)

**Nearest-neighbour matching**: Each query patch is matched to its highest-similarity database patch:
```python
best_db_idx = np.argmax(sim_matrix, axis=1)
best_scores = sim_matrix[np.arange(len(best_db_idx)), best_db_idx]
```

**Ranking**:
- For each database image, compute the top-50 patch matches' average similarity
- Rank all database images by this score (descending)
- The best-ranked image is selected as the match

### Why Cosine Similarity

Cosine similarity measures angular distance between feature vectors, making it invariant to feature magnitude. This is important because DINOv2 features may have varying norms across patches with different content.

---

## 3. RANSAC Geometric Verification

### Implementation

**File**: `src/ransac/geometric_filter.py`

```python
class RANSACFilter:
    def filter_matches(self, query_points, db_points):
        # Dispatches to appropriate geometric model
        if self.method == "fundamental":
            model, mask = self._fit_fundamental(query_pts, db_pts)
        elif self.method == "homography":
            model, mask = self._fit_homography(query_pts, db_pts)
        elif self.method == "affine":
            model, mask = self._fit_affine(query_pts, db_pts)
```

### Supported Geometric Models

#### Fundamental Matrix (default)
- **OpenCV function**: `cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_RANSAC)`
- **Degrees of freedom**: 7
- **Minimum sample size**: 7 point correspondences
- **Applicability**: General two-view geometry (any 3D scene structure)
- Encodes the epipolar constraint: matched points must lie on corresponding epipolar lines

#### Homography
- **OpenCV function**: `cv2.findHomography(pts1, pts2, method=cv2.RANSAC)`
- **Degrees of freedom**: 8
- **Minimum sample size**: 4 point correspondences
- **Applicability**: Planar scenes or pure rotation between views

#### Affine Transform
- **OpenCV function**: `cv2.estimateAffine2D(pts1, pts2, method=cv2.RANSAC)`
- **Degrees of freedom**: 6
- **Minimum sample size**: 3 point correspondences
- **Applicability**: Weak perspective, small viewpoint changes

### Parameters

| Parameter | Value | Rationale |
|---|---|---|
| `RANSAC_REPROJ_THRESH` | 5.0 px | Allows for localization error in edge-based patches |
| `RANSAC_MAX_ITERS` | 2000 | Sufficient for high inlier ratios; caps worst-case runtime |
| `RANSAC_CONFIDENCE` | 0.999 | High confidence that the best model is found |
| `RANSAC_MIN_INLIERS` | 8 | Requires at least 8 geometrically consistent matches |

### Verification Logic

A match "passes" RANSAC if the number of inliers (correspondences consistent with the estimated model) exceeds `RANSAC_MIN_INLIERS`. This confirms the query and database image share a genuine geometric relationship, filtering out false positives from visual aliasing.

---

## 4. Edge-Based Patch Sampling

### Implementation

**File**: `src/GeometryFuncs/edges.py`

This is not an AI method per se, but it serves as the feature-point detector that replaces traditional keypoint detectors (SIFT, ORB, SuperPoint).

**Rationale**: Canny edges capture structural boundaries (buildings, signs, road markings) that are more robust to illumination changes than texture-based keypoints. By sampling patches along edges, we focus DINOv2's attention on geometrically meaningful image regions.

**Process**:
1. Convert to grayscale
2. Apply Canny with thresholds (50, 150)
3. Collect all edge pixels as (x, y) coordinates
4. Subsample at spacing=30 pixels
5. Extract 20×20 colour patches centred on each edge point
6. Cap at 500 patches via uniform subsampling (linspace)

### Design Choices

- **20×20 patches**: Small enough to capture local structure, large enough for DINOv2's 14×14 patch tokenizer to have meaningful content after resizing to 56×56.
- **Spacing of 30**: Prevents redundant overlapping patches while maintaining coverage.
- **500 patch limit**: Balances feature richness against memory and compute (500 × 384 floats = 192KB per image).

---

## 5. Image Denoising

### Implementation

**File**: `src/GeometryFuncs/denoise.py`

Two methods are implemented:

#### Gaussian Blur (default)
```python
cv2.GaussianBlur(patch, (3, 3), 0)
```
- Fast (~0.01s for 500 patches)
- Reduces high-frequency sensor noise
- Minimal impact on edge structure at kernel size 3
- Includes try/except fallback for rare OpenCV C++ exceptions on certain pixel patterns

#### Non-Local Means (optional)
```python
cv2.fastNlMeansDenoisingColored(patch, None, h, h, template_window, search_window)
```
- Higher quality denoising by averaging similar patches within a search window
- Much slower (~9.7s for 500 patches vs. 0.01s for Gaussian)
- Configurable via `DENOISE_METHOD = "nlmeans"` in settings

### Why Denoise Before DINOv2

Sensor noise in small 20×20 patches can dominate the signal, especially in nighttime images. Denoising ensures DINOv2 encodes structural content rather than noise artefacts. Gaussian blur is preferred for its speed with minimal information loss at the 3×3 kernel scale.

---

## 6. End-to-End Inference Pipeline

### Implementation

**File**: `src/main.py`

**Phase 1 — Database Indexing**:
- Load all 4,479 database images
- For each: detect edges → extract patches → denoise → DINOv2 encode
- Store (path, features, points) tuples in memory

**Phase 2 — Query Matching**:
- For each of 947 query images:
  1. Process identically to DB images
  2. Compute cosine similarity against all DB image features
  3. Rank by top-50 average similarity
  4. Apply RANSAC on the best match's point correspondences
  5. Save query image, matched DB image, and metadata to output folder

**Output format**: CSV with columns:
- `query_image`, `match_image`
- `num_query_patches`, `num_match_patches`
- `avg_similarity`, `top_k_avg_similarity`
- `num_matched_pairs`
- `ransac_method`, `ransac_inliers`, `ransac_passed`

### Performance Characteristics

| Stage | Time per image | Notes |
|---|---|---|
| Edge detection + patch extraction | ~0.03s | CPU-bound, OpenCV |
| Gaussian denoising (500 patches) | ~0.01s | CPU-bound, OpenCV |
| DINOv2 feature extraction (500 patches) | ~0.37s | GPU-bound, batched |
| Cosine similarity (500 × 500) | ~0.001s | NumPy matrix multiply |
| RANSAC verification | ~0.01s | CPU-bound, OpenCV |

---

## 7. Technology Stack

| Component | Technology | Version |
|---|---|---|
| Deep Learning Framework | PyTorch | 2.11.0+cu128 |
| Vision Transforms | torchvision | 0.26.0+cu128 |
| Computer Vision | OpenCV | 4.13.0 |
| Numerical Computing | NumPy | 2.4.3 |
| Image I/O | Pillow | 12.1.1 |
| GPU | CUDA | 12.8 compatible |
| Model | DINOv2 ViT-S/14 | via torch.hub |

---

## 8. Design Decisions Summary

| Decision | Chosen | Alternative | Rationale |
|---|---|---|---|
| Keypoint detector | Canny edges | SIFT, ORB, SuperPoint | Illumination-robust structural features |
| Feature encoder | DINOv2 ViT-S/14 | ResNet, NetVLAD, SuperGlue | Self-supervised, no fine-tuning needed |
| Matching strategy | Cosine similarity + top-k | Mutual NN, ratio test | Simple, effective for dense patch sets |
| Geometric verification | RANSAC | MAGSAC, GC-RANSAC | Standard, well-understood, OpenCV native |
| Denoising | Gaussian blur | NLMeans, BM3D | 1000× faster, sufficient for patch preprocessing |
| Input size | 56×56 | 224×224 | Minimal ViT grid (4×4 tokens), fast inference |
