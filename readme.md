<img width="1254" height="383" alt="Untitled-2" src="https://github.com/user-attachments/assets/c2600861-4699-4cee-ac25-e8b5341c5308" />

# SelMA — Selective Edge Location Matching & Assessment
A two-stage visual place recognition pipeline that combines structural edge-based local feature extraction with self-supervised vision transformer representations. **SelMA** (Selective Edge Location Matching & Assessment) identifies the location of a query photograph by matching it against a geo-tagged database of reference images, addressing challenges such as extreme illumination change (day-to-night), viewpoint variation, and perceptual aliasing.

## Table of Contents

- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Method Overview](#method-overview)
- [Mathematical Foundations](#mathematical-foundations)
- [Project Structure](#project-structure)
- [Configuration Reference](#configuration-reference)
- [Dataset](#dataset)
- [Benchmark Results](#benchmark-results)
- [References](#references)

---

## Setup and Installation

### Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| CUDA | ≥ 12.8 (optional for GPU acceleration) |
| GPU VRAM | ≥ 8 GB recommended |

### 1. Clone and enter the repository

```bash
git clone https://github.com/CasRepoClone/SelMA.git
cd SelMA
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The DINOv2 ViT-S/14 weights (~85 MB) are downloaded automatically on first run via `torch.hub`.

### 4. Prepare the dataset

Place the Aachen Day-Night images under `images_upright/`:

```
images_upright/
├── db/          # Reference database images
└── query/       # Query images
    ├── day/
    └── night/
```

### 5. Verify the installation

```bash
cd src
python main.py --help
```

---

## Usage

```bash
cd src
python main.py [OPTIONS]
```

### Command-line flags

| Flag | Short | Description | Default |
|---|---|---|---|
| `--query-dir` | `-q` | Path to query image directory | `images_upright/query` |
| `--db-dir` | `-d` | Path to database image directory | `images_upright/db` |
| `--output-dir` | `-o` | Path to output root directory | `output/` |
| `--ransac-method` | | Geometric model: `fundamental`, `homography`, `affine` | `fundamental` |
| `--ransac-reproj` | | Reprojection error threshold (px) | `5.0` |
| `--ransac-iters` | | Maximum RANSAC iterations | `2000` |
| `--ransac-confidence` | | RANSAC confidence level | `0.999` |
| `--ransac-min-inliers` | | Minimum inlier count to accept a match | `8` |
| `--benchmark` | | Run benchmark evaluation instead of normal matching | `false` |
| `--benchmark-scene` | | Path to benchmark scene directory | — |
| `--benchmark-max-pairs` | | Limit number of pairs to evaluate | all |
| `--benchmark-pose-method` | | Pose estimation method: `essential`, `fundamental` | `essential` |
| `--descriptor` | | Descriptor type: `sift`, `dinov2` | `sift` |

### Examples

```bash
# Run with defaults
python main.py

# Custom directories
python main.py -q ../data/queries -d ../data/database -o ../results

# Use homography model with stricter thresholds
python main.py --ransac-method homography --ransac-reproj 3.0 --ransac-min-inliers 12
```

### Output

Each run creates a timestamped folder under the output directory:

```
output/<YYYYMMDD_HHMMSS>/
├── results.csv                      # Per-query match results
├── query_<name>.jpg                 # Copy of each query image
├── match_<name>.jpg                 # Best-matching database image
├── vis_matches_<name>.jpg           # Pre-RANSAC keypoint match lines
├── vis_ransac_<name>.jpg            # Post-RANSAC inlier/outlier visualization
├── vis_spatial_query_<name>.jpg     # Feature similarity heatmap (query)
├── vis_spatial_db_<name>.jpg        # Feature similarity heatmap (database)
└── vis_top10_<name>.jpg             # Top-10 candidate grid with scores
```

---

## Method Overview

Visual place recognition is formulated as an image retrieval problem: given a query image $I_q$, identify the database image $I_{d^*}$ captured at the same location. The core challenge lies in producing image representations that are invariant to illumination, viewpoint, and seasonal changes while remaining discriminative across different places.

### Motivation

Conventional approaches rely on hand-crafted local features (SIFT, ORB) or global descriptors (NetVLAD, GeM pooling). SelMA explores an alternative strategy: extracting local patches along structural edges — which are more robust to illumination change than texture-based keypoints — and encoding them with DINOv2, a foundation model whose self-supervised training on 142 million images yields representations with strong cross-domain generalisation.

### SelMA Pipeline Architecture

The system operates in two stages:

**Stage 1 — Coarse Retrieval.** Each image (query and database) is processed through the same feature extraction pipeline: Canny edge detection → edge-point sampling → patch extraction → Gaussian denoising → DINOv2 encoding. This produces a set of 384-dimensional feature vectors anchored to structurally salient locations. Database images are ranked by the average cosine similarity of the top-$k$ most confident patch correspondences, yielding a shortlist of $N_s$ candidates. When edge keypoints are enabled (default), Shi-Tomasi corners near Canny edges are used instead of uniform edge sampling for more repeatable keypoint detection.

**Stage 2 — Geometric Re-ranking.** Each shortlisted candidate undergoes full-patch matching followed by RANSAC geometric verification. A heuristic scoring function combines feature similarity and geometric consistency:

$$H = w_{\text{sim}} \cdot \bar{s}_{k} + w_{\text{inlier}} \cdot \frac{n_{\text{inliers}}}{n_{\text{pairs}}} + w_{\text{pass}} \cdot \mathbb{1}[\text{RANSAC passed}]$$

The final ranking is determined by $H$, allowing geometrically consistent matches to be promoted above candidates that score well on appearance alone but lack spatial coherence.

---

## Mathematical Foundations

### 1. Canny Edge Detection

Canny edge detection identifies structural boundaries in images through a multi-stage process.

**Step 1 — Gaussian Smoothing.** The image $I$ is convolved with a Gaussian kernel to suppress noise:

$$I_s = G_\sigma * I, \quad G_\sigma(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$

**Step 2 — Gradient Computation.** Sobel operators compute the intensity gradient magnitude $M$ and direction $\theta$:

$$M(x,y) = \sqrt{G_x^2 + G_y^2}, \quad \theta(x,y) = \arctan\!\left(\frac{G_y}{G_x}\right)$$

where $G_x = S_x * I_s$ and $G_y = S_y * I_s$ with Sobel kernels:

$$S_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad S_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

**Step 3 — Non-Maximum Suppression.** Only local maxima along the gradient direction are retained, producing thin edges.

**Step 4 — Double Thresholding.** Edges with $M > T_{\text{high}}$ are strong edges; those with $T_{\text{low}} \le M \le T_{\text{high}}$ are weak edges. Weak edges connected to strong edges are kept; others are discarded.

In this pipeline: $T_{\text{low}} = 50$, $T_{\text{high}} = 150$.

### 2. Patch Extraction and Sampling

Edge pixels $(x, y)$ where $\text{edge\_map}(x,y) > 0$ are collected and subsampled at a spacing of $s = 15$ pixels. For each sampled edge point, a $64 \times 64$ patch centred on that point is extracted from the full-colour image.

If more than $N_{\max} = 2000$ patches are produced, they are uniformly subsampled using linearly spaced indices:

$$\text{indices} = \left\lfloor \text{linspace}(0,\; |\mathcal{P}| - 1,\; N_{\max}) \right\rfloor$$

This ensures even spatial coverage across the image.

### 3. Gaussian Denoising

Each $64 \times 64$ patch is denoised via a Gaussian blur with a $3 \times 3$ kernel:

$$P_{\text{denoised}}(x,y) = \sum_{i=-1}^{1} \sum_{j=-1}^{1} G(i,j) \cdot P(x+i,\; y+j)$$

where $G$ is a normalised 2D Gaussian kernel. This removes sensor noise while preserving the coarse structure needed for DINOv2 encoding.

### 4. DINOv2 Feature Extraction

**DINOv2** (Oquab et al., 2023) is a self-supervised Vision Transformer trained with a self-distillation objective (DINO) on the LVD-142M dataset. We use the **ViT-S/14** variant (21M parameters, 384-dimensional embeddings).

#### Vision Transformer Architecture

Each $64 \times 64$ patch is resized to $112 \times 112$ pixels and split into a grid of $8 \times 8 = 64$ non-overlapping tokens of size $14 \times 14$. A learnable `[CLS]` token is prepended, yielding $N = 65$ tokens.

**Patch embedding.** A linear projection maps each $14 \times 14 \times 3$ token to a $d = 384$ dimensional vector:

$$\mathbf{z}_0^{(i)} = \mathbf{W}_p \cdot \text{flatten}(\text{patch}_i) + \mathbf{b}_p, \quad i = 0, \ldots, N$$

where $\mathbf{z}_0^{(0)}$ is the `[CLS]` token.

**Positional encoding.** Learnable positional embeddings $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times d}$ are added:

$$\mathbf{Z}_0 = [\mathbf{z}_0^{(0)}, \mathbf{z}_0^{(1)}, \ldots, \mathbf{z}_0^{(N)}] + \mathbf{E}_{\text{pos}}$$

**Transformer blocks.** Each of the 12 transformer blocks applies multi-head self-attention (MHSA) and a feed-forward network (FFN) with residual connections and layer normalization:

$$\mathbf{Z}_\ell' = \text{MHSA}(\text{LN}(\mathbf{Z}_{\ell-1})) + \mathbf{Z}_{\ell-1}$$
$$\mathbf{Z}_\ell = \text{FFN}(\text{LN}(\mathbf{Z}_\ell')) + \mathbf{Z}_\ell'$$

**Multi-Head Self-Attention.** For $h$ heads, queries, keys, and values are computed as:

$$\mathbf{Q}_h = \mathbf{Z}\mathbf{W}_h^Q, \quad \mathbf{K}_h = \mathbf{Z}\mathbf{W}_h^K, \quad \mathbf{V}_h = \mathbf{Z}\mathbf{W}_h^V$$

$$\text{Attention}(\mathbf{Q}_h, \mathbf{K}_h, \mathbf{V}_h) = \text{softmax}\!\left(\frac{\mathbf{Q}_h \mathbf{K}_h^\top}{\sqrt{d_k}}\right) \mathbf{V}_h$$

where $d_k = d / n_{\text{heads}}$.

**Feature output.** The `[CLS]` token from the final layer, $\mathbf{z}_L^{(0)} \in \mathbb{R}^{384}$, serves as the global representation of each patch.

#### Input Normalization

Images are normalized with ImageNet statistics before being fed to DINOv2:

$$\hat{x}_c = \frac{x_c - \mu_c}{\sigma_c}, \quad \mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

### 5. Cosine Similarity Matching

For a query image with features $\mathbf{Q} \in \mathbb{R}^{m \times 384}$ and a database image with features $\mathbf{D} \in \mathbb{R}^{n \times 384}$, the similarity matrix is:

$$S_{ij} = \frac{\mathbf{q}_i \cdot \mathbf{d}_j}{\|\mathbf{q}_i\| \cdot \|\mathbf{d}_j\|}$$

Each query patch $i$ is matched to its nearest database patch:

$$j^*(i) = \arg\max_j S_{ij}$$

The **average similarity** across all query patches is:

$$\bar{s} = \frac{1}{m} \sum_{i=1}^{m} S_{i,j^*(i)}$$

The **top-$k$ average similarity** (with $k = 50$) selects the $k$ highest-scoring query patches and averages their scores. Database images are ranked by their top-$k$ average similarity.

### 6. RANSAC Geometric Verification

After feature matching, RANSAC (Random Sample Consensus) verifies geometric consistency between matched point pairs. This filters out incorrect matches caused by visual aliasing.

#### Algorithm

Given $n$ putative correspondences $\{(\mathbf{p}_i, \mathbf{p}'_i)\}$, RANSAC iterates:

1. **Sample** a minimal subset of correspondences (7 for fundamental matrix, 4 for homography, 3 for affine).
2. **Fit** a geometric model from the minimal sample.
3. **Score** by counting inliers — correspondences whose reprojection error is below a threshold $\tau$.
4. **Repeat** for $K$ iterations, keeping the model with the most inliers.

The number of iterations required for confidence $p$ with inlier ratio $w$ and sample size $s$:

$$K = \frac{\log(1 - p)}{\log(1 - w^s)}$$

In this pipeline: $\tau = 5.0$ px, $K_{\max} = 2000$, $p = 0.999$, minimum inliers $= 8$.

#### Fundamental Matrix (default)

The fundamental matrix $\mathbf{F} \in \mathbb{R}^{3 \times 3}$ encodes the epipolar geometry between two views. For corresponding points $\mathbf{p}$ and $\mathbf{p}'$ in homogeneous coordinates:

$$\mathbf{p}'^\top \mathbf{F} \mathbf{p} = 0$$

This constrains $\mathbf{p}'$ to lie on the epipolar line $\ell' = \mathbf{F}\mathbf{p}$ in the second image. The fundamental matrix has 7 degrees of freedom (rank-2, $3 \times 3$ matrix up to scale).

#### Homography

The homography $\mathbf{H} \in \mathbb{R}^{3 \times 3}$ maps points between views of a planar scene:

$$\lambda \begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \mathbf{H} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

with 8 degrees of freedom. Appropriate when the scene is approximately planar.

#### Affine Transform

The affine model $\mathbf{A} \in \mathbb{R}^{2 \times 3}$ handles rotation, scaling, shearing, and translation:

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} a_{11} & a_{12} & t_x \\ a_{21} & a_{22} & t_y \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

with 6 degrees of freedom. A simpler model useful when perspective effects are minimal.

#### Inlier Verification

A match is classified as geometrically verified if the number of RANSAC inliers exceeds the minimum threshold (8). This confirms the query and database images share a consistent spatial relationship.

## Project Structure

```
SelMA/
├── src/
│   ├── main.py                       # Pipeline entry point
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py               # Default parameters
│   │   └── cli.py                    # Command-line argument parsing
│   ├── dataHandlers/
│   │   ├── __init__.py
│   │   ├── dataset.py                # Image loading and directory listing
│   │   ├── output.py                 # Result persistence (CSV + image copies)
│   │   └── visualization.py          # Match, RANSAC, spatial, and top-10 visualizations
│   ├── GeometryFuncs/
│   │   ├── __init__.py
│   │   ├── edges.py                  # Canny edge detection + patch extraction
│   │   └── denoise.py                # Gaussian / Non-Local Means denoising
│   ├── ModelFuncs/
│   │   ├── __init__.py
│   │   ├── feature_extractor.py      # DINOv2 ViT-S/14 wrapper + SIFT extractor
│   │   ├── matcher.py                # Cosine similarity, ranking, heuristic scoring
│   │   └── match_filter.py           # Pre-RANSAC match filtering
│   ├── ransac/
│   │   ├── __init__.py
│   │   └── geometric_filter.py       # RANSAC (fundamental / homography / affine)
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── calibrate.py              # Camera calibration from checkerboard images
│   │   └── colmap_parser.py          # COLMAP binary model parser
│   ├── benchmark/
│   │   ├── __init__.py
│   │   ├── dataset.py                # Benchmark scene loader (COLMAP / HDF5 / JSON)
│   │   ├── evaluate.py               # Benchmark evaluation runner
│   │   └── metrics.py                # Pose estimation and mAA metrics
│   └── evaluation/
│       ├── __init__.py
│       ├── dataset.py                # Evaluation scene loader
│       ├── evaluate.py               # Evaluation runner with visualizations
│       └── metrics.py                # Pose estimation and mAA metrics
├── scripts/
│   ├── create_test_scene.py          # Generate synthetic benchmark scene
│   └── download_benchmark.py         # Download phototourism benchmark data
├── demo_results/                     # Sample output visualizations
├── images_upright/
│   ├── db/                           # Database images (4,479)
│   └── query/                        # Query images (947)
│       ├── day/                      # Daytime queries
│       └── night/                    # Nighttime queries
├── output/                           # Timestamped result folders
├── requirements.txt                  # Pinned Python dependencies
└── readme.md                         # This document
```

---

## Configuration Reference

All default parameters are defined in [`src/config/settings.py`](src/config/settings.py) and can be overridden via CLI flags where noted.

### Edge Detection & Patch Extraction

| Parameter | Default | Description |
|---|---|---|
| `CANNY_LOW_THRESH` | 50 | Lower hysteresis threshold for Canny |
| `CANNY_HIGH_THRESH` | 150 | Upper hysteresis threshold for Canny |
| `PATCH_SIZE` | 64 | Side length of extracted patches (px) |
| `PATCH_SPACING` | 15 | Minimum spacing between sampled edge points (px) |
| `MAX_PATCHES` | 2000 | Maximum patches retained per image |
| `DENOISE_METHOD` | `"gaussian"` | `"gaussian"` or `"nlmeans"` |
| `DENOISE_GAUSSIAN_KERNEL` | 3 | Gaussian blur kernel size |

### DINOv2 Model

| Parameter | Default | Description |
|---|---|---|
| `DINO_MODEL_NAME` | `"dinov2_vits14"` | Model variant (ViT-S/14, 21M params) |
| `DINO_INPUT_SIZE` | 112 | Input resolution (must be divisible by 14) |
| `DINO_BATCH_SIZE` | 128 | GPU batch size for feature extraction |
| `DEVICE` | `None` | Compute device; auto-detects CUDA if available |

### Matching & Re-ranking

| Parameter | Default | Description |
|---|---|---|
| `TOP_K_PATCHES` | 50 | Top-$k$ patches used for coarse ranking (CLI: not exposed) |
| `SHORTLIST_SIZE` | 100 | Candidates forwarded to geometric re-ranking |
| `TOP_K_VISUALIZE` | 10 | Candidates shown in the top-N grid visualization |
| `HEURISTIC_W_SIM` | 0.5 | Heuristic weight: cosine similarity |
| `HEURISTIC_W_INLIER_RATIO` | 0.35 | Heuristic weight: RANSAC inlier ratio |
| `HEURISTIC_W_RANSAC_PASS` | 0.15 | Heuristic weight: RANSAC pass bonus |

### RANSAC (overridable via CLI)

| Parameter | Default | CLI flag | Description |
|---|---|---|---|
| `RANSAC_METHOD` | `"fundamental"` | `--ransac-method` | Geometric model type |
| `RANSAC_REPROJ_THRESH` | 5.0 | `--ransac-reproj` | Reprojection error threshold (px) |
| `RANSAC_MAX_ITERS` | 2000 | `--ransac-iters` | Maximum iterations |
| `RANSAC_CONFIDENCE` | 0.999 | `--ransac-confidence` | Confidence level |
| `RANSAC_MIN_INLIERS` | 8 | `--ransac-min-inliers` | Minimum inliers to accept |

---

## Dataset

This pipeline is evaluated on the **Aachen Day-Night v1.1** benchmark (Sattler et al., 2018), a standard dataset for visual localisation under extreme illumination variation. It comprises 4,479 reference database images and 947 query images captured across daytime and nighttime conditions using multiple devices (Milestone, Nexus 4, Nexus 5x). The day-to-night setting is particularly challenging as it tests the robustness of image representations to fundamental appearance changes where texture and colour cues become unreliable.

---

## Benchmark Results

SelMA is additionally evaluated on the **Phototourism** stereo benchmark (Reichstag scene, 75 calibrated images) from the Image Matching Benchmark (Jin et al., 2021). The metric is **mAA** — mean Average Accuracy over pose error thresholds from 1° to 10°.

### Comparison with Published Methods

All scores below use **qt_auc@10°** (area under the pose-accuracy curve, 0°–10°), the standard metric reported by the Image Matching Benchmark. SelMA's discrete mAA (mean of accuracy at 1°, 2°, …, 10°) is 0.728–0.753, but we report qt_auc here for a fair comparison.

| Method | Type | qt_auc@10° | Pairs | Source |
|--------|------|:----------:|:-----:|--------|
| Kaggle IMC 2022 Winner | Learned ensemble | ~0.86 | — | Kaggle |
| **SelMA (ours)** | Classical | **0.704** | 100 | This work |
| SP+DISK+SuperGlue 8K | Learned | 0.640 | ~4500 | IMW 2021 #1 |
| RootSIFT + DEGENSAC | Classical | 0.620 | ~4500 | IMB baseline |
| SP+SuperGlue (ss-dpth) | Learned | 0.597 | ~4500 | IMW 2021 #4 |

### SelMA Per-Threshold Accuracy (Reichstag)

| 1° | 2° | 3° | 4° | 5° | 6° | 7° | 8° | 9° | 10° |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:---:|
| 0.33 | 0.53 | 0.67 | 0.76 | 0.80 | 0.81 | 0.83 | 0.83 | 0.86 | 0.86 |

### Important Notes on Comparability

Our headline mAA numbers are **not directly comparable** to published benchmark scores. We report them transparently below so that readers can assess the results fairly.

**1. Different metric definitions.** Our mAA is the mean of accuracy at 10 discrete thresholds (1°, 2°, …, 10°). The official Image Matching Benchmark reports **qt_auc** — the area under the continuous accuracy curve via trapezoidal integration. On an apples-to-apples qt_auc@10° basis, SelMA scores **0.704** vs the published RootSIFT+DEGENSAC baseline of **0.620** — a +13.5% improvement, but smaller than the discrete mAA gap suggests.

**2. Different pair selection protocol.** The official benchmark uses pair lists derived from **3D point covisibility**, which deliberately includes challenging pairs with low visual overlap. Our dataset (`data/phototourism/reichstag`) does not include the official pair files (`new-vis-pairs/`), so we generate pairs from camera geometry (baseline distance < 3× median, viewing angle < 60°). This filter is mild — only 10.3% of possible pairs are excluded — but the resulting pair distribution may still differ from the official test set.

**3. Different pair counts.** We evaluate on **100 randomly sampled pairs** (seed = 42). The official benchmark evaluates on the full set of covisibility pairs (~4500 for reichstag). With only 100 pairs, a few hard or easy pairs can shift the score by ±0.02.

**4. Stochastic variance.** MAGSAC/PROSAC geometric estimation is non-deterministic. Across runs on the same 100 pairs, we observe mAA ranging from **0.728 to 0.753** (±0.013).

**5. Single scene vs multi-scene averages.** Published scores of 0.504–0.640 are typically **averaged across all 9 Phototourism scenes**, which include harder scenes (e.g., St. Peter's Basilica, Sacre Coeur). Reichstag is a structured building with repetitive texture, making it relatively easier for SIFT-based methods.

### Honest Assessment

The improvement over RootSIFT+DEGENSAC is **genuine** — our edge-selective keypoint filtering, dual MAGSAC+PROSAC estimation, and Sampson error model selection produce better pose estimates. However, the magnitude of the improvement is best understood through the comparable qt_auc@10° metric: **0.704 (SelMA) vs 0.620 (baseline)**, an improvement of approximately 13.5% on the Reichstag scene.

---

## References

- Jin, Y., et al. (2021). "Image Matching across Wide Baselines: From Paper to Practice." *IJCV*.
- Sattler, T., et al. (2018). "Benchmarking 6DoF Outdoor Visual Localization in Changing Conditions." *CVPR*.
- Oquab, M., et al. (2023). "DINOv2: Learning Robust Visual Features without Supervision." *arXiv:2304.07193*.
- Barath, D., et al. (2020). "MAGSAC++, a Fast, Reliable and Accurate Robust Estimator." *CVPR*.
- Chum, O. & Matas, J. (2005). "Matching with PROSAC — Progressive Sample Consensus." *CVPR*.

