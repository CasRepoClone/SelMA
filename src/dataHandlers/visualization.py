import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path


def draw_match_lines(img_query, img_db, query_points, db_points,
                     pairs, title="Keypoint Matches"):
    """Draw keypoint matches with connecting lines between query and DB image."""
    h1, w1 = img_query.shape[:2]
    h2, w2 = img_db.shape[:2]
    h_out = max(h1, h2)
    canvas = np.zeros((h_out, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = _ensure_color(img_query)
    canvas[:h2, w1:] = _ensure_color(img_db)

    for qi, di, score in pairs:
        qx, qy = int(query_points[qi][0]), int(query_points[qi][1])
        dx, dy = int(db_points[di][0]) + w1, int(db_points[di][1])
        color = _score_color(score)
        cv2.line(canvas, (qx, qy), (dx, dy), color, 1, cv2.LINE_AA)
        cv2.circle(canvas, (qx, qy), 3, (0, 255, 0), -1)
        cv2.circle(canvas, (dx, dy), 3, (0, 255, 0), -1)

    return canvas


def draw_ransac_matches(img_query, img_db, query_points, db_points,
                        pairs, inlier_mask):
    """Draw matches after RANSAC: inliers in green, outliers in red."""
    h1, w1 = img_query.shape[:2]
    h2, w2 = img_db.shape[:2]
    h_out = max(h1, h2)
    canvas = np.zeros((h_out, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = _ensure_color(img_query)
    canvas[:h2, w1:] = _ensure_color(img_db)

    # Draw outliers first (red, thin)
    for idx, (qi, di, score) in enumerate(pairs):
        if idx < len(inlier_mask) and not inlier_mask[idx]:
            qx, qy = int(query_points[qi][0]), int(query_points[qi][1])
            dx, dy = int(db_points[di][0]) + w1, int(db_points[di][1])
            cv2.line(canvas, (qx, qy), (dx, dy), (0, 0, 180), 1, cv2.LINE_AA)
            cv2.circle(canvas, (qx, qy), 3, (0, 0, 255), -1)
            cv2.circle(canvas, (dx, dy), 3, (0, 0, 255), -1)

    # Draw inliers on top (green, thicker)
    for idx, (qi, di, score) in enumerate(pairs):
        if idx < len(inlier_mask) and inlier_mask[idx]:
            qx, qy = int(query_points[qi][0]), int(query_points[qi][1])
            dx, dy = int(db_points[di][0]) + w1, int(db_points[di][1])
            cv2.line(canvas, (qx, qy), (dx, dy), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(canvas, (qx, qy), 4, (0, 255, 0), -1)
            cv2.circle(canvas, (dx, dy), 4, (0, 255, 0), -1)

    return canvas


def draw_spatial_features(image, points, features, title="Spatial Feature Map"):
    """Visualize DINOv2 feature similarity as a spatial heatmap over the image.

    For each keypoint, computes its average cosine similarity to all other
    keypoints, producing a per-point 'centrality' score that reveals the
    spatial structure of the learned representations.
    """
    if len(points) == 0 or features.shape[0] == 0:
        return _ensure_color(image).copy()

    # Compute pairwise cosine similarity
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    feat_norm = features / norms
    sim_matrix = feat_norm @ feat_norm.T

    # Per-point average similarity (excluding self)
    n = sim_matrix.shape[0]
    np.fill_diagonal(sim_matrix, 0)
    avg_sim = sim_matrix.sum(axis=1) / max(n - 1, 1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    img_rgb = _ensure_color(image)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    sc = ax.scatter(xs, ys, c=avg_sim, cmap="plasma", s=12, alpha=0.85,
                    edgecolors="none", norm=Normalize(vmin=avg_sim.min(),
                                                      vmax=avg_sim.max()))
    plt.colorbar(sc, ax=ax, label="Avg Feature Similarity", shrink=0.7)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    fig.tight_layout()

    # Render to numpy array
    fig.canvas.draw()
    buf = np.frombuffer(np.array(fig.canvas.buffer_rgba())[:,:,:3].tobytes(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)


def save_visualizations(run_dir, query_name, match_canvas,
                        ransac_canvas, spatial_query, spatial_db):
    """Save all visualization images to the output folder."""
    run_dir = Path(run_dir)
    stem = Path(query_name).stem

    cv2.imwrite(str(run_dir / f"vis_matches_{stem}.jpg"), match_canvas)
    cv2.imwrite(str(run_dir / f"vis_ransac_{stem}.jpg"), ransac_canvas)
    cv2.imwrite(str(run_dir / f"vis_spatial_query_{stem}.jpg"), spatial_query)
    cv2.imwrite(str(run_dir / f"vis_spatial_db_{stem}.jpg"), spatial_db)


def draw_top_candidates(query_img, query_name, candidates):
    """Draw a grid showing the query image and its top-N DB candidates.

    Parameters
    ----------
    query_img : ndarray  – BGR query image
    query_name : str     – query filename
    candidates : list of dict, each with keys:
        'image'           – BGR DB image
        'name'            – DB filename
        'rank'            – 1-based rank
        'heuristic_score' – combined heuristic score
        'top_k_avg'       – top-k average cosine similarity
        'ransac_inliers'  – int
        'ransac_passed'   – bool
    """
    THUMB_W, THUMB_H = 320, 240
    PAD = 10
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    n = len(candidates)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols

    # Layout: query on left, candidate grid on right
    grid_w = cols * (THUMB_W + PAD) + PAD
    grid_h = rows * (THUMB_H + 50 + PAD) + PAD   # 50px for text
    total_w = THUMB_W + PAD * 3 + grid_w
    total_h = max(THUMB_H + 80, grid_h + 40)

    canvas = np.full((total_h, total_w, 3), 30, dtype=np.uint8)

    # Draw query image on left
    q_resized = cv2.resize(_ensure_color(query_img), (THUMB_W, THUMB_H))
    y0 = 40
    canvas[y0:y0 + THUMB_H, PAD:PAD + THUMB_W] = q_resized
    cv2.putText(canvas, f"Query: {query_name}", (PAD, y0 - 10),
                FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw candidates in grid
    x_off = THUMB_W + PAD * 2
    for i, cand in enumerate(candidates):
        col = i % cols
        row = i // cols
        cx = x_off + PAD + col * (THUMB_W + PAD)
        cy = 40 + row * (THUMB_H + 50 + PAD)

        thumb = cv2.resize(_ensure_color(cand['image']), (THUMB_W, THUMB_H))

        # Color border by heuristic score (green=high, red=low)
        hscore = cand['heuristic_score']
        border_color = (0, int(min(hscore, 1.0) * 255),
                        int((1 - min(hscore, 1.0)) * 255))
        cv2.rectangle(canvas, (cx - 2, cy - 2),
                       (cx + THUMB_W + 1, cy + THUMB_H + 1),
                       border_color, 2)

        canvas[cy:cy + THUMB_H, cx:cx + THUMB_W] = thumb

        # Labels
        label1 = f"#{cand['rank']}  H={hscore:.3f}  Sim={cand['top_k_avg']:.3f}"
        label2 = f"RANSAC: {cand['ransac_inliers']}{'  PASS' if cand['ransac_passed'] else ''}"
        label3 = cand['name']
        cv2.putText(canvas, label1, (cx, cy + THUMB_H + 15),
                    FONT, 0.4, (200, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, label2, (cx, cy + THUMB_H + 30),
                    FONT, 0.4, (200, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, label3, (cx, cy + THUMB_H + 45),
                    FONT, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

    return canvas


def save_top_candidates(run_dir, query_name, canvas):
    """Save the top candidates grid image."""
    run_dir = Path(run_dir)
    stem = Path(query_name).stem
    cv2.imwrite(str(run_dir / f"vis_top10_{stem}.jpg"), canvas)


def _ensure_color(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _score_color(score):
    """Map similarity score [0,1] to BGR color (blue=low, green=high)."""
    g = int(score * 255)
    b = int((1 - score) * 255)
    return (b, g, 0)
