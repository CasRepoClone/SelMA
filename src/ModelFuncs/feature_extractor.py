"""Feature extraction backends: DINOv2 ViT-S/14 and SIFT at edge keypoints.

DINOv2Extractor encodes image patches into 384-d feature vectors using
a self-supervised Vision Transformer. SIFTAtEdgeKeypoints detects SIFT
features filtered to Canny edge locations for the benchmark pipeline.
"""

import torch
import torch.nn.functional as F_torch
import torchvision.transforms as T
import numpy as np
import cv2

from config import settings


class SIFTAtEdgeKeypoints:
    """Detect SIFT features, keep only those near Canny edges.

    SelMA's 'Selective Edge Location' filtering with SIFT's full
    multi-scale detection + orientation assignment for robust descriptors.

    Enhancements:
    - Multi-scale edge detection (union of multiple Canny thresholds)
    - CLAHE contrast normalization
    - Spatial diversity: top non-edge features for translation estimation
    """

    def __init__(self, nfeatures=0, n_octave_layers=3, contrast_thresh=0.04,
                 edge_threshold=10, sigma=1.6):
        self.sift = cv2.SIFT_create(
            nfeatures=nfeatures,
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_thresh,
            edgeThreshold=edge_threshold,
            sigma=sigma,
        )

    def detect_and_compute(self, image, edge_map, edge_proximity=None):
        """Detect SIFT keypoints, keep only those near edges.

        Parameters
        ----------
        image : ndarray
            Full image (color or grayscale).
        edge_map : ndarray
            Binary Canny edge map.
        edge_proximity : int
            Max pixel distance from nearest edge.

        Returns
        -------
        descriptors : ndarray (N, 128) or None
        valid_points : list of (x, y)
        """
        edge_proximity = edge_proximity or settings.KEYPOINT_EDGE_PROXIMITY
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # CLAHE: local contrast normalization for more robust features
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)

        # SIFT detection on CLAHE-enhanced image
        kps, desc = self.sift.detectAndCompute(gray_clahe, None)

        if desc is None or len(kps) == 0:
            return None, []

        # Filter: keep only keypoints near edges (SelMA identity)
        kernel_size = edge_proximity * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        edge_dilated = cv2.dilate(edge_map, kernel)

        # Vectorized edge proximity check
        coords = np.array([(int(round(kp.pt[0])), int(round(kp.pt[1])))
                           for kp in kps], dtype=np.int32)
        xs, ys = coords[:, 0], coords[:, 1]
        h_ed, w_ed = edge_dilated.shape[:2]
        in_bounds = (ys >= 0) & (ys < h_ed) & (xs >= 0) & (xs < w_ed)
        mask = np.zeros(len(kps), dtype=bool)
        bounded_idx = np.where(in_bounds)[0]
        if len(bounded_idx) > 0:
            mask[bounded_idx] = edge_dilated[
                ys[bounded_idx], xs[bounded_idx]] > 0

        if mask.sum() == 0:
            return None, []

        filtered_desc = desc[mask]
        # Keep subpixel precision — rounding costs 1-2° pose accuracy
        filtered_pts = [(kps[i].pt[0], kps[i].pt[1])
                        for i in range(len(kps)) if mask[i]]

        return filtered_desc, filtered_pts

    def compute(self, image, points, patch_size=None):
        """Compute SIFT descriptors at specified keypoint locations."""
        patch_size = patch_size or settings.PATCH_SIZE
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        kps = [cv2.KeyPoint(float(x), float(y), float(patch_size))
               for x, y in points]

        kps, desc = self.sift.compute(gray, kps)

        if desc is None or len(kps) == 0:
            return None, []

        valid_points = [(int(round(k.pt[0])), int(round(k.pt[1]))) for k in kps]
        return desc, valid_points


class DINOv2Extractor:
    def __init__(self, model_name=None, device=None):
        model_name = model_name or settings.DINO_MODEL_NAME
        if device is None:
            device = settings.DEVICE
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = torch.hub.load(
            "facebookresearch/dinov2", model_name
        )
        self.model.eval().to(self.device)

        # Pre-compute normalization tensors for batch processing
        self._mean = torch.tensor(
            settings.DINO_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        self._std = torch.tensor(
            settings.DINO_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self._input_size = settings.DINO_INPUT_SIZE

        # Enable FP16 automatic mixed precision on CUDA
        self._use_amp = (self.device.type == "cuda")

        # Keep legacy transform as fallback for edge cases
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((settings.DINO_INPUT_SIZE, settings.DINO_INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=settings.DINO_MEAN, std=settings.DINO_STD),
        ])

    def _prepare_batch_tensor(self, patches):
        """Convert list of numpy patches to a normalized (N,3,H,W) tensor.

        Replaces the per-patch ToPILImage → Resize → ToTensor → Normalize
        pipeline with a single batched operation for major speed gains.
        """
        # Handle grayscale → 3-channel conversion
        needs_convert = False
        for p in patches:
            if len(p.shape) == 2 or (len(p.shape) == 3 and p.shape[2] == 1):
                needs_convert = True
                break

        if needs_convert:
            processed = []
            for patch in patches:
                if len(patch.shape) == 2:
                    patch = np.stack([patch, patch, patch], axis=-1)
                elif patch.shape[2] == 1:
                    patch = np.concatenate([patch, patch, patch], axis=-1)
                processed.append(patch)
            batch_np = np.stack(processed, axis=0)
        else:
            # Fast path: all patches are 3-channel, stack directly
            batch_np = np.stack(patches, axis=0)

        # (N, H, W, 3) uint8 → (N, 3, H, W) float32 [0, 1]
        batch_tensor = torch.from_numpy(
            batch_np).permute(0, 3, 1, 2).float().div_(255.0)

        # Batch resize to DINOv2 input size
        h, w = batch_tensor.shape[2], batch_tensor.shape[3]
        if h != self._input_size or w != self._input_size:
            batch_tensor = F_torch.interpolate(
                batch_tensor,
                size=(self._input_size, self._input_size),
                mode='bilinear',
                align_corners=False,
            )

        # Batch normalize (ImageNet stats)
        batch_tensor.sub_(self._mean).div_(self._std)

        return batch_tensor

    @torch.inference_mode()
    def extract(self, patches):
        if len(patches) == 0:
            return np.empty((0, self.model.embed_dim), dtype=np.float32)

        # Batch tensor preparation (replaces per-patch PIL transform)
        batch_tensor = self._prepare_batch_tensor(patches)

        all_features = []
        batch_size = settings.DINO_BATCH_SIZE
        for i in range(0, len(batch_tensor), batch_size):
            batch = batch_tensor[i:i + batch_size].to(self.device)
            if self._use_amp:
                with torch.amp.autocast('cuda'):
                    features = self.model(batch)
                features = features.float()
            else:
                features = self.model(batch)
            all_features.append(features.cpu().numpy())

        return np.concatenate(all_features, axis=0)
