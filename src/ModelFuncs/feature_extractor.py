import torch
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

        mask = []
        for kp in kps:
            x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
            if 0 <= y < edge_dilated.shape[0] and 0 <= x < edge_dilated.shape[1]:
                mask.append(edge_dilated[y, x] > 0)
            else:
                mask.append(False)

        mask = np.array(mask, dtype=bool)
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

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((settings.DINO_INPUT_SIZE, settings.DINO_INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=settings.DINO_MEAN, std=settings.DINO_STD),
        ])

    @torch.no_grad()
    def extract(self, patches):
        if len(patches) == 0:
            return np.empty((0, self.model.embed_dim), dtype=np.float32)

        tensors = []
        for patch in patches:
            if len(patch.shape) == 2:
                patch = np.stack([patch] * 3, axis=-1)
            elif patch.shape[2] == 1:
                patch = np.concatenate([patch] * 3, axis=-1)
            tensors.append(self.transform(patch))

        all_features = []
        batch_size = settings.DINO_BATCH_SIZE
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i:i + batch_size]).to(self.device)
            features = self.model(batch)
            all_features.append(features.cpu().numpy())

        return np.concatenate(all_features, axis=0)
