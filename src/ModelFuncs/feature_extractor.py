import torch
import torchvision.transforms as T
import numpy as np

from config import settings


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
