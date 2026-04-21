from typing import List
import numpy as np
from PIL import Image
import torch


class OpenCLIPImageEmbedder:
    """
    Memory-safe dev embedder.
    - If OpenCLIP loads successfully, use it.
    - If not, fallback to lightweight RGB-histogram embedding (384-dim).
    """

    def __init__(
        self,
        model_name: str = "RN50",
        pretrained: str = "openai",
        device: str = None,
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._mode = "fallback"
        self.model = None
        self.preprocess = None

        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            model.to(self.device)
            model.eval()
            self.model = model
            self.preprocess = preprocess
            self._mode = "open_clip"
        except Exception as e:
            print(f"[ImageEmbedder] OpenCLIP unavailable, using fallback embedder. Reason: {e}")

    def _fallback_embed_384(self, image: Image.Image) -> List[float]:
        """
        384-dim fallback:
        RGB histograms with 128 bins each => 128*3 = 384
        """
        image = image.convert("RGB").resize((224, 224))
        arr = np.array(image, dtype=np.float32)

        r = arr[:, :, 0].flatten()
        g = arr[:, :, 1].flatten()
        b = arr[:, :, 2].flatten()

        bins = 128
        rh, _ = np.histogram(r, bins=bins, range=(0, 255), density=True)
        gh, _ = np.histogram(g, bins=bins, range=(0, 255), density=True)
        bh, _ = np.histogram(b, bins=bins, range=(0, 255), density=True)

        vec = np.concatenate([rh, gh, bh]).astype(np.float32)  # 384
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        return vec.tolist()

    @torch.no_grad()
    def embed_pil_image(self, image: Image.Image) -> List[float]:
        image = image.convert("RGB")

        if self._mode == "open_clip":
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            features = self.model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            vec = features.squeeze(0).cpu().numpy().astype(np.float32)

            # ensure DB compatibility with vector(384)
            if vec.shape[0] > 384:
                vec = vec[:384]
            elif vec.shape[0] < 384:
                vec = np.pad(vec, (0, 384 - vec.shape[0]))

            return vec.tolist()

        return self._fallback_embed_384(image)

    @property
    def mode(self) -> str:
        return self._mode