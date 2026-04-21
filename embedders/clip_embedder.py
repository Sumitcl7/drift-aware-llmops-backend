import torch
import open_clip
from PIL import Image
import numpy as np
from typing import Union, Tuple
import logging

logger = logging.getLogger(__name__)

class OpenCLIPEmbedder:
    """
    Multimodal embedder using OpenCLIP.
    Embeds both text and images in same semantic space.
    """
    
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        """
        Args:
            model_name: OpenCLIP model (ViT-B-32, ViT-L-14, etc.)
            pretrained: Which pretrained weights (openai, laion400M_e31, etc.)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading OpenCLIP on device: {self.device}")
        
        try:
            self.model, self.preprocess = open_clip.create_model_and_transforms(
                model_name, 
                pretrained=pretrained,
                device=self.device
            )
            self.model.eval()
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.embedding_dim = 512 if "B-32" in model_name else 768
            logger.info(f"OpenCLIP loaded: {model_name}, embedding_dim={self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load OpenCLIP: {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed text to 512-dim vector."""
        try:
            tokens = self.tokenizer(text).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(tokens)
                # Normalize to unit sphere for cosine similarity
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy().flatten().astype(np.float32)
        except Exception as e:
            logger.error(f"Text embedding failed: {e}")
            raise
    
    def embed_image(self, image_input: Union[str, Image.Image]) -> np.ndarray:
        """
        Embed image to 512-dim vector.
        
        Args:
            image_input: PIL Image or path to image file
            
        Returns:
            512-dim numpy array
        """
        try:
            # Load image if path provided
            if isinstance(image_input, str):
                image = Image.open(image_input).convert("RGB")
            else:
                image = image_input.convert("RGB")
            
            # Preprocess
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Embed
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten().astype(np.float32)
        except Exception as e:
            logger.error(f"Image embedding failed: {e}")
            raise
    
    def embed_batch(self, images: list) -> np.ndarray:
        """Batch embed multiple images for efficiency."""
        try:
            processed = []
            for img in images:
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                else:
                    img = img.convert("RGB")
                processed.append(self.preprocess(img))
            
            batch_tensor = torch.stack(processed).to(self.device)
            
            with torch.no_grad():
                features = self.model.encode_image(batch_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
            
            return features.cpu().numpy().astype(np.float32)
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            raise