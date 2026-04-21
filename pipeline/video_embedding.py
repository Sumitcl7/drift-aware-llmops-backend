from typing import List
import cv2
import numpy as np
from PIL import Image


def embed_video_file(
    video_path: str,
    image_embedder,
    sample_fps: float = 1.0,
    max_frames: int = 24
) -> List[float]:
    """
    Sample frames from video, embed each frame using image embedder, then mean-pool.
    Returns 384-d vector (same as image embedding pipeline).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    step = max(int(round(fps / sample_fps)), 1)

    frame_embeddings = []
    idx = 0
    taken = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if idx % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                vec = image_embedder.embed_pil_image(pil)
                frame_embeddings.append(np.array(vec, dtype=np.float32))
                taken += 1

                if taken >= max_frames:
                    break

            idx += 1
    finally:
        cap.release()

    if not frame_embeddings:
        raise ValueError("No frames were sampled from video")

    pooled = np.mean(np.stack(frame_embeddings, axis=0), axis=0)
    pooled = pooled / (np.linalg.norm(pooled) + 1e-12)
    return pooled.astype(np.float32).tolist()