import os
from datetime import datetime, timezone
from typing import Dict, Any, List
import numpy as np

from database.supabase_client import supabase


# -----------------------------
# Config (env-driven)
# -----------------------------
MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", "20"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.25"))
LOOKBACK_LIMIT = int(os.getenv("DRIFT_LOOKBACK_LIMIT", "500"))
COOLDOWN_HOURS = int(os.getenv("RETRAIN_COOLDOWN_HOURS", "6"))
FORCE_RETRAIN = os.getenv("FORCE_RETRAIN", "false").lower() == "true"


def _safe_rows(resp) -> List[dict]:
    return resp.data or []


def _parse_embedding(e):
    """
    Accepts embedding as list[float] OR string like "[0.1, 0.2, ...]".
    Returns np.ndarray or None.
    """
    if e is None:
        return None

    if isinstance(e, list):
        try:
            return np.array(e, dtype=np.float32)
        except Exception:
            return None

    if isinstance(e, str):
        s = e.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                vals = [float(x.strip()) for x in s[1:-1].split(",") if x.strip()]
                return np.array(vals, dtype=np.float32)
            except Exception:
                return None

    return None


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    sim = float(np.dot(a, b) / denom)
    return 1.0 - sim


def _hours_since(ts_iso: str) -> float:
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        diff = now - dt.astimezone(timezone.utc)
        return diff.total_seconds() / 3600.0
    except Exception:
        return 10**9  # effectively "long ago"


def _get_latest_retrain_event():
    rows = _safe_rows(
        supabase.table("retrain_events")
        .select("id, status, created_at")
        .order("id", desc=True)
        .limit(1)
        .execute()
    )
    return rows[0] if rows else None


def _load_recent_embeddings():
    """
    Pull recent multimodal logs that have embeddings.
    Expects interaction_log table with fields:
      - embedding
      - created_at
      - modality (optional)
    """
    rows = _safe_rows(
        supabase.table("interaction_log")
        .select("embedding, created_at, modality")
        .order("id", desc=True)
        .limit(LOOKBACK_LIMIT)
        .execute()
    )

    parsed = []
    for r in rows:
        vec = _parse_embedding(r.get("embedding"))
        if vec is not None and vec.size > 0:
            parsed.append(
                {
                    "embedding": vec,
                    "created_at": r.get("created_at"),
                    "modality": r.get("modality", "unknown"),
                }
            )
    return parsed


def _compute_drift_score(emb_rows: List[dict]) -> float:
    """
    Simple baseline drift:
      - split into older half vs newer half
      - compute centroid distance (cosine distance)
    """
    if len(emb_rows) < 2:
        return 0.0

    # reverse to chronological ascending (old -> new)
    emb_rows = list(reversed(emb_rows))
    vecs = [r["embedding"] for r in emb_rows]

    # Ensure same dimensionality
    dim = vecs[0].shape[0]
    vecs = [v for v in vecs if v.shape[0] == dim]
    if len(vecs) < 2:
        return 0.0

    mid = len(vecs) // 2
    if mid == 0 or mid == len(vecs):
        return 0.0

    older = np.stack(vecs[:mid], axis=0)
    newer = np.stack(vecs[mid:], axis=0)

    c_old = older.mean(axis=0)
    c_new = newer.mean(axis=0)

    return float(_cosine_distance(c_old, c_new))


def evaluate_retrain_need() -> Dict[str, Any]:
    """
    Returns:
    {
      "trigger_retrain": bool,
      "reason": str,
      "drift_score": float,
      "sample_count": int,
      "config": {...}
    }
    """
    # 0) Force mode for demo day
    if FORCE_RETRAIN:
        return {
            "trigger_retrain": True,
            "reason": "force_retrain_enabled",
            "drift_score": 1.0,
            "sample_count": 0,
            "config": {
                "min_samples": MIN_SAMPLES,
                "drift_threshold": DRIFT_THRESHOLD,
                "cooldown_hours": COOLDOWN_HOURS,
                "force_retrain": FORCE_RETRAIN,
            },
        }

    # 1) Cooldown gate
    latest_event = _get_latest_retrain_event()
    if latest_event:
        hrs = _hours_since(latest_event.get("created_at", ""))
        if hrs < COOLDOWN_HOURS:
            return {
                "trigger_retrain": False,
                "reason": f"cooldown_active ({hrs:.2f}h < {COOLDOWN_HOURS}h)",
                "drift_score": 0.0,
                "sample_count": 0,
                "config": {
                    "min_samples": MIN_SAMPLES,
                    "drift_threshold": DRIFT_THRESHOLD,
                    "cooldown_hours": COOLDOWN_HOURS,
                    "force_retrain": FORCE_RETRAIN,
                },
            }

    # 2) Load embeddings
    emb_rows = _load_recent_embeddings()
    sample_count = len(emb_rows)

    if sample_count < MIN_SAMPLES:
        return {
            "trigger_retrain": False,
            "reason": f"insufficient_samples ({sample_count} < {MIN_SAMPLES})",
            "drift_score": 0.0,
            "sample_count": sample_count,
            "config": {
                "min_samples": MIN_SAMPLES,
                "drift_threshold": DRIFT_THRESHOLD,
                "cooldown_hours": COOLDOWN_HOURS,
                "force_retrain": FORCE_RETRAIN,
            },
        }

    # 3) Drift compute
    drift_score = _compute_drift_score(emb_rows)

    if drift_score >= DRIFT_THRESHOLD:
        return {
            "trigger_retrain": True,
            "reason": f"drift_above_threshold ({drift_score:.4f} >= {DRIFT_THRESHOLD})",
            "drift_score": drift_score,
            "sample_count": sample_count,
            "config": {
                "min_samples": MIN_SAMPLES,
                "drift_threshold": DRIFT_THRESHOLD,
                "cooldown_hours": COOLDOWN_HOURS,
                "force_retrain": FORCE_RETRAIN,
            },
        }

    return {
        "trigger_retrain": False,
        "reason": f"drift_below_threshold ({drift_score:.4f} < {DRIFT_THRESHOLD})",
        "drift_score": drift_score,
        "sample_count": sample_count,
        "config": {
            "min_samples": MIN_SAMPLES,
            "drift_threshold": DRIFT_THRESHOLD,
            "cooldown_hours": COOLDOWN_HOURS,
            "force_retrain": FORCE_RETRAIN,
        },
    }