from sentence_transformers import SentenceTransformer
from database.supabase_client import supabase

_text_model = None


def get_text_model():
    global _text_model
    if _text_model is None:
        _text_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _text_model


def log_query(query: str):
    """
    Backward-compatible existing text logger.
    Text model is lazy-loaded to avoid startup crashes on Windows.
    """
    embedding = get_text_model().encode(query).tolist()

    response = supabase.table("interaction_log").insert({
        "user_query": query,
        "model_response": "placeholder",
        "embedding": embedding,
        "refusal_flag": False,
        "toxicity_flag": False
    }).execute()

    return response


def log_multimodal_event(
    request_id: str,
    modality: str,           # text | image | video
    input_ref: str,          # query text / filename / URL
    embedding: list,
    model_name: str,
    model_response: str = "placeholder",
    refusal_flag: bool = False,
    toxicity_flag: bool = False
):
    payload = {
        "request_id": request_id,
        "modality": modality,
        "input_ref": input_ref,
        "user_query": input_ref if modality == "text" else None,
        "model_response": model_response,
        "embedding": embedding,
        "model_name": model_name,
        "refusal_flag": refusal_flag,
        "toxicity_flag": toxicity_flag
    }

    return supabase.table("interaction_log").insert(payload).execute()