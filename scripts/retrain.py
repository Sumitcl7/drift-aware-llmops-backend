from datetime import datetime, timezone

def run_retrain_pipeline() -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    finished_at = datetime.now(timezone.utc).isoformat()
    return {
        "status": "success",
        "message": "Retrain pipeline stub executed",
        "started_at": started_at,
        "finished_at": finished_at
    }