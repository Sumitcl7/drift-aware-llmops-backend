from datetime import datetime, timezone
from apscheduler.schedulers.background import BackgroundScheduler

from monitoring.retrain_trigger import evaluate_retrain_need
from scripts.retrain import run_retrain_pipeline
from database.supabase_client import supabase

scheduler = BackgroundScheduler(timezone="UTC")


def log_retrain_event(
    event_type: str,
    drift_score: float,
    sample_count: int,
    status: str,
    reason: str
):
    payload = {
        "event_type": event_type,
        "drift_score": float(drift_score),
        "sample_count": int(sample_count),
        "status": str(status),
        "reason": str(reason),
        # created_at uses DB default now()
    }

    print("[Scheduler] Inserting retrain_event payload:", payload)
    resp = supabase.table("retrain_events").insert(payload).execute()
    print("[Scheduler] Insert response:", resp)
    return resp


def scheduled_retrain_check():
    try:
        print("[Scheduler] Running retrain evaluation...")
        decision = evaluate_retrain_need()
        print("[Scheduler] Decision:", decision)

        trigger = bool(decision.get("trigger_retrain", False))
        drift_score = float(decision.get("drift_score", 0.0))
        sample_count = int(decision.get("sample_count", 0))
        reason = str(decision.get("reason", ""))

        if trigger:
            print("[Scheduler] Trigger is TRUE. Running retrain pipeline...")
            result = run_retrain_pipeline()

            status = "triggered_success" if result.get("status") == "success" else "triggered_failed"
            reason_full = f"{reason} | retrain_result={result.get('message', '')}"

            log_retrain_event(
                event_type="retrain_trigger",
                drift_score=drift_score,
                sample_count=sample_count,
                status=status,
                reason=reason_full
            )
            print("[Scheduler] Retrain result:", result)
        else:
            print("[Scheduler] Trigger is FALSE. Skipping retrain.")
            log_retrain_event(
                event_type="retrain_trigger",
                drift_score=drift_score,
                sample_count=sample_count,
                status="skipped",
                reason=reason
            )

    except Exception as e:
        import traceback
        print("[Scheduler] ERROR in scheduled_retrain_check:", str(e))
        traceback.print_exc()


def start_retrain_scheduler(hours: int = 6):
    """
    Starts periodic retrain evaluation every `hours`.
    """
    if scheduler.running:
        print("[Scheduler] Scheduler already running.")
        return

    scheduler.add_job(
        scheduled_retrain_check,
        trigger="interval",
        hours=hours,
        id="retrain_check_job",
        replace_existing=True,
        max_instances=1,
        coalesce=True
    )
    scheduler.start()
    print(f"[Scheduler] Started retrain scheduler (every {hours}h).")


def stop_retrain_scheduler():
    if scheduler.running:
        scheduler.shutdown(wait=False)
        print("[Scheduler] Stopped retrain scheduler.")