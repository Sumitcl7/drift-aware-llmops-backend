from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from database.supabase_client import supabase


def _safe_select(table: str, columns: str = "*", order_col: Optional[str] = None, desc: bool = False):
    try:
        q = supabase.table(table).select(columns)
        if order_col:
            q = q.order(order_col, desc=desc)
        resp = q.execute()
        return resp.data or []
    except Exception as e:
        print(f"[dashboard_data] select failed table={table} columns={columns} err={e}")
        return []


def _parse_dt(v: Any) -> Optional[datetime]:
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except Exception:
        return None


def _pick(row: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def _load_retrain_events():
    # try common names and minimal columns first
    for table in ["retrain_events", "retrain_log", "retrain_history"]:
        rows = _safe_select(table, "*", "created_at", False)
        if rows:
            return rows
    return []


def get_summary() -> Dict[str, Any]:
    interactions = _safe_select("interaction_log", "*", "created_at", False)
    retrain_events = _load_retrain_events()

    model_counter = Counter()
    modality_counter = Counter()

    for r in interactions:
        model = _pick(r, ["model_used", "model_name", "model"], "unknown")
        modality = _pick(r, ["modality"], "text")
        model_counter[str(model)] += 1
        modality_counter[str(modality)] += 1

    latest_drift_score = 0.0
    for ev in reversed(retrain_events):
        ds = ev.get("drift_score")
        if ds is not None:
            try:
                latest_drift_score = float(ds)
                break
            except Exception:
                pass

    retrain_trigger_count = 0
    for ev in retrain_events:
        trig = ev.get("triggered")
        status = str(ev.get("status", "")).lower()
        if isinstance(trig, bool):
            if trig:
                retrain_trigger_count += 1
        elif status in ["triggered", "success", "ran", "run", "true"]:
            retrain_trigger_count += 1

    return {
        "total_interactions": len(interactions),
        "model_usage": dict(model_counter),
        "modality_usage": dict(modality_counter),
        "latest_drift_score": latest_drift_score,
        "retrain_trigger_count": retrain_trigger_count,
    }


def get_charts() -> Dict[str, Any]:
    interactions = _safe_select("interaction_log", "*", "created_at", False)
    retrain_events = _load_retrain_events()

    queries_over_time = defaultdict(int)
    model_usage_distribution = Counter()
    modality_distribution = Counter()
    drift_over_time = {}
    retrain_timeline = defaultdict(lambda: {"skipped": 0, "triggered": 0})

    for r in interactions:
        dt = _parse_dt(r.get("created_at"))
        day = dt.date().isoformat() if dt else "unknown"
        queries_over_time[day] += 1
        model_usage_distribution[str(_pick(r, ["model_used", "model_name", "model"], "unknown"))] += 1
        modality_distribution[str(_pick(r, ["modality"], "text"))] += 1

    for ev in retrain_events:
        dt = _parse_dt(ev.get("created_at"))
        day = dt.date().isoformat() if dt else "unknown"

        ds = ev.get("drift_score")
        if ds is not None:
            try:
                drift_over_time[day] = float(ds)
            except Exception:
                pass

        trig = ev.get("triggered")
        status = str(ev.get("status", "")).lower()
        is_triggered = (trig is True) or (status in ["triggered", "success", "ran", "run", "true"])
        retrain_timeline[day]["triggered" if is_triggered else "skipped"] += 1

    return {
        "queries_over_time": dict(sorted(queries_over_time.items())),
        "model_usage_distribution": dict(model_usage_distribution),
        "modality_distribution": dict(modality_distribution),
        "drift_over_time": dict(sorted(drift_over_time.items())),
        "retrain_timeline": dict(sorted(retrain_timeline.items())),
    }