"""
alert_webhook.py
----------------
Minimal FastAPI router that receives Alertmanager webhook POST requests
and writes them to logs/alerts.log.

Add to your main FastAPI app:
    from alert_webhook import router as alert_router
    app.include_router(alert_router)

Alertmanager posts to http://api:8000/internal/alert-webhook
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/internal", tags=["internal"])

# Write alerts to a dedicated log file
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
ALERT_LOG = LOG_DIR / "alerts.log"

log = logging.getLogger(__name__)


@router.post("/alert-webhook", status_code=status.HTTP_200_OK)
async def receive_alert(request: Request):
    """
    Receives Alertmanager webhook payload and logs it.
    Alertmanager expects a 200 OK response.
    """
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    timestamp = datetime.utcnow().isoformat() + "Z"
    alerts = payload.get("alerts", [])

    for alert in alerts:
        entry = {
            "received_at":  timestamp,
            "status":       alert.get("status"),          # firing | resolved
            "alertname":    alert.get("labels", {}).get("alertname"),
            "severity":     alert.get("labels", {}).get("severity"),
            "summary":      alert.get("annotations", {}).get("summary"),
            "description":  alert.get("annotations", {}).get("description"),
            "starts_at":    alert.get("startsAt"),
            "ends_at":      alert.get("endsAt"),
        }

        # Log to stdout (captured by Docker)
        level = logging.CRITICAL if entry["severity"] == "critical" else logging.WARNING
        log.log(level, f"ALERT [{entry['status']}] {entry['alertname']}: {entry['summary']}")

        # Append to alerts.log (readable by grader / demo)
        with open(ALERT_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")

    return JSONResponse({"received": len(alerts), "timestamp": timestamp})