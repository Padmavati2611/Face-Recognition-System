"""Attendance reporting helpers shared by dashboard views."""

from datetime import datetime, timezone

from src.database import Database


def dashboard_metrics(database: Database) -> dict[str, int]:
    today = datetime.now(timezone.utc).date().isoformat()
    rows = database.get_attendance()
    return {
        "registered_users": len(database.get_users()),
        "today_attendance": sum(row["timestamp"].startswith(today) for row in rows),
        "total_records": len(rows),
    }