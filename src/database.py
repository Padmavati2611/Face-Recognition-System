"""Local SQLite storage for registered face embeddings and attendance."""

from __future__ import annotations

import csv
import io
import json
import math
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections.abc import Iterable, Iterator


_USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class DuplicateUserError(ValueError):
    """Raised when a user ID is already registered."""


class InvalidUserError(ValueError):
    """Raised when a user ID, name, or embedding is invalid."""


class Database:
    """Small connection-per-operation SQLite repository."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create the schema, and surface corrupt database files clearly."""
        try:
            with self._connect() as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        display_name TEXT NOT NULL,
                        embedding TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                        timestamp TEXT NOT NULL,
                        recognition_status TEXT NOT NULL DEFAULT 'recognized'
                    );
                    CREATE INDEX IF NOT EXISTS idx_attendance_timestamp
                        ON attendance(timestamp DESC);
                    CREATE INDEX IF NOT EXISTS idx_attendance_user_timestamp
                        ON attendance(user_id, timestamp DESC);
                    """
                )
        except sqlite3.DatabaseError as exc:
            raise sqlite3.DatabaseError(
                f"Could not initialize the SQLite database at {self.path}: {exc}"
            ) from exc

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    def register_user(
        self, user_id: str, display_name: str, embedding: Iterable[float]
    ) -> None:
        user_id = user_id.strip()
        display_name = display_name.strip()
        if not _USER_ID_PATTERN.fullmatch(user_id):
            raise InvalidUserError(
                "User ID must be 1-64 characters using letters, numbers, '.', '_' or '-'."
            )
        if not display_name or len(display_name) > 100:
            raise InvalidUserError("Name must contain 1-100 characters.")
        try:
            values = [float(value) for value in embedding]
        except (TypeError, ValueError) as exc:
            raise InvalidUserError("The face embedding is invalid.") from exc
        if not values or any(not math.isfinite(value) for value in values):
            raise InvalidUserError("The face embedding is empty or invalid.")

        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO users (user_id, display_name, embedding, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        user_id,
                        display_name,
                        json.dumps(values, separators=(",", ":")),
                        self._utc_now().isoformat(timespec="seconds"),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            if "UNIQUE" in str(exc).upper() or "PRIMARY KEY" in str(exc).upper():
                raise DuplicateUserError(f"User ID '{user_id}' is already registered.") from exc
            raise

    def get_users(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT user_id, display_name, embedding, created_at "
                "FROM users ORDER BY display_name COLLATE NOCASE"
            ).fetchall()
        users: list[dict[str, object]] = []
        for row in rows:
            try:
                embedding = json.loads(row["embedding"])
            except (TypeError, json.JSONDecodeError):
                continue
            users.append(
                {
                    "user_id": row["user_id"],
                    "display_name": row["display_name"],
                    "embedding": embedding,
                    "created_at": row["created_at"],
                }
            )
        return users

    def delete_user(self, user_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            return cursor.rowcount > 0

    def record_attendance(
        self,
        user_id: str,
        duplicate_window_seconds: int = 3600,
        now: datetime | None = None,
    ) -> bool:
        """Record a recognized user unless they were recorded in the cooldown window."""
        current = now or self._utc_now()
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        current = current.astimezone(timezone.utc)
        timestamp = current.isoformat(timespec="seconds")
        cutoff = (current - timedelta(seconds=max(0, duplicate_window_seconds))).isoformat(
            timespec="seconds"
        )
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
            if exists is None:
                raise InvalidUserError(f"User ID '{user_id}' is not registered.")
            recent = connection.execute(
                "SELECT 1 FROM attendance WHERE user_id = ? AND timestamp >= ? LIMIT 1",
                (user_id, cutoff),
            ).fetchone()
            if recent:
                return False
            connection.execute(
                "INSERT INTO attendance (user_id, timestamp, recognition_status) "
                "VALUES (?, ?, 'recognized')",
                (user_id, timestamp),
            )
        return True

    def get_attendance(self, limit: int | None = None) -> list[dict[str, str]]:
        query = (
            "SELECT a.user_id, u.display_name, a.timestamp, a.recognition_status "
            "FROM attendance a JOIN users u ON u.user_id = a.user_id "
            "ORDER BY a.timestamp DESC, a.id DESC"
        )
        parameters: tuple[int, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            parameters = (max(0, limit),)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def export_attendance_csv(self) -> str:
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(["user_id", "display_name", "timestamp", "recognition_status"])
        for row in self.get_attendance():
            writer.writerow(
                [row["user_id"], row["display_name"], row["timestamp"], row["recognition_status"]]
            )
        return output.getvalue()