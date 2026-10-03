import csv
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.database import Database, DuplicateUserError, InvalidUserError


class DatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.temp_dir.name) / "test.db")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def register(self) -> None:
        self.database.register_user("alex-01", "Alex Morgan", [0.1, 0.2, 0.3])

    def test_database_creation_and_user_registration(self) -> None:
        self.register()
        users = self.database.get_users()
        self.assertEqual(len(users), 1)
        self.assertEqual(users[0]["user_id"], "alex-01")
        self.assertEqual(users[0]["embedding"], [0.1, 0.2, 0.3])

    def test_duplicate_user_id_is_rejected(self) -> None:
        self.register()
        with self.assertRaises(DuplicateUserError):
            self.register()

    def test_invalid_user_id_is_rejected(self) -> None:
        with self.assertRaises(InvalidUserError):
            self.database.register_user("alex 01", "Alex", [0.1])

    def test_attendance_insertion_and_duplicate_prevention(self) -> None:
        self.register()
        now = datetime(2025, 1, 1, 12, tzinfo=timezone.utc)
        self.assertTrue(self.database.record_attendance("alex-01", 3600, now))
        self.assertFalse(
            self.database.record_attendance("alex-01", 3600, now + timedelta(minutes=10))
        )
        self.assertEqual(len(self.database.get_attendance()), 1)

    def test_attendance_allowed_after_cooldown(self) -> None:
        self.register()
        now = datetime(2025, 1, 1, 12, tzinfo=timezone.utc)
        self.database.record_attendance("alex-01", 3600, now)
        self.assertTrue(
            self.database.record_attendance("alex-01", 3600, now + timedelta(hours=2))
        )

    def test_csv_export(self) -> None:
        self.register()
        self.database.record_attendance("alex-01")
        rows = list(csv.DictReader(self.database.export_attendance_csv().splitlines()))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["display_name"], "Alex Morgan")
        self.assertEqual(rows[0]["recognition_status"], "recognized")

    def test_delete_user_removes_linked_attendance(self) -> None:
        self.register()
        self.database.record_attendance("alex-01")
        self.assertTrue(self.database.delete_user("alex-01"))
        self.assertEqual(self.database.get_attendance(), [])


if __name__ == "__main__":
    unittest.main()