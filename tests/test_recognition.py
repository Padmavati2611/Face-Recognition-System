import unittest

import numpy as np

from src.face_recognition import FaceMatcher


class FaceMatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matcher = FaceMatcher(threshold=0.8)
        self.users = [
            {"user_id": "member-1", "display_name": "Sam", "embedding": [1.0, 0.0, 0.0]}
        ]

    def test_matching_embedding_is_identified(self) -> None:
        user, score = self.matcher.identify(np.array([0.99, 0.01, 0.0]), self.users)
        self.assertEqual(user["user_id"], "member-1")
        self.assertGreater(score, 0.8)

    def test_below_threshold_embedding_stays_unknown(self) -> None:
        user, _ = self.matcher.identify(np.array([0.0, 1.0, 0.0]), self.users)
        self.assertIsNone(user)

    def test_empty_registry_stays_unknown(self) -> None:
        user, score = self.matcher.identify(np.array([1.0, 0.0, 0.0]), [])
        self.assertIsNone(user)
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()