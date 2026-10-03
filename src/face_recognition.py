"""Conservative cosine-similarity matching for SFace embeddings."""

import numpy as np


class FaceMatcher:
    def __init__(self, threshold: float = 0.42) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("Recognition threshold must be between 0 and 1.")
        self.threshold = threshold

    def identify(
        self, embedding: np.ndarray, users: list[dict[str, object]]
    ) -> tuple[dict[str, object] | None, float]:
        probe = np.asarray(embedding, dtype=np.float32).reshape(-1)
        probe_norm = float(np.linalg.norm(probe))
        if probe_norm == 0:
            return None, 0.0
        best_user: dict[str, object] | None = None
        best_score = -1.0
        for user in users:
            try:
                candidate = np.asarray(user["embedding"], dtype=np.float32).reshape(-1)
                candidate_norm = float(np.linalg.norm(candidate))
                if candidate.shape != probe.shape or candidate_norm == 0:
                    continue
                score = float(np.dot(probe, candidate) / (probe_norm * candidate_norm))
            except (KeyError, TypeError, ValueError):
                continue
            if score > best_score:
                best_user, best_score = user, score
        if best_score < self.threshold:
            return None, max(best_score, 0.0)
        return best_user, best_score