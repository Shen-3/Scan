from __future__ import annotations


class AlignmentError(RuntimeError):
    """Raised when automatic alignment cannot be trusted."""

    def __init__(self, message: str, *, inliers: int = 0, total_matches: int = 0) -> None:
        super().__init__(message)
        self.inliers = inliers
        self.total_matches = total_matches


__all__ = ["AlignmentError"]

