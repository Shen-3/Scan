from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from app.models import ProcessingResult


def export_csv(result: ProcessingResult, export_dir: Path) -> Tuple[Path, Path]:
    """Export processing result to summary and points CSV files."""
    export_dir.mkdir(parents=True, exist_ok=True)
    summary_path = export_dir / f"{result.target_id}_summary.csv"
    points_path = export_dir / f"{result.target_id}_points.csv"

    summary_df = pd.DataFrame([result.to_csv_summary()])
    summary_df.to_csv(summary_path, index=False)

    points_df = pd.DataFrame(result.to_points_table())
    points_df.to_csv(points_path, index=False)
    return summary_path, points_path


__all__ = ["export_csv"]
