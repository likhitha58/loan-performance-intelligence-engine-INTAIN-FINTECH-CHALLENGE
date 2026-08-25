from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


REQUIRED_FILES = {
    "train": "loan_monthly_performance_train.csv",
    "test": "loan_monthly_performance_test.csv",
    "static": "loan_static_attributes.csv",
    "servicer_updates": "servicer_updates.csv",
    "macro_scenarios": "macro_scenarios.csv",
    "submission_template": "submission_template.csv",
}


def check_required_files(data_dir: Path = DATA_DIR) -> Dict[str, Path]:
    """
    Verify that all required CSV artifacts exist.

    Returns
    -------
    dict
        Mapping between logical dataset name and file path.

    Raises
    ------
    FileNotFoundError
        If one or more required files are missing.
    """
    files = {}

    missing = []

    for dataset_name, filename in REQUIRED_FILES.items():
        path = data_dir / filename

        if path.exists():
            files[dataset_name] = path
        else:
            missing.append(str(path))

    if missing:
        message = (
            "Required data files are missing:\n"
            + "\n".join(f" - {path}" for path in missing)
        )
        raise FileNotFoundError(message)

    return files


def load_data_pack(data_dir: Path = DATA_DIR) -> Dict[str, pd.DataFrame]:
    """
    Load the synthetic/organizer data pack into memory.

    Parameters
    ----------
    data_dir:
        Directory containing the data-pack CSV files.

    Returns
    -------
    dict
        Dictionary containing pandas DataFrames.
    """
    files = check_required_files(data_dir)

    datasets = {}

    for dataset_name, path in files.items():
        datasets[dataset_name] = pd.read_csv(path)

    return datasets


def summarize_shapes(
    datasets: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Return row/column counts for every loaded dataset.
    """
    rows = []

    for name, df in datasets.items():
        rows.append(
            {
                "dataset": name,
                "rows": len(df),
                "columns": len(df.columns),
            }
        )

    return pd.DataFrame(rows)