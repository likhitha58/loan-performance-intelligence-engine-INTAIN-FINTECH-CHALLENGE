from pathlib import Path

from src.data_intelligence.loader import (
    DATA_DIR,
    check_required_files,
    load_data_pack,
    summarize_shapes,
)


def test_required_files_exist():
    files = check_required_files(DATA_DIR)

    assert len(files) == 6

    for path in files.values():
        assert Path(path).exists()


def test_data_pack_loads():
    datasets = load_data_pack(DATA_DIR)

    assert set(datasets.keys()) == {
        "train",
        "test",
        "static",
        "servicer_updates",
        "macro_scenarios",
        "submission_template",
    }

    for df in datasets.values():
        assert not df.empty


def test_shape_summary():
    datasets = load_data_pack(DATA_DIR)

    summary = summarize_shapes(datasets)

    assert len(summary) == 6
    assert set(summary.columns) == {
        "dataset",
        "rows",
        "columns",
    }