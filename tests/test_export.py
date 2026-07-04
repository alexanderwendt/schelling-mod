"""Tests for single-run exports."""

import json

from schelling_mod.app import build_config_from_cli_args
from schelling_mod.app import build_default_streamlit_config
from schelling_mod.app import build_parser
from schelling_mod.export import write_run_export


def test_write_run_export_creates_metrics_csv_and_config_json(tmp_path) -> None:
    """Exporter should write required files."""
    config = build_default_streamlit_config()
    rows = [{"run_id": "run-1", "iteration": 0, "satisfaction_percentage": 100.0}]

    metrics_path, config_path = write_run_export(tmp_path, config, rows)

    assert metrics_path.name == "metrics.csv"
    assert config_path.name == "config.json"
    assert "run_id,iteration" in metrics_path.read_text(encoding="utf-8")
    assert json.loads(config_path.read_text(encoding="utf-8"))["config_version"] == 1


def test_exported_dynamic_config_can_be_loaded_by_cli(tmp_path) -> None:
    """CLI --config-json should preserve dynamic group count."""
    config = build_default_streamlit_config()
    config["group_count"] = 2
    config["groups"] = config["groups"][:2]
    _, config_path = write_run_export(tmp_path, config, [])

    args = build_parser().parse_args(["--run_simulation", "--config-json", str(config_path)])
    loaded = build_config_from_cli_args(args)

    assert loaded["group_count"] == 2
    assert len(loaded["groups"]) == 2
