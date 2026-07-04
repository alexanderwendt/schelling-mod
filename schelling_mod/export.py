"""Single-run export helpers."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import migrate_config


def serialize_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return JSON-safe migrated config."""
    return migrate_config(config)


def build_run_id(config: Mapping[str, Any]) -> str:
    """Build stable run id from reproducibility-relevant config."""
    payload = json.dumps(serialize_config(config), sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    seed = serialize_config(config)["simulation"].get("seed", 0)
    return f"seed-{seed}-{digest}"


def write_run_export(
    output_dir: Path,
    config: Mapping[str, Any],
    metric_rows: list[Mapping[str, object]],
) -> tuple[Path, Path]:
    """Write metrics.csv and config.json for one run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    config_path = output_dir / "config.json"

    fieldnames: list[str] = []
    for row in metric_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with metrics_path.open("w", encoding="utf-8", newline="") as metrics_file:
        if fieldnames:
            writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in metric_rows:
                writer.writerow(row)

    exported_config = serialize_config(config)
    exported_config.setdefault("export_metadata", {})
    exported_config["export_metadata"].update(
        {
            "run_id": build_run_id(exported_config),
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(exported_config, config_file, indent=2, sort_keys=True)
        config_file.write("\n")

    return metrics_path, config_path
