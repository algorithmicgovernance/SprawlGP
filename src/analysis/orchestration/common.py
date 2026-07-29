"""Shared configuration, provenance, grid and Earth Engine utilities for Day 3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import ee
import yaml


PROJECT_MARKERS = ("pyproject.toml", ".git")


def find_project_root(start: Path | None = None) -> Path:
    """Locate the repository root by walking upward from a starting path."""
    current = (start or Path.cwd()).resolve()

    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in PROJECT_MARKERS):
            return candidate

    raise FileNotFoundError(
        "Could not locate the repository root. Run the command from inside "
        "the project or ensure pyproject.toml exists."
    )


def resolve_project_path(path_value: str | Path, project_root: Path) -> Path:
    """Resolve an absolute or repository-relative path consistently."""
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else project_root / path


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping and reject missing or malformed documents."""
    if not path.is_file():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as stream:
        content = yaml.safe_load(stream)

    if not isinstance(content, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")

    return content


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object and reject missing or non-object documents."""
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")

    content = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(content, dict):
        raise ValueError(f"Expected a JSON object in {path}.")

    return content


def write_json(path: Path, content: dict[str, Any]) -> None:
    """Write deterministic, human-readable JSON metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(content, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def write_yaml(path: Path, content: dict[str, Any]) -> None:
    """Write YAML metadata while preserving the supplied key order."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(content, stream, sort_keys=False, allow_unicode=True)


def sha256_file(path: Path) -> str:
    """Calculate a SHA-256 checksum without loading the whole file in memory."""
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def stable_object_hash(value: Any) -> str:
    """Hash a JSON-serialisable object using canonical key ordering."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def initialize_earth_engine(project: str) -> None:
    """Initialize Earth Engine with an actionable authentication error."""
    try:
        ee.Initialize(project=project)
    except Exception as error:
        raise RuntimeError(
            "Earth Engine initialization failed. Authenticate once with "
            "`earthengine authenticate` and confirm that the configured "
            f"project '{project}' has Earth Engine access."
        ) from error


def load_grid_specification(path: Path) -> dict[str, Any]:
    """Load and validate the authoritative Day 1 grid definition."""
    grid = load_json(path)
    required = {
        "crs",
        "resolution_m",
        "transform",
        "width",
        "height",
        "extent",
    }
    missing = required.difference(grid)

    if missing:
        raise ValueError(f"Missing grid fields: {sorted(missing)}")

    if len(grid["transform"]) != 6:
        raise ValueError("The frozen affine transform must contain six values.")

    return grid


def exact_grid_region(grid: dict[str, Any]) -> ee.Geometry:
    """Create the exact projected export rectangle from the frozen grid extent."""
    extent = grid["extent"]
    return ee.Geometry.Rectangle(
        [
            float(extent["xmin"]),
            float(extent["ymin"]),
            float(extent["xmax"]),
            float(extent["ymax"]),
        ],
        proj=str(grid["crs"]),
        geodesic=False,
    )


def asset_exists(asset_id: str) -> bool:
    """Return whether an Earth Engine asset exists without hiding API errors."""
    try:
        ee.data.getAsset(asset_id)
        return True
    except Exception as error:
        message = str(error).casefold()

        if "not found" in message or "does not exist" in message:
            return False

        raise


def ensure_asset_folder(folder_id: str) -> None:
    """Create all missing folders below a project asset root."""
    marker = "/assets/"

    if marker not in folder_id:
        raise ValueError(
            "Asset folders must use the projects/<project>/assets/... form."
        )

    prefix, suffix = folder_id.split(marker, 1)
    current = f"{prefix}{marker.rstrip('/')}"

    for part in [value for value in suffix.split("/") if value]:
        current = f"{current}/{part}"

        if not asset_exists(current):
            ee.data.createAsset({"type": "FOLDER"}, current)


def metadata_directory(config: dict[str, Any], project_root: Path) -> Path:
    """Create and return the configured Day 3 metadata directory."""
    path = resolve_project_path(config["metadata"]["directory"], project_root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def report_directory(config: dict[str, Any], project_root: Path) -> Path:
    """Create and return the configured Day 3 report directory."""
    path = resolve_project_path(config["reports"]["directory"], project_root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def task_status(task_id: str) -> dict[str, Any]:
    """Retrieve one submitted task by ID using the public task-list interface."""
    for task in ee.batch.Task.list():
        if task.id == task_id:
            return task.status()

    raise LookupError(
        f"Earth Engine task '{task_id}' was not found among recent tasks."
    )
