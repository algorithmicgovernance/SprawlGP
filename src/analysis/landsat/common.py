"""Shared path, configuration, geometry and Earth Engine helpers for Day 2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import ee
import geopandas as gpd
import yaml
from shapely.geometry import mapping


def find_project_root(start: Path | None = None) -> Path:
    """Locate the repository root using ``pyproject.toml`` or ``.git``."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project root.")


def resolve_project_path(value: str | Path, root: Path) -> Path:
    """Resolve a configuration path relative to the repository root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping and reject empty or invalid root objects."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        content = yaml.safe_load(stream)
    if not isinstance(content, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return content


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    if not path.is_file():
        raise FileNotFoundError(path)
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return content


def load_grid_specification(path: Path) -> dict[str, Any]:
    """Load and validate the authoritative Day 1 grid specification."""
    specification = load_json(path)
    required = {"crs", "resolution_m", "transform", "width", "height"}
    missing = required.difference(specification)
    if missing:
        raise ValueError(f"Missing grid fields: {sorted(missing)}")
    if len(specification["transform"]) != 6:
        raise ValueError("The affine transform must contain six values.")
    return specification


def load_single_geometry(path: Path) -> gpd.GeoDataFrame:
    """Read one dissolved geometry and validate CRS and content."""
    frame = gpd.read_file(path)
    if len(frame) != 1 or frame.crs is None:
        raise ValueError(f"Expected one georeferenced geometry in {path}.")
    if frame.geometry.iloc[0].is_empty:
        raise ValueError(f"Empty geometry in {path}.")
    return frame


def load_ee_geometry(path: Path) -> ee.Geometry:
    """Convert one local dissolved boundary to an Earth Engine geometry."""
    frame = load_single_geometry(path).to_crs("EPSG:4326")
    return ee.Geometry(mapping(frame.geometry.iloc[0]))


def query_rectangle(path: Path) -> ee.Geometry:
    """Create a simple WGS84 rectangle for efficient ``filterBounds`` calls."""
    frame = load_single_geometry(path).to_crs("EPSG:4326")
    xmin, ymin, xmax, ymax = map(float, frame.total_bounds)
    return ee.Geometry.Rectangle([xmin, ymin, xmax, ymax], geodesic=False)


def initialize_earth_engine(project: str | None) -> None:
    """Initialize Earth Engine and raise an actionable authentication error."""
    try:
        ee.Initialize(project=project) if project else ee.Initialize()
    except Exception as error:
        raise RuntimeError(
            "Earth Engine initialization failed. Run `earthengine authenticate` "
            "and verify the configured Cloud project."
        ) from error


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 checksum of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
