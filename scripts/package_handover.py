#!/usr/bin/env python3
"""Create a compact, checksummed Day 7 handover package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile


REQUIRED_FILES = [
    "README.md",
    "docs/data_inventory.md",
    "docs/reproduction_and_handover.md",
    "docs/known_issues.md",
    "reports/day7/qa_report.md",
    "configs/study_area.yaml",
    "configs/landsat_catalog.yaml",
    "configs/orchestrate_sources.yaml",
    "configs/built_up_candidates.yaml",
    "configs/mapping_validation.yaml",
    "configs/final_dataset.yaml",
    (
        "data/final/yaounde_urban_expansion_30m_v2/"
        "historical_demand.csv"
    ),
    (
        "data/final/yaounde_urban_expansion_30m_v2/"
        "urban_sprawl_metrics.csv"
    ),
    (
        "data/final/yaounde_urban_expansion_30m_v2/"
        "primary_feature_schema.yaml"
    ),
    (
        "data/final/yaounde_urban_expansion_30m_v2/"
        "data_dictionary.yaml"
    ),
    (
        "data/final/yaounde_urban_expansion_30m_v2/"
        "dataset_card.md"
    ),
    (
        "data/metadata/final_dataset/"
        "final_dataset_version.json"
    ),
    "data/metadata/final_dataset/checksums.sha256",
]


INCLUDE_ROOTS = [
    "configs",
    "docs",
    "reports/day7",
    "data/metadata",
    (
        "data/final/"
        "yaounde_urban_expansion_30m_v2"
    ),
    "tests/reference",
]


EXCLUDED_PARTS = {
    "_checkpoints",
    "__pycache__",
    ".pytest_cache",
}


EXCLUDED_SUFFIXES = {
    ".parquet",
    ".gpkg",
    ".tif",
}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_included(path: Path, root: Path) -> bool:
    """Return whether one file belongs in the metadata package."""
    relative = path.relative_to(root)

    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False

    if path.suffix.casefold() in EXCLUDED_SUFFIXES:
        return False

    return True


def collect_files(root: Path) -> list[Path]:
    """Collect required documentation and compact metadata files."""
    files: set[Path] = {
        root / "README.md",
        root / "Makefile",
        root / "pyproject.toml",
    }

    for item in INCLUDE_ROOTS:
        candidate = root / item
        if not candidate.exists():
            continue

        if candidate.is_file():
            files.add(candidate)
            continue

        for path in candidate.rglob("*"):
            if path.is_file() and is_included(path, root):
                files.add(path)

    return sorted(path for path in files if path.is_file())


def validate_required(root: Path) -> list[str]:
    """Return required files that are absent."""
    return [
        relative
        for relative in REQUIRED_FILES
        if not (root / relative).is_file()
    ]


def write_manifest(
    files: list[Path],
    root: Path,
    manifest_path: Path,
) -> None:
    """Write the handover file manifest."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "relative_path",
                "size_bytes",
                "sha256",
            ]
        )

        for path in files:
            writer.writerow(
                [
                    path.relative_to(root).as_posix(),
                    path.stat().st_size,
                    sha256(path),
                ]
            )


def main() -> None:
    """Validate and package the current release."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
    )
    arguments = parser.parse_args()

    root = arguments.root.resolve()
    missing = validate_required(root)

    if missing:
        raise SystemExit(
            "Missing required handover files:\n- "
            + "\n- ".join(missing)
        )

    files = collect_files(root)
    dist = root / "dist"
    manifest = dist / "HANDOVER_MANIFEST.csv"
    write_manifest(files, root, manifest)

    summary = {
        "status": "PASS",
        "file_count": len(files),
        "manifest": str(manifest),
    }

    if arguments.check_only:
        print(json.dumps(summary, indent=2))
        return

    archive_path = (
        dist
        / "yaounde_urban_expansion_30m_v2_handover.zip"
    )
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in files:
            archive.write(
                path,
                path.relative_to(root),
            )
        archive.write(
            manifest,
            manifest.relative_to(root),
        )

    summary["archive"] = str(archive_path)
    summary["archive_sha256"] = sha256(archive_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
