"""
Export mesh + eigenfunction data to JSON for web app. Float32 to minimize size.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def export_json(
    vertices: np.ndarray,
    faces: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    path: str | Path,
) -> None:
    """
    vertices (V,3), faces (F,3), eigenvalues (N,), eigenvectors (N,V).
    JSON fields: vertices, faces, eigenvalues, eigenvectors.
    eigenvectors: list of lists row-wise (N rows, each length V). Float32.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "vertices": vertices.astype(np.float32).tolist(),
        "faces": faces.astype(np.int32).tolist(),
        "eigenvalues": eigenvalues.astype(np.float32).tolist(),
        "eigenvectors": eigenvectors.astype(np.float32).tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))


def load_json(path: str | Path) -> dict[str, Any]:
    """Load exported JSON (for verification/testing)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
