from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ASC_COLUMNS = ["x", "y", "z", "r", "g", "b", "intensity", "classification"]


def read_cras_asc_sample(zip_path: str | Path, max_points: int = 200_000) -> pd.DataFrame:
    zip_path = Path(zip_path)
    rows: list[list[float]] = []
    with zipfile.ZipFile(zip_path) as zf:
        asc_names = [n for n in zf.namelist() if n.lower().endswith(".asc")]
        if not asc_names:
            raise FileNotFoundError(f"no ASC file found in {zip_path}")
        with zf.open(asc_names[0]) as f:
            for raw in f:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line or line.startswith("//"):
                    continue
                parts = line.split()
                if len(parts) != 8:
                    continue
                try:
                    rows.append([float(v) for v in parts])
                except ValueError:
                    continue
                if len(rows) >= max_points:
                    break
    df = pd.DataFrame(rows, columns=ASC_COLUMNS)
    for col in ["r", "g", "b", "classification"]:
        df[col] = df[col].astype(np.int32)
    return df


def associate_points_to_ifc(
    points: pd.DataFrame,
    mesh_npz: str | Path,
    triangle_element_map_path: str | Path,
    elements_path: str | Path,
    output_dir: str | Path,
    association_distance: float = 0.05,
    calibration_points: int = 1000,
    enable_translation_calibration: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = np.load(mesh_npz)
    vertices = mesh["vertices"]
    faces = mesh["faces"]
    tri_map = np.load(triangle_element_map_path)
    elements = pd.read_parquet(elements_path)
    coords = points[["x", "y", "z"]].to_numpy(dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    method = "trimesh.closest_point"
    try:
        import trimesh

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if enable_translation_calibration and len(coords):
            n_cal = min(int(calibration_points), len(coords))
            closest_cal, distances_cal, _ = trimesh.proximity.closest_point(mesh, coords[:n_cal])
            if float(np.median(distances_cal)) > association_distance:
                translation = np.median(coords[:n_cal] - closest_cal, axis=0)
        registered_coords = coords - translation
        _, distances, triangle_indices = trimesh.proximity.closest_point(mesh, registered_coords)
    except Exception as exc:
        method = f"triangle_centroid_kdtree_fallback: {exc}"
        centroids = vertices[faces].mean(axis=1)
        tree = cKDTree(centroids)
        registered_coords = coords - translation
        distances, triangle_indices = tree.query(registered_coords, k=1, workers=-1)
    matched = distances <= association_distance
    element_indices = tri_map[triangle_indices]
    guids = elements["GlobalId"].to_numpy()
    classes = elements["ifc_class"].to_numpy()
    out = points.copy()
    out["registered_x"] = registered_coords[:, 0]
    out["registered_y"] = registered_coords[:, 1]
    out["registered_z"] = registered_coords[:, 2]
    out["nearest_triangle_index"] = triangle_indices.astype(np.int64)
    out["nearest_distance_m"] = distances.astype(np.float64)
    out["matched"] = matched
    out["element_index"] = np.where(matched, element_indices, -1).astype(np.int64)
    out["element_guid"] = np.where(matched, guids[element_indices], "")
    out["ifc_class"] = np.where(matched, classes[element_indices], "unmatched")
    output_path = output_dir / "cras_point_sample_associations.parquet"
    out.to_parquet(output_path, index=False)
    summary = {
        "input_points": int(len(points)),
        "matched_points": int(matched.sum()),
        "unmatched_points": int((~matched).sum()),
        "matched_ratio": float(matched.mean()) if len(points) else 0.0,
        "association_distance_m": float(association_distance),
        "method": method,
        "translation_point_minus_ifc_m": translation.tolist(),
        "mean_nearest_distance_m": float(distances.mean()) if len(points) else None,
        "p95_nearest_distance_m": float(np.quantile(distances, 0.95)) if len(points) else None,
        "output": str(output_path),
    }
    (output_dir / "cras_point_sample_association_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def preprocess_cras_point_sample(config: dict[str, Any], output_dir: str | Path = "data/processed") -> dict[str, Any]:
    max_points = int(config.get("max_points", 200_000))
    association_distance = float(config.get("association_distance", 0.05))
    calibration_points = int(config.get("calibration_points", 1000))
    points = read_cras_asc_sample(config.get("pointcloud_zip", "data/raw/craslabannotated.zip"), max_points=max_points)
    points.to_parquet(Path(output_dir) / "cras_point_sample.parquet", index=False)
    return associate_points_to_ifc(
        points,
        Path(output_dir) / "ifc_mesh.npz",
        Path(output_dir) / "triangle_element_map.npy",
        Path(output_dir) / "ifc_elements.parquet",
        output_dir,
        association_distance=association_distance,
        calibration_points=calibration_points,
    )
