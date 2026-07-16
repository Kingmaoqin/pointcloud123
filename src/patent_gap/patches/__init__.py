"""
Patch segmentation: region-growing on IFC triangles per element.

Each IFC element is segmented into planar/near-planar surface patches by
grouping adjacent triangles whose normals agree within normal_threshold_deg.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _triangle_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    safe = norms.ravel() > 1e-12
    n[safe] /= norms[safe]
    n[~safe] = [0.0, 0.0, 1.0]
    return n


def _triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def _triangle_centroids(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return (vertices[faces[:, 0]] + vertices[faces[:, 1]] + vertices[faces[:, 2]]) / 3.0


def _build_adjacency(local_faces: np.ndarray, n_tris: int) -> list[list[int]]:
    edge_map: dict[tuple[int, int], list[int]] = {}
    for i, f in enumerate(local_faces):
        for a, b in [(int(f[0]), int(f[1])), (int(f[1]), int(f[2])), (int(f[2]), int(f[0]))]:
            key = (min(a, b), max(a, b))
            edge_map.setdefault(key, []).append(i)
    adj: list[list[int]] = [[] for _ in range(n_tris)]
    for neighbors in edge_map.values():
        for u in neighbors:
            for v in neighbors:
                if u != v:
                    adj[u].append(v)
    return adj


def _region_grow(local_normals: np.ndarray, adj: list[list[int]], cos_threshold: float) -> np.ndarray:
    n = len(local_normals)
    labels = np.full(n, -1, dtype=np.int32)
    patch_id = 0
    for start in range(n):
        if labels[start] >= 0:
            continue
        queue = [start]
        labels[start] = patch_id
        seed_normal = local_normals[start]
        head = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            ref_n = local_normals[cur]
            for nb in adj[cur]:
                if labels[nb] >= 0:
                    continue
                if (
                    float(np.dot(ref_n, local_normals[nb])) >= cos_threshold
                    and float(np.dot(seed_normal, local_normals[nb])) >= cos_threshold
                ):
                    labels[nb] = patch_id
                    queue.append(nb)
        patch_id += 1
    return labels


def build_patches(
    mesh_npz_path: str | Path,
    triangle_element_map_path: str | Path,
    elements_parquet_path: str | Path,
    materials_parquet_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Segment each IFC element's triangles into planar surface patches via
    normal-similarity region-growing.  Results are cached to output_path.
    """
    cfg = config or {}
    normal_threshold_deg = float(cfg.get("normal_threshold_deg", 20.0))
    min_patch_area = float(cfg.get("min_patch_area", 0.005))
    cos_threshold = float(np.cos(np.deg2rad(normal_threshold_deg)))

    importance_map: dict[str, float] = cfg.get(
        "engineering_importance",
        {
            "IfcWall": 0.8, "IfcWallStandardCase": 0.8,
            "IfcSlab": 0.9, "IfcColumn": 0.95, "IfcBeam": 0.95,
            "IfcDoor": 0.7, "IfcWindow": 0.6,
            "IfcStairFlight": 0.75, "IfcRoof": 0.85,
            "IfcFurnishingElement": 0.3, "IfcOpeningElement": 0.2,
            "IfcCovering": 0.4, "IfcBuildingElementProxy": 0.5,
        },
    )

    output_path = Path(output_path)
    triangle_patch_map_path = output_path.with_name(f"{output_path.stem}_triangle_map.npy")
    metadata_path = output_path.with_name(f"{output_path.stem}_cache.json")
    source_paths = [
        Path(mesh_npz_path),
        Path(triangle_element_map_path),
        Path(elements_parquet_path),
        Path(materials_parquet_path),
    ]
    fingerprint = {
        "normal_threshold_deg": normal_threshold_deg,
        "min_patch_area": min_patch_area,
        "sources": {
            str(path): {
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in source_paths
        },
    }
    if output_path.exists() and triangle_patch_map_path.exists() and metadata_path.exists():
        try:
            cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = {}
        if cached == fingerprint:
            return pd.read_parquet(output_path)

    mesh = np.load(str(mesh_npz_path))
    vertices: np.ndarray = mesh["vertices"].astype(np.float64)
    faces: np.ndarray = mesh["faces"].astype(np.int64)
    tri_map: np.ndarray = np.load(str(triangle_element_map_path))
    elements = pd.read_parquet(str(elements_parquet_path))
    materials = pd.read_parquet(str(materials_parquet_path))

    mat_id_to_name = (
        {int(r.material_id): str(r.material_name) for r in materials.itertuples()}
        if {"material_id", "material_name"}.issubset(materials.columns)
        else {}
    )

    face_normals = _triangle_normals(vertices, faces)
    face_areas = _triangle_areas(vertices, faces)
    face_centroids = _triangle_centroids(vertices, faces)

    patch_rows: list[dict] = []
    global_patch_id = 0
    triangle_patch_map = np.full(len(faces), -1, dtype=np.int64)

    for elem_row in elements.itertuples():
        elem_idx = int(elem_row.element_index)
        guid = str(elem_row.GlobalId)
        ifc_class = str(elem_row.ifc_class)
        name = str(elem_row.Name)

        local_mask = tri_map == elem_idx
        local_face_indices = np.where(local_mask)[0]
        if len(local_face_indices) == 0:
            continue

        local_faces = faces[local_face_indices]
        local_normals = face_normals[local_face_indices]
        local_areas = face_areas[local_face_indices]
        local_centroids = face_centroids[local_face_indices]

        try:
            mat_ids = json.loads(elem_row.material_ids or "[]")
            mat_names = [mat_id_to_name.get(int(mid), "") for mid in mat_ids] if mat_ids else []
            mat_name: str | None = mat_names[0] if mat_names else None
        except Exception:
            mat_name = None

        eng_importance = importance_map.get(ifc_class, 0.5)
        adj = _build_adjacency(local_faces, len(local_face_indices))
        local_labels = _region_grow(local_normals, adj, cos_threshold)

        for lbl in np.unique(local_labels):
            mask = local_labels == lbl
            p_areas = local_areas[mask]
            total_area = float(p_areas.sum())
            if total_area < min_patch_area:
                continue
            w = p_areas / (total_area + 1e-12)
            centroid = (local_centroids[mask] * w[:, None]).sum(axis=0)
            normal_raw = (local_normals[mask] * w[:, None]).sum(axis=0)
            norm_len = float(np.linalg.norm(normal_raw))
            normal = normal_raw / norm_len if norm_len > 1e-8 else np.array([0.0, 0.0, 1.0])

            local_verts = vertices[local_faces[mask].ravel()]
            bbox_min = local_verts.min(axis=0)
            bbox_max = local_verts.max(axis=0)

            patch_rows.append({
                "patch_id": global_patch_id,
                "element_index": elem_idx,
                "element_guid": guid,
                "ifc_class": ifc_class,
                "name": name,
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "centroid_z": float(centroid[2]),
                "normal_x": float(normal[0]),
                "normal_y": float(normal[1]),
                "normal_z": float(normal[2]),
                "area": total_area,
                "triangle_count": int(mask.sum()),
                "material_name": mat_name,
                "engineering_importance": eng_importance,
                "bbox_min_x": float(bbox_min[0]),
                "bbox_min_y": float(bbox_min[1]),
                "bbox_min_z": float(bbox_min[2]),
                "bbox_max_x": float(bbox_max[0]),
                "bbox_max_y": float(bbox_max[1]),
                "bbox_max_z": float(bbox_max[2]),
                "gt_missing": False,
            })
            triangle_patch_map[local_face_indices[mask]] = global_patch_id
            global_patch_id += 1

    df = pd.DataFrame(patch_rows)
    if df.empty:
        raise RuntimeError("patch segmentation produced no patches")
    df["centroid"] = list(zip(df["centroid_x"], df["centroid_y"], df["centroid_z"]))
    df["normal"] = list(zip(df["normal_x"], df["normal_y"], df["normal_z"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    np.save(triangle_patch_map_path, triangle_patch_map)
    metadata_path.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
    return df
