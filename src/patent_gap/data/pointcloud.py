from __future__ import annotations

import json
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

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


def iter_cras_asc_chunks(
    zip_path: str | Path,
    chunk_size: int,
    max_points: int | None = None,
    start_chunk: int = 0,
) -> Iterator[tuple[int, pd.DataFrame]]:
    zip_path = Path(zip_path)
    chunk_index = 0
    rows: list[list[float]] = []
    yielded_points = 0
    with zipfile.ZipFile(zip_path) as zf:
        asc_names = [n for n in zf.namelist() if n.lower().endswith(".asc")]
        if not asc_names:
            raise FileNotFoundError(f"no ASC file found in {zip_path}")
        with zf.open(asc_names[0]) as f:
            for raw in f:
                if max_points is not None and yielded_points >= max_points:
                    break
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
                if len(rows) >= chunk_size:
                    take = rows
                    rows = []
                    if chunk_index >= start_chunk:
                        df = pd.DataFrame(take, columns=ASC_COLUMNS)
                        for col in ["r", "g", "b", "classification"]:
                            df[col] = df[col].astype(np.int32)
                        yield chunk_index, df
                    yielded_points += len(take)
                    chunk_index += 1
            if rows and (max_points is None or yielded_points < max_points):
                if max_points is not None:
                    remaining = max_points - yielded_points
                    rows = rows[:remaining]
                if chunk_index >= start_chunk:
                    df = pd.DataFrame(rows, columns=ASC_COLUMNS)
                    for col in ["r", "g", "b", "classification"]:
                        df[col] = df[col].astype(np.int32)
                    yield chunk_index, df


def associate_points_to_ifc(
    points: pd.DataFrame,
    mesh_npz: str | Path,
    triangle_element_map_path: str | Path,
    elements_path: str | Path,
    output_dir: str | Path,
    association_distance: float = 0.05,
    calibration_points: int = 1000,
    enable_translation_calibration: bool = True,
    backend: str = "open3d",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = np.load(mesh_npz)
    vertices = mesh["vertices"]
    faces = mesh["faces"]
    tri_map = np.load(triangle_element_map_path)
    elements = pd.read_parquet(elements_path)
    coords = points[["x", "y", "z"]].to_numpy(dtype=np.float64)
    engine = build_proximity_engine(vertices, faces, backend=backend)
    translation = estimate_translation(engine, coords, association_distance, calibration_points, enable_translation_calibration)
    registered_coords = coords - translation
    distances, triangle_indices = query_proximity(engine, registered_coords)
    method = engine["method"]
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
    backend = str(config.get("association_backend", "open3d"))
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
        backend=backend,
    )


def build_proximity_engine(vertices: np.ndarray, faces: np.ndarray, backend: str = "open3d") -> dict[str, Any]:
    if backend == "open3d":
        try:
            import open3d as o3d

            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(
                o3d.core.Tensor(vertices.astype(np.float32), dtype=o3d.core.Dtype.Float32),
                o3d.core.Tensor(faces.astype(np.uint32), dtype=o3d.core.Dtype.UInt32),
            )
            return {"backend": "open3d", "method": "open3d.compute_closest_points", "scene": scene}
        except Exception:
            pass
    try:
        import trimesh

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        return {"backend": "trimesh", "method": "trimesh.closest_point", "mesh": mesh}
    except Exception as exc:
        centroids = vertices[faces].mean(axis=1)
        return {"backend": "centroid", "method": f"triangle_centroid_kdtree_fallback: {exc}", "tree": cKDTree(centroids)}


def query_proximity(engine: dict[str, Any], coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if engine["backend"] == "open3d":
        import open3d as o3d

        tensor = o3d.core.Tensor(coords.astype(np.float32), dtype=o3d.core.Dtype.Float32)
        answer = engine["scene"].compute_closest_points(tensor)
        closest = answer["points"].numpy().astype(np.float64)
        triangle_indices = answer["primitive_ids"].numpy().astype(np.int64)
        distances = np.linalg.norm(coords - closest, axis=1)
        return distances, triangle_indices
    if engine["backend"] == "trimesh":
        _, distances, triangle_indices = engine["mesh"].nearest.on_surface(coords)
        return distances.astype(np.float64), triangle_indices.astype(np.int64)
    distances, triangle_indices = engine["tree"].query(coords, k=1, workers=-1)
    return distances.astype(np.float64), triangle_indices.astype(np.int64)


def estimate_translation(
    engine: dict[str, Any],
    coords: np.ndarray,
    association_distance: float,
    calibration_points: int,
    enable_translation_calibration: bool,
) -> np.ndarray:
    translation = np.zeros(3, dtype=np.float64)
    if not enable_translation_calibration or len(coords) == 0:
        return translation
    n_cal = min(int(calibration_points), len(coords))
    cal = coords[:n_cal]
    if engine["backend"] == "open3d":
        import open3d as o3d

        answer = engine["scene"].compute_closest_points(o3d.core.Tensor(cal.astype(np.float32), dtype=o3d.core.Dtype.Float32))
        closest = answer["points"].numpy().astype(np.float64)
        distances = np.linalg.norm(cal - closest, axis=1)
    elif engine["backend"] == "trimesh":
        closest, distances, _ = engine["mesh"].nearest.on_surface(cal)
    else:
        distances, _ = engine["tree"].query(cal, k=1, workers=-1)
        closest = cal
    if float(np.median(distances)) > association_distance and engine["backend"] != "centroid":
        translation = np.median(cal - closest, axis=0)
    return translation


def associate_cras_stream(
    config: dict[str, Any],
    output_dir: str | Path = "data/processed",
    run_name: str = "cras_full_assoc",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    run_dir = output_dir / run_name
    parts_dir = run_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = int(config.get("chunk_size", 1_000_000))
    max_points_raw = config.get("max_points")
    max_points = None if max_points_raw in (None, "null", "None", -1, 0) else int(max_points_raw)
    association_distance = float(config.get("association_distance", 0.05))
    calibration_points = int(config.get("calibration_points", 5000))
    backend = str(config.get("association_backend", "open3d"))
    write_parts = bool(config.get("write_point_parts", False))
    resume = bool(config.get("resume", True))
    if not resume:
        for old_file in parts_dir.glob("chunk_*.summary.json"):
            old_file.unlink()
        for old_file in parts_dir.glob("chunk_*.parquet"):
            old_file.unlink()

    existing_summaries = sorted(parts_dir.glob("chunk_*.summary.json")) if resume else []
    completed_chunks = {int(p.stem.split("_")[1].split(".")[0]) for p in existing_summaries}
    start_chunk = max(completed_chunks) + 1 if completed_chunks else 0

    mesh = np.load(output_dir / "ifc_mesh.npz")
    vertices = mesh["vertices"]
    faces = mesh["faces"]
    tri_map = np.load(output_dir / "triangle_element_map.npy")
    elements = pd.read_parquet(output_dir / "ifc_elements.parquet")
    guids = elements["GlobalId"].to_numpy()
    classes = elements["ifc_class"].to_numpy()
    engine = build_proximity_engine(vertices, faces, backend=backend)

    translation: np.ndarray | None = None
    aggregate = load_existing_stream_aggregate(existing_summaries)
    run_started = time.time()
    for chunk_index, points in iter_cras_asc_chunks(config.get("pointcloud_zip", "data/raw/craslabannotated.zip"), chunk_size, max_points, start_chunk):
        chunk_started = time.time()
        coords = points[["x", "y", "z"]].to_numpy(dtype=np.float64)
        if translation is None:
            translation = estimate_translation(engine, coords, association_distance, calibration_points, True)
        registered = coords - translation
        distances, triangle_indices = query_proximity(engine, registered)
        matched = distances <= association_distance
        element_indices = tri_map[triangle_indices]
        update_aggregate(aggregate, points, distances, matched, element_indices, classes)

        summary = {
            "chunk_index": int(chunk_index),
            "points": int(len(points)),
            "matched_points": int(matched.sum()),
            "matched_ratio": float(matched.mean()) if len(points) else 0.0,
            "distance_sum": float(distances.sum()),
            "distance_count": int(len(distances)),
            "mean_nearest_distance_m": float(distances.mean()) if len(points) else None,
            "p95_nearest_distance_m": float(np.quantile(distances, 0.95)) if len(points) else None,
            "translation_point_minus_ifc_m": translation.tolist(),
            "method": engine["method"],
            "ifc_class_counts": dict(Counter(np.where(matched, classes[element_indices], "unmatched").tolist()).most_common()),
            "classification_counts": dict(Counter(points["classification"].astype(int).astype(str).tolist()).most_common()),
            "element_counts": {str(int(k)): int(v) for k, v in Counter(element_indices[matched].astype(int).tolist()).items()},
            "runtime_s": float(time.time() - chunk_started),
        }
        (parts_dir / f"chunk_{chunk_index:06d}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if write_parts:
            out = points.copy()
            out["nearest_distance_m"] = distances.astype(np.float32)
            out["matched"] = matched
            out["element_index"] = np.where(matched, element_indices, -1).astype(np.int64)
            out["element_guid"] = np.where(matched, guids[element_indices], "")
            out["ifc_class"] = np.where(matched, classes[element_indices], "unmatched")
            out.to_parquet(parts_dir / f"chunk_{chunk_index:06d}.parquet", index=False)
        write_stream_summary(run_dir, aggregate, association_distance, engine["method"], translation, started_at=run_started)

    return write_stream_summary(run_dir, aggregate, association_distance, engine["method"], translation, started_at=run_started)


def load_existing_stream_aggregate(summary_files: list[Path]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "points": 0,
        "matched_points": 0,
        "distance_sum": 0.0,
        "distance_count": 0,
        "ifc_class_counts": Counter(),
        "classification_counts": Counter(),
        "element_counts": defaultdict(int),
    }
    for path in summary_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        aggregate["points"] += int(data.get("points", 0))
        aggregate["matched_points"] += int(data.get("matched_points", 0))
        aggregate["distance_sum"] += float(data.get("distance_sum", 0.0))
        aggregate["distance_count"] += int(data.get("distance_count", 0))
        aggregate["ifc_class_counts"].update(data.get("ifc_class_counts", {}))
        aggregate["classification_counts"].update(data.get("classification_counts", {}))
        for key, value in data.get("element_counts", {}).items():
            aggregate["element_counts"][key] += int(value)
    return aggregate


def update_aggregate(
    aggregate: dict[str, Any],
    points: pd.DataFrame,
    distances: np.ndarray,
    matched: np.ndarray,
    element_indices: np.ndarray,
    classes: np.ndarray,
) -> None:
    aggregate["points"] += int(len(points))
    aggregate["matched_points"] += int(matched.sum())
    aggregate["distance_sum"] += float(distances.sum())
    aggregate["distance_count"] += int(len(distances))
    aggregate["classification_counts"].update(points["classification"].astype(int).astype(str).tolist())
    matched_classes = np.where(matched, classes[element_indices], "unmatched")
    aggregate["ifc_class_counts"].update(matched_classes.tolist())
    for idx in element_indices[matched]:
        aggregate["element_counts"][str(int(idx))] += 1


def write_stream_summary(
    run_dir: Path,
    aggregate: dict[str, Any],
    association_distance: float,
    method: str,
    translation: np.ndarray | None,
    started_at: float | None = None,
) -> dict[str, Any]:
    points = int(aggregate["points"])
    matched = int(aggregate["matched_points"])
    summary = {
        "points": points,
        "matched_points": matched,
        "unmatched_points": points - matched,
        "matched_ratio": matched / points if points else 0.0,
        "association_distance_m": association_distance,
        "method": method,
        "translation_point_minus_ifc_m": None if translation is None else translation.tolist(),
        "mean_nearest_distance_m": aggregate["distance_sum"] / aggregate["distance_count"] if aggregate["distance_count"] else None,
        "ifc_class_counts": dict(aggregate["ifc_class_counts"].most_common()),
        "classification_counts": dict(aggregate["classification_counts"].most_common()),
    }
    if started_at is not None:
        runtime_s = float(time.time() - started_at)
        summary["runtime_s"] = runtime_s
        summary["points_per_second"] = points / runtime_s if runtime_s > 0 else None
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(
        [{"element_index": int(k), "matched_points": int(v)} for k, v in aggregate["element_counts"].items()]
    ).to_csv(run_dir / "element_counts.csv", index=False)
    return summary
