from __future__ import annotations

import hashlib
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from patent_gap.evidence import (
    compute_controlled_withheld_evidence,
    compute_evidence,
    compute_evidence_from_synthetic,
)
from patent_gap.gap.scoring import compute_patch_scores, rank_components
from patent_gap.ifc.reader import triangulate_ifc
from patent_gap.materials import build_material_table
from patent_gap.patches import build_patches
from patent_gap.semantics import (
    compute_controlled_semantic_scores,
    compute_semantic_scores,
    compute_semantic_scores_synthetic,
)
from patent_gap.simulation.closed_loop import apply_supplemental_observation
from patent_gap.viewpoints.ranking import generate_candidates, score_candidates
from patent_gap.visualization.interactive import (
    fig_before_after_mesh,
    fig_candidate_views_3d,
    fig_closed_loop_curve,
    fig_component_gap_bar,
    fig_gap_components_heatmap,
    fig_ifc_gap_mesh,
)


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
WEB_CACHE = OUTPUTS / "web_cache"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_VERTICES = 5_000_000
MAX_FACES = 5_000_000
SUPPORTED_MODEL_SUFFIXES = {".ifc", ".npz", ".obj", ".ply", ".stl", ".glb"}
MAX_SESSION_STATES = 32
_STATE_STORE: dict[str, dict[str, Any]] = {}
_STATE_LOCK = threading.Lock()

DEFAULT_GAP_CONFIG = {
    "alpha_sem": 0.16,
    "alpha_mat_missing": 0.14,
    "alpha_mat_conflict": 0.10,
    "alpha_obs": 0.22,
    "alpha_ang": 0.16,
    "alpha_geo": 0.22,
    "lambda_importance": 0.30,
    "high_gap_threshold": 0.55,
}


CSS = """
* { box-sizing: border-box; }
html, body { overflow-x: hidden; }
.gradio-container { max-width: none !important; width: 100% !important; margin: 0 auto !important; padding: 18px 24px !important; }
#app-title h1 { font-size: 24px !important; line-height: 1.2 !important; margin: 0 !important; }
#app-title p { margin: 4px 0 0 !important; color: #53606d !important; }
#workspace-row { flex-wrap: nowrap !important; align-items: flex-start !important; }
.control-panel { border-right: 1px solid #d8dee5; padding-right: 16px; min-width: 300px !important; max-width: 360px !important; }
.visual-workspace { min-width: 0 !important; width: 100% !important; }
.visual-workspace .tabitem { min-width: 0 !important; overflow: hidden; }
.metric-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
.metric-item { border: 1px solid #d8dee5; border-radius: 6px; padding: 9px 10px; background: #fff; }
.metric-label { color: #66727e; font-size: 12px; }
.metric-value { color: #17212b; font-size: 19px; font-weight: 650; margin-top: 2px; }
.status-line { margin-top: 8px; padding: 8px 10px; border-left: 3px solid #188977; background: #f3f8f7; }
@media (max-width: 900px) {
  .gradio-container { padding: 12px !important; }
  #workspace-row { display: block !important; width: 100% !important; }
  #workspace-row > * { width: 100% !important; max-width: 100% !important; min-width: 0 !important; flex: none !important; }
  .control-panel { border-right: 0; padding-right: 0; border-bottom: 1px solid #d8dee5; padding-bottom: 12px; }
  .control-panel, .visual-workspace { min-width: 0 !important; max-width: none !important; width: 100% !important; }
  .metric-grid { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
}
"""


def _file_path(value: Any) -> Path:
    if value is None:
        raise ValueError("请选择模型文件")
    raw_path = value if isinstance(value, (str, Path)) else getattr(value, "name", value)
    path = Path(raw_path).resolve()
    if not path.exists() or not path.is_file():
        raise ValueError("上传文件不存在")
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError("上传文件超过 512 MB 限制")
    return path


def _sha256_prefix(path: Path, length: int = 16) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def _save_state(state: dict[str, Any], state_key: str | None = None) -> str:
    key = state_key or uuid.uuid4().hex
    state["_last_access"] = time.monotonic()
    with _STATE_LOCK:
        _STATE_STORE[key] = state
        if len(_STATE_STORE) > MAX_SESSION_STATES:
            oldest = min(
                _STATE_STORE,
                key=lambda item: float(_STATE_STORE[item].get("_last_access", 0.0)),
            )
            if oldest != key:
                _STATE_STORE.pop(oldest, None)
    return key


def _get_state(state_key: str | None) -> dict[str, Any]:
    if not state_key:
        raise ValueError("请先加载项目或导入模型")
    with _STATE_LOCK:
        state = _STATE_STORE.get(str(state_key))
    if state is None:
        raise ValueError("当前会话已过期，请重新加载项目")
    state["_last_access"] = time.monotonic()
    return state


def _validate_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("模型 vertices 必须是 N x 3")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("模型 faces 必须是 M x 3 三角面")
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("模型没有可渲染的三角网格")
    if len(vertices) > MAX_VERTICES or len(faces) > MAX_FACES:
        raise ValueError("模型规模超过 500 万顶点/三角面限制")
    if not np.isfinite(vertices).all():
        raise ValueError("模型顶点包含 NaN 或无穷值")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("模型面索引越界")
    return vertices, faces


def _load_generic_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as mesh:
            if "vertices" not in mesh or "faces" not in mesh:
                raise ValueError("NPZ 必须包含 vertices 和 faces")
            return _validate_mesh(mesh["vertices"], mesh["faces"])

    import trimesh

    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces)
        ]
        if not geometries:
            raise ValueError("模型场景中没有三角网格")
        loaded = trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError("无法读取该模型格式")
    return _validate_mesh(np.asarray(loaded.vertices), np.asarray(loaded.faces))


def _mesh_area_normal_centroid(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[float, tuple[float, float, float], tuple[float, float, float]]:
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    area = double_area * 0.5
    total = float(area.sum())
    weights = area / max(total, 1e-12)
    centroid = (triangles.mean(axis=1) * weights[:, None]).sum(axis=0)
    normal_raw = (cross * weights[:, None]).sum(axis=0)
    norm = float(np.linalg.norm(normal_raw))
    normal = normal_raw / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])
    return total, tuple(centroid.tolist()), tuple(normal.tolist())


def _scene_from_gap_proxy(patches: pd.DataFrame, proxy_scores: pd.Series) -> dict[str, pd.DataFrame]:
    proxy = pd.to_numeric(proxy_scores, errors="coerce").fillna(0.5).clip(0, 1)
    patch_ids = patches["patch_id"].to_numpy()
    observations = pd.DataFrame(
        {
            "patch_id": patch_ids,
            "directly_observed": proxy < 0.5,
            "number_of_views": (proxy < 0.5).astype(int),
            "number_of_valid_views": (proxy < 0.5).astype(int),
            "best_frontality": 1.0 - proxy,
            "mean_frontality": 1.0 - proxy,
            "angular_diversity": 1.0 - proxy,
            "coverage_ratio": 1.0 - proxy,
            "density_ratio": 1.0 - proxy,
            "projected_resolution_quality": 1.0 - proxy,
            "registration_confidence": np.full(len(patches), 0.9),
            "source_type": np.where(proxy < 0.5, "MEASURED", "MODEL"),
            "D_obs": proxy,
            "D_ang": proxy,
            "D_geo": proxy,
        }
    )
    semantics = pd.DataFrame({"patch_id": patch_ids, "D_sem": np.nan})
    materials = pd.DataFrame(
        {
            "patch_id": patch_ids,
            "D_mat_missing": np.nan,
            "D_mat_conflict": np.nan,
            "material_name": patches.get("material_name"),
            "visual_material_category": None,
        }
    )
    return {
        "patches": patches,
        "observations": observations,
        "semantics": semantics,
        "materials": materials,
    }


def _apply_score_file(patches: pd.DataFrame, score_file: Any) -> tuple[pd.Series, str]:
    if score_file is None:
        return pd.Series(0.65, index=patches.index), "未提供评分文件，使用 0.65 初始缺口代理"
    path = _file_path(score_file)
    if path.suffix.lower() != ".csv":
        raise ValueError("评分文件必须是 CSV")
    scores = pd.read_csv(path)
    if "G_gap" not in scores.columns:
        raise ValueError("评分 CSV 必须包含 G_gap")
    for key in ("patch_id", "element_guid", "element_index"):
        if key in scores.columns and key in patches.columns:
            merged = patches[[key]].merge(
                scores[[key, "G_gap"]].drop_duplicates(key),
                on=key,
                how="left",
                validate="many_to_one",
            )
            return merged["G_gap"].fillna(0.65), f"按 {key} 导入评分"
    if len(scores) == len(patches):
        return scores["G_gap"].reset_index(drop=True), "按行顺序导入评分"
    return (
        pd.Series(float(pd.to_numeric(scores["G_gap"], errors="coerce").mean()), index=patches.index),
        "评分无法逐项匹配，使用 CSV 的 G_gap 均值",
    )


def _base_state(
    source: str,
    scene: dict[str, pd.DataFrame],
    mesh_path: Path,
    tri_map: np.ndarray,
    elements: pd.DataFrame,
    gap_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = {**DEFAULT_GAP_CONFIG, **(gap_config or {})}
    scores = compute_patch_scores(scene, config)
    target_ids = set(scores.loc[scores["gt_missing"], "patch_id"].astype(int))
    target_basis = "gt_missing" if target_ids else "all_patches"
    if not target_ids:
        target_ids = set(scores["patch_id"].astype(int))
    initial_gap_area = float(
        (
            scores.loc[scores["patch_id"].isin(target_ids), "G_gap"]
            * scores.loc[scores["patch_id"].isin(target_ids), "area"]
        ).sum()
    )
    return {
        "source": source,
        "scene": scene,
        "scores": scores,
        "initial_scores": scores.copy(deep=True),
        "mesh_path": str(mesh_path),
        "tri_map": np.asarray(tri_map, dtype=np.int64),
        "elements": elements,
        "gap_config": config,
        "target_ids": target_ids,
        "target_basis": target_basis,
        "initial_gap_area": initial_gap_area,
        "history": [],
        "selected_view_ids": [],
        "ranked": pd.DataFrame(),
        "components": rank_components(scores, config),
        "message": "",
    }


def _load_bundled_state(source: str) -> dict[str, Any]:
    required = [
        PROCESSED / "ifc_mesh.npz",
        PROCESSED / "triangle_element_map.npy",
        PROCESSED / "ifc_elements.parquet",
        PROCESSED / "ifc_materials.parquet",
        PROCESSED / "patches_real.parquet",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少项目数据: {', '.join(str(path) for path in missing)}")

    elements = pd.read_parquet(PROCESSED / "ifc_elements.parquet")
    materials = pd.read_parquet(PROCESSED / "ifc_materials.parquet")
    patches = pd.read_parquet(PROCESSED / "patches_real.parquet")
    mesh_path = PROCESSED / "ifc_mesh.npz"
    tri_map = np.load(PROCESSED / "triangle_element_map.npy")

    if source in {"synthetic", "raycast"}:
        synth_dir = PROCESSED / "synthetic_scan"
        elem_stats = pd.read_csv(synth_dir / "element_stats.csv")
        patch_stats_path = synth_dir / "patch_stats.csv"
        patch_stats = pd.read_csv(patch_stats_path) if patch_stats_path.exists() else None
        patches = patches.copy()
        patches["gt_missing"] = patches["centroid_y"] > 14.0
        if source == "synthetic":
            observations = compute_controlled_withheld_evidence(patches, seed=42)
            semantics = compute_controlled_semantic_scores(patches, seed=42)
            source_label = "合成留出实验"
            message = "已加载 Y>14m 预声明留出区；标签与观测证据独立"
        else:
            observations = compute_evidence_from_synthetic(
                patches,
                elements,
                elem_stats,
                patch_stats_df=patch_stats,
            )
            stats = patch_stats if patch_stats is not None else elem_stats
            semantics = compute_semantic_scores_synthetic(patches, stats)
            source_label = "射线诊断"
            message = "已加载真实射线可见性诊断；不作为独立性能评估"
        material_table = build_material_table(patches, elements, materials, assoc_df=None)
        scene = {
            "patches": patches,
            "observations": observations,
            "semantics": semantics,
            "materials": material_table,
        }
        state = _base_state(source_label, scene, mesh_path, tri_map, elements)
        state["message"] = message
        return state

    assoc_path = PROCESSED / "cras_point_sample_associations.parquet"
    if not assoc_path.exists():
        raise FileNotFoundError(f"缺少 {assoc_path}")
    assoc = pd.read_parquet(assoc_path)
    observations = compute_evidence(
        patches,
        elements,
        assoc,
        chunk_dir=PROCESSED / "cras_full_assoc",
    )
    semantics = compute_semantic_scores(patches, assoc)
    material_table = build_material_table(patches, elements, materials, assoc)
    scene = {
        "patches": patches,
        "observations": observations,
        "semantics": semantics,
        "materials": material_table,
    }
    state = _base_state("CRAS 实际关联", scene, mesh_path, tri_map, elements)
    state["message"] = "已加载 CRAS 全量关联结果；无扫描站位时 D_ang 保持未知"
    return state


def _load_uploaded_state(model_file: Any, score_file: Any) -> dict[str, Any]:
    model_path = _file_path(model_file)
    suffix = model_path.suffix.lower()
    if suffix not in SUPPORTED_MODEL_SUFFIXES:
        raise ValueError(f"不支持 {suffix}；支持 IFC/NPZ/OBJ/PLY/STL/GLB")
    digest = _sha256_prefix(model_path)
    cache_dir = WEB_CACHE / digest
    cache_dir.mkdir(parents=True, exist_ok=True)

    if suffix == ".ifc":
        result = triangulate_ifc(model_path, cache_dir)
        if not result.get("triangulated"):
            raise ValueError(f"IFC 三角化失败: {result.get('error', 'unknown error')}")
        mesh_path = cache_dir / "ifc_mesh.npz"
        elements = pd.read_parquet(cache_dir / "ifc_elements.parquet")
        tri_map = np.load(cache_dir / "triangle_element_map.npy")
        patches = build_patches(
            mesh_path,
            cache_dir / "triangle_element_map.npy",
            cache_dir / "ifc_elements.parquet",
            cache_dir / "ifc_materials.parquet",
            cache_dir / "patches.parquet",
            config={"min_patch_area": 0.01},
        )
        proxy, score_message = _apply_score_file(patches, score_file)
        scene = _scene_from_gap_proxy(patches, proxy)
        state = _base_state(model_path.name, scene, mesh_path, tri_map, elements)
        state["message"] = f"IFC 导入完成；{score_message}"
        return state

    vertices, faces = _load_generic_mesh(model_path)
    mesh_path = cache_dir / "mesh.npz"
    np.savez_compressed(mesh_path, vertices=vertices, faces=faces)
    total_area, centroid, normal = _mesh_area_normal_centroid(vertices, faces)
    patches = pd.DataFrame(
        [
            {
                "patch_id": 0,
                "element_index": 0,
                "element_guid": digest,
                "ifc_class": "ImportedMesh",
                "name": model_path.stem,
                "centroid": centroid,
                "normal": normal,
                "area": max(total_area, 1e-6),
                "engineering_importance": 0.5,
                "material_name": None,
                "gt_missing": False,
            }
        ]
    )
    proxy, score_message = _apply_score_file(patches, score_file)
    scene = _scene_from_gap_proxy(patches, proxy)
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    elements = pd.DataFrame(
        [
            {
                "element_index": 0,
                "GlobalId": digest,
                "ifc_class": "ImportedMesh",
                "Name": model_path.stem,
                "bbox_min_x": bounds_min[0],
                "bbox_min_y": bounds_min[1],
                "bbox_min_z": bounds_min[2],
                "bbox_max_x": bounds_max[0],
                "bbox_max_y": bounds_max[1],
                "bbox_max_z": bounds_max[2],
            }
        ]
    )
    state = _base_state(
        model_path.name,
        scene,
        mesh_path,
        np.zeros(len(faces), dtype=np.int64),
        elements,
    )
    state["message"] = f"模型导入完成；{score_message}"
    return state


def _rank_state(state: dict[str, Any], threshold: float) -> dict[str, Any]:
    view_config = {
        "gap_threshold": float(threshold),
        "max_target_patches": 30,
        "field_of_view_deg": 100.0,
        "max_range_m": 30.0,
        "distances": [1.5, 2.5, 4.0],
        "azimuth_offsets_deg": [-45, 0, 45],
        "elevation_offsets_deg": [-20, 0, 20],
        "eta": 0.10,
    }
    candidates = generate_candidates(state["scores"], view_config)
    ranked = score_candidates(state["scores"], candidates, view_config)
    selected = set(state.get("selected_view_ids", []))
    if selected and not ranked.empty:
        ranked = ranked[~ranked["view_id"].isin(selected)].reset_index(drop=True)
    state["ranked"] = ranked
    state["components"] = rank_components(state["scores"], state["gap_config"])
    state["view_config"] = view_config
    return state


def _gap_area(state: dict[str, Any], scores: pd.DataFrame) -> float:
    target = scores["patch_id"].isin(state["target_ids"])
    return float((scores.loc[target, "G_gap"] * scores.loc[target, "area"]).sum())


def _status_html(state: dict[str, Any], threshold: float) -> str:
    scores = state["scores"]
    mean_gap = float(scores["G_gap"].mean())
    high_count = int((scores["G_gap"] >= threshold).sum())
    current_area = _gap_area(state, scores)
    initial_area = float(state["initial_gap_area"])
    recovery = 0.0 if initial_area <= 0 else max(0.0, 1.0 - current_area / initial_area)
    message = state.get("message", "")
    return f"""
    <div class="metric-grid">
      <div class="metric-item"><div class="metric-label">Patch</div><div class="metric-value">{len(scores):,}</div></div>
      <div class="metric-item"><div class="metric-label">构件</div><div class="metric-value">{scores['element_guid'].nunique():,}</div></div>
      <div class="metric-item"><div class="metric-label">平均 G_gap</div><div class="metric-value">{mean_gap:.3f}</div></div>
      <div class="metric-item"><div class="metric-label">高缺口 Patch</div><div class="metric-value">{high_count:,}</div></div>
      <div class="metric-item"><div class="metric-label">已选视角</div><div class="metric-value">{len(state.get('selected_view_ids', []))}</div></div>
      <div class="metric-item"><div class="metric-label">缺口恢复</div><div class="metric-value">{recovery:.1%}</div></div>
    </div>
    <div class="status-line"><b>{state['source']}</b><br>{message}</div>
    """


def _mesh_figure(state: dict[str, Any], scores: pd.DataFrame, title: str) -> go.Figure:
    fig = fig_ifc_gap_mesh(
        state["mesh_path"],
        state["tri_map"],
        scores,
        state["elements"],
        max_faces=30_000,
    )
    fig.update_layout(title=title, height=620)
    return fig


def _comparison_figure(state: dict[str, Any], threshold: float = 0.55) -> go.Figure:
    return fig_before_after_mesh(
        state["mesh_path"],
        state["tri_map"],
        state["initial_scores"],
        state["scores"],
        state["elements"],
        threshold=threshold,
        max_faces=25_000,
    )


def _view_table(state: dict[str, Any], top_k: int) -> pd.DataFrame:
    ranked = state["ranked"].head(int(top_k)).copy()
    if ranked.empty:
        return pd.DataFrame(
            columns=["rank", "view_id", "target_patch_id", "value", "visible_gap_area", "mean_quality", "position"]
        )
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    ranked["position"] = ranked["position"].map(str)
    columns = [
        "rank",
        "view_id",
        "target_patch_id",
        "value",
        "visible_gap_area",
        "mean_quality",
        "position",
    ]
    return ranked[columns].round({"value": 4, "visible_gap_area": 3, "mean_quality": 4})


def _component_table(state: dict[str, Any]) -> pd.DataFrame:
    columns = [
        "element_guid",
        "ifc_class",
        "G_component",
        "mean_gap",
        "max_gap",
        "high_gap_area_ratio",
    ]
    available = [column for column in columns if column in state["components"].columns]
    return state["components"].head(30)[available].round(4)


def _history_df(state: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(state.get("history", []))


def _dashboard_values(
    state: dict[str, Any],
    threshold: float,
    top_k: int,
    state_key: str | None = None,
) -> tuple[Any, ...]:
    state = _rank_state(state, threshold)
    state_key = _save_state(state, state_key)
    history = _history_df(state)
    return (
        state_key,
        _status_html(state, threshold),
        _mesh_figure(state, state["scores"], "BIM 信息缺口热力图"),
        fig_candidate_views_3d(state["ranked"], state["scores"], top_k=int(top_k)),
        _comparison_figure(state, threshold),
        fig_gap_components_heatmap(state["scores"], max_patches=60),
        fig_component_gap_bar(state["components"]),
        fig_closed_loop_curve(history),
        _view_table(state, top_k),
        _component_table(state),
        history,
    )


def load_source(source: str, threshold: float, top_k: int) -> tuple[Any, ...]:
    try:
        state = _load_bundled_state(source)
        return _dashboard_values(state, threshold, top_k)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def import_model(
    model_file: Any,
    score_file: Any,
    threshold: float,
    top_k: int,
) -> tuple[Any, ...]:
    try:
        state = _load_uploaded_state(model_file, score_file)
        return _dashboard_values(state, threshold, top_k)
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def refresh_recommendations(
    state_key: str | None,
    threshold: float,
    top_k: int,
) -> tuple[Any, ...]:
    try:
        state = _get_state(state_key)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    state["message"] = "候选补扫视角已重新计算"
    return _dashboard_values(state, threshold, top_k, state_key)


def run_scan_steps(
    state_key: str | None,
    recovery_per_view: float,
    threshold: float,
    top_k: int,
    steps: int,
) -> tuple[Any, ...]:
    try:
        state = _get_state(state_key)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    steps = int(steps)
    if steps <= 0:
        raise gr.Error("补扫步数必须大于 0")

    for _ in range(steps):
        state = _rank_state(state, threshold)
        if state["ranked"].empty:
            state["message"] = "没有剩余候选视角"
            break
        before_scores = state["scores"]
        before_area = _gap_area(state, before_scores)
        best = state["ranked"].iloc[0]
        visible = {
            int(value)
            for value in str(best.get("visible_patch_ids", "")).split(";")
            if value.strip()
        }
        if not visible:
            state["selected_view_ids"].append(str(best["view_id"]))
            continue
        state["scene"]["observations"] = apply_supplemental_observation(
            state["scene"]["observations"],
            visible,
            before_scores,
            float(recovery_per_view),
        )
        state["selected_view_ids"].append(str(best["view_id"]))
        state["scores"] = compute_patch_scores(state["scene"], state["gap_config"])
        after_area = _gap_area(state, state["scores"])
        initial = float(state["initial_gap_area"])
        recovery = 0.0 if initial <= 0 else float(np.clip(1.0 - after_area / initial, 0, 1))
        state["history"].append(
            {
                "step": len(state["history"]) + 1,
                "selected_view_id": str(best["view_id"]),
                "remaining_gap_area_before": before_area,
                "remaining_gap_area": after_area,
                "recovered_missing_area": max(0.0, initial - after_area),
                "recovery_rate": recovery,
                "visible_patch_count": len(visible),
                "view_value": float(best.get("value", 0.0)),
            }
        )
        state["message"] = (
            f"已执行视角 {best['view_id']}，覆盖 {len(visible)} 个 patch"
        )
    return _dashboard_values(state, threshold, top_k, state_key)


def run_one_scan(
    state_key: str | None,
    recovery_per_view: float,
    threshold: float,
    top_k: int,
) -> tuple[Any, ...]:
    return run_scan_steps(state_key, recovery_per_view, threshold, top_k, 1)


def build_app() -> gr.Blocks:
    # Start with empty plots so the browser loads fast; data loads on first button click.
    initial_values = (
        None,
        '<div class="status-line">请点击 <b>加载项目</b> 开始分析。</div>',
        None,
        None,
        None,
        None,
        None,
        None,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )

    theme = gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="orange",
        neutral_hue="slate",
    )
    with gr.Blocks(
        css=CSS,
        title="BIM 信息缺口与补扫工作台",
        theme=theme,
        fill_width=True,
    ) as demo:
        gr.Markdown(
            "# BIM 信息缺口与补扫工作台\nCRAS IFC / 自定义模型",
            elem_id="app-title",
        )
        app_state = gr.State(value=initial_values[0])

        with gr.Row(elem_id="workspace-row"):
            with gr.Column(scale=1, min_width=300, elem_classes="control-panel"):
                source = gr.Dropdown(
                    choices=[
                        ("合成留出实验", "synthetic"),
                        ("射线可见性诊断", "raycast"),
                        ("CRAS 实际关联", "real"),
                    ],
                    value="synthetic",
                    label="数据源",
                )
                load_button = gr.Button("加载项目", variant="primary")

                with gr.Accordion("模型导入", open=False):
                    model_file = gr.File(
                        label="模型",
                        file_types=[".ifc", ".npz", ".obj", ".ply", ".stl", ".glb"],
                        type="filepath",
                    )
                    score_file = gr.File(
                        label="Patch 评分 CSV",
                        file_types=[".csv"],
                        type="filepath",
                    )
                    import_button = gr.Button("导入模型")

                threshold = gr.Slider(
                    0.0,
                    1.0,
                    value=0.55,
                    step=0.01,
                    label="高缺口阈值",
                )
                top_k = gr.Slider(1, 20, value=8, step=1, label="显示候选视角")
                recovery = gr.Slider(
                    0.05,
                    1.0,
                    value=0.55,
                    step=0.05,
                    label="单次补扫恢复率",
                )
                batch_steps = gr.Slider(1, 10, value=5, step=1, label="连续补扫步数")

                refresh_button = gr.Button("重新推荐视角")
                with gr.Row():
                    scan_button = gr.Button("执行一次补扫", variant="primary")
                    batch_button = gr.Button("连续补扫")
                reset_button = gr.Button("重置场景")
                status = gr.HTML(value=initial_values[1])

            with gr.Column(scale=4, min_width=0, elem_classes="visual-workspace"):
                with gr.Tabs():
                    with gr.Tab("缺口模型"):
                        mesh_plot = gr.Plot(value=initial_values[2], label="BIM 信息缺口")
                        comparison_plot = gr.Plot(value=initial_values[4], label="补扫前后")
                    with gr.Tab("补扫视角"):
                        candidate_plot = gr.Plot(value=initial_values[3], label="候选视角")
                        view_table = gr.Dataframe(
                            value=initial_values[8],
                            label="候选视角排名",
                            interactive=False,
                            row_count=5,
                        )
                    with gr.Tab("指标分析"):
                        heatmap_plot = gr.Plot(value=initial_values[5], label="缺口指标")
                        component_plot = gr.Plot(value=initial_values[6], label="构件优先级")
                        component_table = gr.Dataframe(
                            value=initial_values[9],
                            label="构件排名",
                            interactive=False,
                            row_count=5,
                        )
                    with gr.Tab("闭环记录"):
                        recovery_plot = gr.Plot(value=initial_values[7], label="补扫收敛")
                        history_table = gr.Dataframe(
                            value=initial_values[10],
                            label="补扫历史",
                            interactive=False,
                            row_count=5,
                        )

        outputs = [
            app_state,
            status,
            mesh_plot,
            candidate_plot,
            comparison_plot,
            heatmap_plot,
            component_plot,
            recovery_plot,
            view_table,
            component_table,
            history_table,
        ]

        load_button.click(
            load_source,
            inputs=[source, threshold, top_k],
            outputs=outputs,
        )
        import_button.click(
            import_model,
            inputs=[model_file, score_file, threshold, top_k],
            outputs=outputs,
        )
        refresh_button.click(
            refresh_recommendations,
            inputs=[app_state, threshold, top_k],
            outputs=outputs,
        )
        scan_button.click(
            run_one_scan,
            inputs=[app_state, recovery, threshold, top_k],
            outputs=outputs,
        )
        batch_button.click(
            run_scan_steps,
            inputs=[app_state, recovery, threshold, top_k, batch_steps],
            outputs=outputs,
        )
        reset_button.click(
            load_source,
            inputs=[source, threshold, top_k],
            outputs=outputs,
        )
        demo.load(
            load_source,
            inputs=[source, threshold, top_k],
            outputs=outputs,
        )
    return demo


def launch(
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = False,
) -> None:
    WEB_CACHE.mkdir(parents=True, exist_ok=True)
    app = build_app()
    app.queue(default_concurrency_limit=1).launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_error=True,
        strict_cors=False,   # allow browser access from any host (remote SSH / reverse proxy)
    )
