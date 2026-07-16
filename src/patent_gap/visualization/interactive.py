"""
Interactive 3D visualization for collaborator demos.

Generates a self-contained HTML file (no server needed) using Plotly.
Includes:
  1. IFC mesh colored by G_gap score (heatmap)
  2. Point-cloud scatter (sampled) colored by semantic label / D_obs
  3. Candidate viewpoint arrows (top-K views)
  4. Per-component gap summary bar chart
  5. Closed-loop recovery curve
  6. Individual gap-component radar / parallel-coordinates chart

All figures are combined into a single multi-tab HTML page.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ──────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ──────────────────────────────────────────────────────────────────────────────

COLORSCALE_GAP = [
    [0.0, "#2ecc71"],   # green  = no gap
    [0.4, "#f1c40f"],   # yellow = moderate
    [0.7, "#e67e22"],   # orange = high
    [1.0, "#e74c3c"],   # red    = critical
]

IFC_CLASS_COLORS = {
    "IfcWall": "#95a5a6",
    "IfcWallStandardCase": "#95a5a6",
    "IfcSlab": "#bdc3c7",
    "IfcDoor": "#e67e22",
    "IfcWindow": "#3498db",
    "IfcColumn": "#8e44ad",
    "IfcFurnishingElement": "#e74c3c",
    "IfcOpeningElement": "#ecf0f1",
    "IfcCovering": "#d5d8dc",
    "default": "#aab7b8",
}

SEMANTIC_COLORS = {
    "wall": "#95a5a6", "floor": "#d5dbdb", "ceiling": "#d5dbdb",
    "door": "#e67e22", "window": "#3498db", "column": "#8e44ad",
    "furniture": "#e74c3c", "pipe": "#f39c12", "equipment": "#1abc9c",
    "unknown": "#bfc9ca",
}


def _as_vec3(value: Any) -> np.ndarray:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            value = np.fromstring(value.strip("[]()").replace(",", " "), sep=" ")
    vec = np.asarray(value, dtype=float).reshape(-1)
    if vec.size != 3 or not np.isfinite(vec).all():
        raise ValueError(f"expected a finite 3-vector, got {value!r}")
    return vec


# ──────────────────────────────────────────────────────────────────────────────
# Mesh helpers
# ──────────────────────────────────────────────────────────────────────────────

def _stratified_sample_faces(tri_map: np.ndarray, max_faces: int) -> np.ndarray:
    """Sample faces proportionally per element so every element stays visible."""
    unique_elems = np.unique(tri_map)
    n_elems = len(unique_elems)
    if n_elems == 0 or len(tri_map) <= max_faces:
        return np.arange(len(tri_map))
    per_elem = max(1, max_faces // n_elems)
    rng = np.random.default_rng(42)
    parts = []
    for ei in unique_elems:
        idx = np.where(tri_map == ei)[0]
        n = min(len(idx), per_elem)
        parts.append(rng.choice(idx, n, replace=False) if n < len(idx) else idx)
    keep = np.concatenate(parts)
    if len(keep) < max_faces:
        pool = np.setdiff1d(np.arange(len(tri_map)), keep)
        extra = min(len(pool), max_faces - len(keep))
        if extra > 0:
            keep = np.concatenate([keep, rng.choice(pool, extra, replace=False)])
    return keep


def _apply_keep(verts: np.ndarray, faces: np.ndarray, keep: np.ndarray, scores: np.ndarray):
    """Subset faces/verts by keep index array; returns (verts, faces, scores)."""
    sel_faces = faces[keep]
    sel_scores = scores[keep]
    used = np.unique(sel_faces)
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[sel_faces], sel_scores


def _vert_avg(faces: np.ndarray, tri_vals: np.ndarray, n_verts: int) -> np.ndarray:
    """Vectorised average of per-triangle values onto vertices."""
    v = np.zeros(n_verts)
    c = np.zeros(n_verts)
    for col in range(3):
        np.add.at(v, faces[:, col], tri_vals)
        np.add.at(c, faces[:, col], 1.0)
    c = np.where(c > 0, c, 1.0)
    return v / c


def _decimate_mesh(vertices: np.ndarray, faces: np.ndarray, max_faces: int = 80_000):
    """Stratified face-subsampling (kept for backward compat; needs tri_map for full effect)."""
    if len(faces) <= max_faces:
        return vertices, faces
    rng = np.random.default_rng(42)
    keep = rng.choice(len(faces), max_faces, replace=False)
    kept_faces = faces[keep]
    used_verts = np.unique(kept_faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used_verts] = np.arange(len(used_verts))
    new_verts = vertices[used_verts]
    new_faces = remap[kept_faces]
    return new_verts, new_faces


def _gap_per_triangle(
    patches_df: pd.DataFrame,
    tri_map: np.ndarray,
    elements_df: pd.DataFrame,
    score_col: str = "G_gap",
    aggregation: str = "max",
) -> np.ndarray:
    """Map patch G_gap scores onto every triangle.

    aggregation='max'  – conservative worst-case per element (heatmap).
    aggregation='mean' – representative average per element (comparison).
    """
    scores = np.full(len(tri_map), np.nan)
    guid_to_ei: dict[str, int] = {
        str(r.GlobalId): int(r.element_index) for r in elements_df.itertuples()
    }

    elem_vals: dict[int, list[float]] = {}
    for patch in patches_df.itertuples():
        raw = getattr(patch, score_col, np.nan)
        val = 0.5 if pd.isna(raw) else float(raw)
        if hasattr(patch, "element_index") and patch.element_index is not None:
            try:
                ei = int(patch.element_index)
            except (TypeError, ValueError):
                ei = guid_to_ei.get(str(getattr(patch, "element_guid", "")), -1)
        else:
            ei = guid_to_ei.get(str(getattr(patch, "element_guid", "")), -1)
        if ei < 0:
            continue
        elem_vals.setdefault(ei, []).append(val)

    for ei, vals in elem_vals.items():
        agg = float(np.max(vals)) if aggregation == "max" else float(np.mean(vals))
        scores[tri_map == ei] = agg

    nan_mask = np.isnan(scores)
    if nan_mask.any():
        scores[nan_mask] = float(np.nanmedian(scores)) if not np.all(nan_mask) else 0.5
    return np.clip(scores, 0, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Individual figure builders
# ──────────────────────────────────────────────────────────────────────────────

def fig_ifc_gap_mesh(
    mesh_npz_path: str | Path,
    tri_map: np.ndarray,
    patches_df: pd.DataFrame,
    elements_df: pd.DataFrame,
    score_col: str = "G_gap",
    max_faces: int = 80_000,
) -> "go.Figure":
    mesh = np.load(str(mesh_npz_path))
    verts, faces = mesh["vertices"].astype(np.float64), mesh["faces"].astype(np.int64)
    gap_scores = _gap_per_triangle(patches_df, tri_map, elements_df, score_col)

    # Stratified sampling: every IFC element stays visible
    keep = _stratified_sample_faces(tri_map, max_faces)
    verts, faces, gap_scores = _apply_keep(verts, faces, keep, gap_scores)

    vert_scores = _vert_avg(faces, gap_scores, len(verts))

    fig = go.Figure(go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=vert_scores,
        colorscale=COLORSCALE_GAP,
        cmin=0, cmax=1,
        colorbar=dict(title="G_gap", thickness=15, len=0.6),
        opacity=0.9,
        name="IFC BIM",
        hovertemplate="G_gap: %{intensity:.3f}<extra></extra>",
        lighting=dict(ambient=0.55, diffuse=0.85, roughness=0.4, specular=0.1),
        lightposition=dict(x=100, y=200, z=150),
    ))
    fig.update_layout(
        title="BIM 信息缺口热力图（G_gap 每面）",
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
            bgcolor="white",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=650,
    )
    return fig


def _categorical_traces(
    verts: np.ndarray,
    faces: np.ndarray,
    tri_mask: dict[str, np.ndarray],
    colors: dict[str, str],
    legendgroup: str,
) -> list["go.Mesh3d"]:
    """Build one fixed-color Mesh3d trace per category."""
    traces = []
    for label, mask in tri_mask.items():
        if not mask.any():
            continue
        sub = faces[mask]
        used = np.unique(sub)
        if len(used) == 0:
            continue
        rmp = np.full(len(verts), -1, dtype=np.int64)
        rmp[used] = np.arange(len(used))
        traces.append(go.Mesh3d(
            x=verts[used, 0], y=verts[used, 1], z=verts[used, 2],
            i=rmp[sub[:, 0]], j=rmp[sub[:, 1]], k=rmp[sub[:, 2]],
            color=colors[label],
            opacity=0.9,
            lighting=dict(ambient=0.55, diffuse=0.85, roughness=0.4),
            name=label,
            showlegend=True,
            legendgroup=legendgroup,
            legendgrouptitle_text=legendgroup,
            hovertemplate=f"{label}<extra></extra>",
        ))
    return traces


def fig_before_after_mesh(
    mesh_npz_path: str | Path,
    tri_map: np.ndarray,
    initial_patches_df: pd.DataFrame,
    current_patches_df: pd.DataFrame,
    elements_df: pd.DataFrame,
    threshold: float = 0.55,
    max_faces: int = 25_000,
) -> "go.Figure":
    """
    Side-by-side comparison with matching categorical color schemes:

    LEFT (补扫前)                   RIGHT (补扫后)
    ■ gray  = 无缺口 (G_gap<thr)    ■ gray  = 仍无缺口
    ■ red   = 有缺口 (G_gap≥thr)    ■ green = 已恢复 ✓  (was bad → now OK)
                                    ■ red   = 仍缺失 ✗  (still bad)

    Uses per-element MEAN G_gap so one bad patch can't paint the whole element red.
    """
    mesh = np.load(str(mesh_npz_path))
    verts0 = mesh["vertices"].astype(np.float64)
    faces0 = mesh["faces"].astype(np.int64)

    # Use MEAN per element: prevents one bad patch from flagging an entire wall
    init_tri = _gap_per_triangle(initial_patches_df, tri_map, elements_df, aggregation="mean")
    curr_tri = _gap_per_triangle(current_patches_df, tri_map, elements_df, aggregation="mean")

    keep = _stratified_sample_faces(tri_map, max_faces)
    verts, faces, init_s = _apply_keep(verts0, faces0, keep, init_tri)
    _, _, curr_s = _apply_keep(verts0, faces0, keep, curr_tri)

    was_bad = init_s >= threshold
    is_bad  = curr_s >= threshold

    # ── Left: 2-state categorical (initial) ────────────────────────────────
    left_masks = {
        "无缺口": ~was_bad,
        "有缺口": was_bad,
    }
    left_colors = {"无缺口": "#95a5a6", "有缺口": "#e74c3c"}

    # ── Right: 3-state categorical (current) ───────────────────────────────
    right_masks = {
        "无缺口 (从未缺失)": ~was_bad,
        "已恢复 ✓":         was_bad & ~is_bad,
        "仍缺失 ✗":         was_bad &  is_bad,
    }
    right_colors = {
        "无缺口 (从未缺失)": "#95a5a6",
        "已恢复 ✓":         "#2ecc71",
        "仍缺失 ✗":         "#e74c3c",
    }

    # ── Subtitle stats ──────────────────────────────────────────────────────
    n   = len(faces)
    n_bad      = int(was_bad.sum())
    n_recov    = int((was_bad & ~is_bad).sum())
    n_still    = int((was_bad & is_bad).sum())
    n_fine     = n - n_bad
    pct = lambda k: f"{k/n:.0%}" if n > 0 else "0%"

    subtitle_before = f"■灰={pct(n_fine)}无缺口  ■红={pct(n_bad)}有缺口  (阈值={threshold:.2f})"
    subtitle_after  = (
        f"■灰={pct(n_fine)}无缺口  "
        f"■绿={pct(n_recov)}已恢复  "
        f"■红={pct(n_still)}仍缺失"
    )

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(f"补扫前 — {subtitle_before}", f"补扫后 — {subtitle_after}"),
        horizontal_spacing=0.02,
    )
    for t in _categorical_traces(verts, faces, left_masks, left_colors, "补扫前"):
        fig.add_trace(t, row=1, col=1)
    for t in _categorical_traces(verts, faces, right_masks, right_colors, "补扫后"):
        fig.add_trace(t, row=1, col=2)

    camera = dict(eye=dict(x=1.5, y=-1.5, z=1.2))
    scene_cfg = dict(
        aspectmode="data", camera=camera, bgcolor="white",
        xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
    )
    fig.update_layout(
        scene=scene_cfg, scene2=scene_cfg,
        height=600,
        margin=dict(l=0, r=0, t=65, b=30),
        showlegend=True,
        legend=dict(x=0.5, y=-0.04, orientation="h", xanchor="center", groupclick="toggleitem"),
    )
    return fig


def fig_point_cloud(
    assoc_df: pd.DataFrame,
    color_by: str = "ifc_class",
    max_pts: int = 50_000,
) -> "go.Figure":
    """Scatter3d of the sample point cloud, coloured by IFC class or obs status."""
    df = assoc_df.copy()
    if "registered_x" in df.columns:
        df["px"] = df["registered_x"]
        df["py"] = df["registered_y"]
        df["pz"] = df["registered_z"]
    else:
        df["px"] = df["x"]
        df["py"] = df["y"]
        df["pz"] = df["z"]

    if len(df) > max_pts:
        df = df.sample(max_pts, random_state=42)

    traces = []
    if color_by == "ifc_class" and "ifc_class" in df.columns:
        for cls, grp in df.groupby("ifc_class", dropna=False):
            cls_str = str(cls) if pd.notna(cls) else "unmatched"
            color = IFC_CLASS_COLORS.get(cls_str, IFC_CLASS_COLORS["default"])
            traces.append(go.Scatter3d(
                x=grp["px"], y=grp["py"], z=grp["pz"],
                mode="markers",
                marker=dict(size=1.5, color=color, opacity=0.7),
                name=cls_str,
                hovertemplate=f"Class: {cls_str}<br>x=%{{x:.2f}} y=%{{y:.2f}} z=%{{z:.2f}}<extra></extra>",
            ))
    else:
        matched = df.get("matched", pd.Series(True, index=df.index))
        for matched_val, grp in df.groupby(matched):
            color = "#2ecc71" if matched_val else "#95a5a6"
            label = "matched" if matched_val else "unmatched"
            traces.append(go.Scatter3d(
                x=grp["px"], y=grp["py"], z=grp["pz"],
                mode="markers",
                marker=dict(size=1.5, color=color, opacity=0.6),
                name=label,
            ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"CRAS Point Cloud (coloured by {color_by}, {len(df):,} pts)",
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
    )
    return fig


def fig_candidate_views_3d(
    ranked_views: pd.DataFrame,
    patches_df: pd.DataFrame,
    top_k: int = 10,
) -> "go.Figure":
    """3D plot: patch centroids coloured by G_gap + top-K candidate view arrows."""
    fig = go.Figure()

    # Patch centroids
    if "centroid_x" in patches_df.columns:
        cx = patches_df["centroid_x"].values
        cy = patches_df["centroid_y"].values
        cz = patches_df["centroid_z"].values
    else:
        cx = np.array([c[0] for c in patches_df["centroid"]])
        cy = np.array([c[1] for c in patches_df["centroid"]])
        cz = np.array([c[2] for c in patches_df["centroid"]])

    gap = patches_df["G_gap"].values if "G_gap" in patches_df.columns else np.zeros(len(patches_df))

    fig.add_trace(go.Scatter3d(
        x=cx, y=cy, z=cz,
        mode="markers",
        marker=dict(
            size=5,
            color=gap,
            colorscale=COLORSCALE_GAP,
            cmin=0, cmax=1,
            colorbar=dict(title="G_gap", thickness=12, len=0.5, x=1.02),
            opacity=0.9,
        ),
        name="Patch centroids",
        hovertemplate="G_gap=%{marker.color:.3f}<br>x=%{x:.2f} y=%{y:.2f} z=%{z:.2f}<extra></extra>",
    ))

    # Candidate view arrows (cones)
    if not ranked_views.empty:
        top_views = ranked_views.head(top_k)
        pos_arr = np.vstack([_as_vec3(p) for p in top_views["position"]])
        ori_arr = np.vstack([_as_vec3(o) for o in top_views["orientation"]])
        values = top_views["value"].values if "value" in top_views.columns else np.ones(len(top_views))

        fig.add_trace(go.Cone(
            x=pos_arr[:, 0], y=pos_arr[:, 1], z=pos_arr[:, 2],
            u=ori_arr[:, 0], v=ori_arr[:, 1], w=ori_arr[:, 2],
            colorscale="Blues",
            sizemode="absolute", sizeref=0.4,
            anchor="tail",
            showscale=False,
            name=f"Top-{top_k} candidate views",
            hovertemplate="View value: %{customdata:.3f}<extra></extra>",
            customdata=values[:, None],
        ))

        # Rank labels
        for i, row in enumerate(top_views.itertuples()):
            p = _as_vec3(row.position)
            fig.add_trace(go.Scatter3d(
                x=[p[0]], y=[p[1]], z=[p[2]],
                mode="text",
                text=[f"#{i+1}"],
                textfont=dict(size=11, color="navy"),
                showlegend=False,
                hoverinfo="skip",
            ))

    fig.update_layout(
        title=f"Top-{top_k} Recommended Supplemental Scan Viewpoints",
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.8, y=-1.8, z=1.5)),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=650,
    )
    return fig


def fig_component_gap_bar(component_ranking: pd.DataFrame) -> "go.Figure":
    """Horizontal bar chart of G_component scores, colour-coded by IFC class."""
    if component_ranking.empty:
        fig = go.Figure()
        fig.update_layout(title="Component Verification Priority", height=400)
        return fig
    df = (
        component_ranking.nlargest(30, "G_component")
        .sort_values("G_component", ascending=True)
    )
    colors = [IFC_CLASS_COLORS.get(str(c), IFC_CLASS_COLORS["default"])
              for c in df.get("ifc_class", ["default"] * len(df))]
    # Overlay gap type breakdown
    gap_cols = [c for c in ["mean_gap", "max_gap", "p90_gap"] if c in df.columns]
    fig = go.Figure()
    if gap_cols:
        palette = ["#3498db", "#e74c3c", "#f39c12"]
        for i, col in enumerate(gap_cols):
            fig.add_trace(go.Bar(
                y=df["element_guid"].str[:20],
                x=df[col],
                name=col,
                orientation="h",
                marker_color=palette[i],
                opacity=0.75,
            ))
        fig.update_layout(barmode="overlay")
    else:
        fig.add_trace(go.Bar(
            y=df["element_guid"].str[:20],
            x=df["G_component"],
            orientation="h",
            marker_color=colors,
            name="G_component",
        ))

    fig.update_layout(
        title="Component Verification Priority (G_component)",
        xaxis_title="Gap Score",
        yaxis_title="Element GUID",
        height=max(400, 20 * len(df)),
        margin=dict(l=160, r=20, t=50, b=40),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def fig_gap_components_heatmap(patch_scores: pd.DataFrame, max_patches: int = 60) -> "go.Figure":
    """Heatmap of all gap components per patch — shows which component drives the gap."""
    if patch_scores.empty:
        fig = go.Figure()
        fig.update_layout(title="Gap Component Heatmap", height=350)
        return fig
    cols = ["D_sem", "D_mat_missing", "D_mat_conflict", "D_obs", "D_ang", "D_geo", "G_gap"]
    cols = [c for c in cols if c in patch_scores.columns]
    df = patch_scores.sort_values("G_gap", ascending=False).head(max_patches)[
        ["patch_id"] + cols
    ].set_index("patch_id")

    labels = [f"patch_{i}" for i in df.index]

    # Fill NaN with -0.05 so they appear as a distinct "no data" shade
    z = df[cols].fillna(-0.05).values.T

    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels,
        y=cols,
        colorscale=[[0, "#bdc3c7"], [0.05, "#2ecc71"], [0.4, "#f1c40f"], [0.7, "#e67e22"], [1.0, "#e74c3c"]],
        zmin=-0.05, zmax=1.0,
        colorbar=dict(title="Score", thickness=12),
        hovertemplate="Patch: %{x}<br>Component: %{y}<br>Value: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Gap Component Heatmap (top-{max_patches} patches by G_gap)",
        xaxis=dict(tickangle=45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=11)),
        height=350,
        margin=dict(l=120, r=20, t=50, b=100),
    )
    return fig


def fig_closed_loop_curve(history_df: pd.DataFrame) -> "go.Figure":
    """Closed-loop recovery curve: remaining gap area and recovery rate vs. iteration."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if history_df.empty:
        fig.update_layout(title="Closed-Loop Supplemental Scan Convergence", height=420)
        return fig

    fig.add_trace(go.Scatter(
        x=history_df["step"],
        y=history_df["remaining_gap_area"],
        mode="lines+markers",
        name="Remaining gap area (weighted)",
        line=dict(color="#e74c3c", width=2.5),
        marker=dict(size=7),
    ), secondary_y=False)

    if "recovered_missing_area" in history_df.columns:
        fig.add_trace(go.Scatter(
            x=history_df["step"],
            y=history_df["recovered_missing_area"],
            mode="lines+markers",
            name="Recovered missing area (m²)",
            line=dict(color="#2ecc71", width=2.5, dash="dash"),
            marker=dict(size=7, symbol="square"),
        ), secondary_y=False)

    if "recovery_rate" in history_df.columns:
        fig.add_trace(go.Scatter(
            x=history_df["step"],
            y=history_df["recovery_rate"] * 100,
            mode="lines+markers",
            name="Recovery rate (%)",
            line=dict(color="#3498db", width=2),
            marker=dict(size=6, symbol="diamond"),
        ), secondary_y=True)

    fig.update_xaxes(title_text="Supplemental scan iteration")
    fig.update_yaxes(title_text="Area-weighted value", secondary_y=False)
    fig.update_yaxes(title_text="Recovery rate (%)", secondary_y=True, range=[0, 105])
    fig.update_layout(
        title="Closed-Loop Supplemental Scan Convergence",
        height=420,
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=60, r=60, t=50, b=80),
    )
    return fig


def fig_baseline_comparison(baseline_df: pd.DataFrame) -> "go.Figure":
    """Grouped bar chart of AUROC / AUPRC / F1 across baseline methods."""
    if baseline_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Baseline Method Comparison", height=420)
        return fig
    metrics = [c for c in ["AUROC", "AUPRC", "F1"] if c in baseline_df.columns]
    methods = baseline_df["method"].tolist() if "method" in baseline_df.columns else [
        f"method_{i}" for i in range(len(baseline_df))
    ]

    fig = go.Figure()
    palette = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]
    for mi, metric in enumerate(metrics):
        # Average across seeds
        if "seed" in baseline_df.columns:
            vals = baseline_df.groupby("method")[metric].mean().reindex(
                baseline_df["method"].unique()
            ).values
            meth_labels = baseline_df["method"].unique().tolist()
        else:
            vals = baseline_df[metric].values
            meth_labels = methods
        fig.add_trace(go.Bar(
            name=metric,
            x=meth_labels,
            y=vals,
            marker_color=palette[mi % len(palette)],
        ))

    fig.update_layout(
        title="Baseline Method Comparison",
        barmode="group",
        yaxis=dict(range=[0, 1.05], title="Score"),
        xaxis=dict(tickangle=20),
        legend=dict(orientation="h", y=-0.25),
        height=420,
        margin=dict(l=60, r=20, t=50, b=100),
    )
    return fig


def fig_ablation_table_chart(ablation_df: pd.DataFrame) -> "go.Figure":
    """Parallel-coordinates chart of ablation results."""
    if ablation_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Ablation Study", height=400)
        return fig
    metrics = [c for c in ["AUROC", "AUPRC", "F1", "Brier"] if c in ablation_df.columns]
    abl_col = "ablation" if "ablation" in ablation_df.columns else ablation_df.columns[0]

    # Average across seeds if present
    if "seed" in ablation_df.columns:
        df = ablation_df.groupby(abl_col)[metrics].mean().reset_index()
    else:
        df = ablation_df.copy()

    dims = [dict(label=m, values=df[m]) for m in metrics]
    dims.insert(0, dict(
        label=abl_col,
        values=list(range(len(df))),
        tickvals=list(range(len(df))),
        ticktext=df[abl_col].tolist(),
    ))

    fig = go.Figure(go.Parcoords(
        line=dict(
            color=df[metrics[0]].values if metrics else list(range(len(df))),
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=metrics[0] if metrics else ""),
        ),
        dimensions=dims,
    ))
    fig.update_layout(
        title="Ablation Study — Parallel Coordinates",
        height=400,
        margin=dict(l=120, r=40, t=60, b=20),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Master HTML builder
# ──────────────────────────────────────────────────────────────────────────────

def build_interactive_report(
    patches_df: pd.DataFrame,
    component_ranking: pd.DataFrame,
    ranked_views: pd.DataFrame,
    history_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    mesh_npz_path: str | Path,
    tri_map: np.ndarray,
    elements_df: pd.DataFrame,
    assoc_df: pd.DataFrame,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
) -> str:
    """
    Build a single-file interactive HTML report for collaborator presentations.

    Returns the path to the HTML file.
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for interactive visualization. Install with: pip install plotly")

    cfg = config or {}
    top_k = int(cfg.get("top_k_views", 10))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge G_gap into patches_df for centroid colouring
    score_df = patches_df.copy()

    # ── Build all figures ────────────────────────────────────────────────────
    print("  [viz] Building IFC gap mesh …")
    fig_mesh = fig_ifc_gap_mesh(mesh_npz_path, tri_map, score_df, elements_df)

    print("  [viz] Building point cloud scatter …")
    fig_pc = fig_point_cloud(assoc_df, color_by="ifc_class")

    print("  [viz] Building candidate views …")
    fig_views = fig_candidate_views_3d(ranked_views, score_df, top_k=top_k)

    print("  [viz] Building component gap bar …")
    fig_comp = fig_component_gap_bar(component_ranking)

    print("  [viz] Building gap component heatmap …")
    fig_hm = fig_gap_components_heatmap(score_df)

    print("  [viz] Building closed-loop curve …")
    fig_cl = fig_closed_loop_curve(history_df)

    print("  [viz] Building baseline comparison …")
    fig_bl = fig_baseline_comparison(baseline_df)

    print("  [viz] Building ablation chart …")
    fig_abl = fig_ablation_table_chart(ablation_df)

    # ── Assemble HTML with tabs ──────────────────────────────────────────────
    figures = [
        ("BIM Gap Heatmap", fig_mesh),
        ("Point Cloud", fig_pc),
        ("Candidate Views", fig_views),
        ("Component Ranking", fig_comp),
        ("Gap Components", fig_hm),
        ("Closed-Loop", fig_cl),
        ("Baselines", fig_bl),
        ("Ablation", fig_abl),
    ]

    # Convert each figure to an HTML div (no full-page wrapper)
    divs = []
    for title, fig in figures:
        div_html = pio.to_html(
            fig,
            full_html=False,
            include_plotlyjs=True if not divs else False,
        )
        divs.append((title, div_html))

    # Build tabbed page
    tab_buttons = "\n".join(
        f'<button class="tablink" onclick="openTab(event,\'{i}\')">{title}</button>'
        for i, (title, _) in enumerate(divs)
    )
    tab_contents = "\n".join(
        f'<div id="{i}" class="tabcontent" style="display:{"block" if i==0 else "none"}">'
        f'<h2>{title}</h2>{div}</div>'
        for i, (title, div) in enumerate(divs)
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>BIM Gap Analysis — Interactive Report</title>
<style>
  body {{ font-family: "Segoe UI", Arial, sans-serif; margin: 0; background: #f4f6f8; color: #2c3e50; }}
  h1 {{ background: #2c3e50; color: white; padding: 18px 28px; margin: 0;
       font-size: 1.4em; letter-spacing: 0.05em; }}
  .subtitle {{ background: #34495e; color: #ecf0f1; padding: 6px 28px;
               font-size: 0.85em; margin: 0; }}
  .tabbar {{ background: #2980b9; padding: 0 16px; display: flex; flex-wrap: wrap; }}
  .tablink {{ background: transparent; border: none; color: #ecf0f1; padding: 12px 18px;
              cursor: pointer; font-size: 0.92em; border-bottom: 3px solid transparent; }}
  .tablink:hover {{ background: rgba(255,255,255,0.12); }}
  .tablink.active {{ border-bottom: 3px solid #f1c40f; color: #f1c40f; font-weight: bold; }}
  .tabcontent {{ padding: 18px 24px; }}
  h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 6px;
        margin-top: 0; }}
  .legend-box {{ background: white; border-radius: 6px; padding: 10px 16px;
                 margin-bottom: 12px; font-size: 0.82em; box-shadow: 0 1px 4px #0002; }}
  .legend-dot {{ display: inline-block; width: 12px; height: 12px; border-radius: 50%;
                 margin-right: 6px; vertical-align: middle; }}
</style>
</head>
<body>
<h1>BIM-3D Vision Joint Modeling — Spatial Information Gap Analysis</h1>
<div class="subtitle">
  Patent: 《基于BIM-三维视觉联合建模的空间信息缺口分析与补扫视角评估方法》 &nbsp;|&nbsp;
  Dataset: CRAS Lab BIM Dataset (Zenodo 7948116) &nbsp;|&nbsp;
  <span style="color:#f1c40f">&#9650; High gap</span>
  &nbsp;<span style="color:#e67e22">&#9679; Medium gap</span>
  &nbsp;<span style="color:#2ecc71">&#9660; Low gap</span>
</div>
<div class="tabbar" id="tabbar">
{tab_buttons}
</div>
<div class="legend-box">
  <b>Colour scale:</b>
  <span class="legend-dot" style="background:#2ecc71"></span>G_gap ≈ 0 (well-observed) &nbsp;
  <span class="legend-dot" style="background:#f1c40f"></span>G_gap ≈ 0.4 (moderate) &nbsp;
  <span class="legend-dot" style="background:#e67e22"></span>G_gap ≈ 0.7 (high) &nbsp;
  <span class="legend-dot" style="background:#e74c3c"></span>G_gap ≈ 1 (critical) &nbsp;|&nbsp;
  <b>Blue cones</b> = recommended supplemental scan viewpoints
</div>
{tab_contents}
<script>
function openTab(evt, tabId) {{
  document.querySelectorAll('.tabcontent').forEach(t => t.style.display = 'none');
  document.querySelectorAll('.tablink').forEach(b => b.classList.remove('active'));
  document.getElementById(tabId).style.display = 'block';
  evt.currentTarget.classList.add('active');
}}
document.querySelector('.tablink').classList.add('active');
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"  [viz] Interactive report saved → {output_path}")
    return str(output_path)
