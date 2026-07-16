#!/usr/bin/env python3
"""
Synthetic scan generator for CRAS IFC model.

Shoots rays from a grid of virtual scanner positions inside the building using
Open3D RaycastingScene.  Produces a point cloud with per-point element/semantic
information that the full pipeline can consume without needing real scan data.

Intentionally leaves the far end of the building unscanned (Y > 14 m) so that
those patches receive gt_missing=True, giving meaningful binary labels for
evaluation.

Outputs (in data/processed/synthetic_scan/):
  points.parquet         – per-point data (x,y,z, classification, elem_index, …)
  element_stats.csv      – per-element point count + semantic distribution
  patch_stats.csv        – per-patch point count + view/frontality statistics
  scanner_positions.csv  – scanner positions used
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd

from patent_gap.patches import build_patches

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT_DIR = PROCESSED / "synthetic_scan"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── scan parameters ────────────────────────────────────────────────────────────
AZ_STEP_DEG   = 1.0    # azimuth step (°)
EL_MIN_DEG    = -30.0  # minimum elevation (°)  – picks up floor
EL_MAX_DEG    =  80.0  # maximum elevation (°)  – nearly vertical for ceiling
EL_STEP_DEG   =  1.0   # elevation step (°)
MAX_RANGE_M   = 20.0   # rays beyond this distance are discarded
MIN_RANGE_M   =  0.05  # self-intersection guard
NOISE_FRAC    =  0.12  # fraction of labels randomly mis-classified

# ── scanner positions ──────────────────────────────────────────────────────────
# Building interior spans roughly X∈[-9,8] Y∈[-4,20] Z∈[-1,3].
# Scanners placed at chest height (Z=1.5 m) in a 3-column × 3-row grid
# covering Y=0..14.  The far end (Y≈15-20) is intentionally unscanned
# → those patches will receive gt_missing=True.
SCANNER_XY = [
    (-6.0,  0.5), (-1.0,  0.5), ( 4.0,  0.5),   # row 0, Y≈0
    (-6.0,  6.0), (-1.0,  6.0), ( 4.0,  6.0),   # row 1, Y≈6
    (-6.0, 12.0), (-1.0, 12.0), ( 4.0, 12.0),   # row 2, Y≈12
]
SCANNER_Z = 1.5  # scanner height inside room

# ── IFC class → classification code ───────────────────────────────────────────
# Mirrors the CRAS point-cloud label scheme used by the semantics module.
IFC_TO_CODE: dict[str, int] = {
    "IfcWallStandardCase":     1,   # wall
    "IfcSlab":                 2,   # floor (ceiling handled separately by normal)
    "IfcColumn":              14,   # column
    "IfcDoor":                 4,   # door
    "IfcWindow":               5,   # window
    "IfcFurnishingElement":    6,   # furniture (chair/table/other)
    "IfcOpeningElement":       0,   # hole/void → unclassified
    "IfcCovering":             2,   # floor/ceiling covering
    "IfcStairFlight":          2,   # stair treated as floor
    "IfcBuildingElementProxy": 0,   # generic → unclassified
}
ALL_CODES = [0, 1, 2, 3, 4, 5, 6, 14, 30, 31]

rng = np.random.default_rng(42)


def _base_label(elem_index: int, ifc_class: str, face_normal: np.ndarray) -> int:
    """Return the ground-truth classification code for a hit face."""
    code = IFC_TO_CODE.get(ifc_class, 0)
    # IfcSlab or IfcCovering: distinguish floor vs ceiling by face normal
    if ifc_class in ("IfcSlab", "IfcCovering") and face_normal is not None:
        code = 3 if float(face_normal[2]) < -0.5 else 2
    return code


def _inject_noise(codes: np.ndarray, fraction: float) -> np.ndarray:
    """Randomly perturb a fraction of semantic labels."""
    noisy = codes.copy()
    n_noise = int(len(codes) * fraction)
    if n_noise == 0:
        return noisy
    idx = rng.choice(len(codes), size=n_noise, replace=False)
    for i in idx:
        candidates = [c for c in ALL_CODES if c != noisy[i]]
        noisy[i] = int(rng.choice(candidates))
    return noisy


def build_raycasting_scene(verts: np.ndarray, faces: np.ndarray) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene()
    tmesh = o3d.t.geometry.TriangleMesh(
        vertex_positions=o3d.core.Tensor(verts.astype(np.float32)),
        triangle_indices=o3d.core.Tensor(faces.astype(np.uint32)),
    )
    scene.add_triangles(tmesh)
    return scene


def generate_rays(scanner_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (origins, directions) arrays, shape (N_rays, 3)."""
    az = np.deg2rad(np.arange(0, 360, AZ_STEP_DEG))
    el = np.deg2rad(np.arange(EL_MIN_DEG, EL_MAX_DEG + 1e-9, EL_STEP_DEG))
    AZ, EL = np.meshgrid(az, el)
    dx = np.cos(EL) * np.cos(AZ)
    dy = np.cos(EL) * np.sin(AZ)
    dz = np.sin(EL)
    dirs = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1).astype(np.float32)
    origins = np.tile(scanner_pos.astype(np.float32), (len(dirs), 1))
    return origins, dirs


def compute_face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-face unit normals, shape (N_faces, 3)."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return n / norms


def main() -> None:
    build_patches(
        mesh_npz_path=PROCESSED / "ifc_mesh.npz",
        triangle_element_map_path=PROCESSED / "triangle_element_map.npy",
        elements_parquet_path=PROCESSED / "ifc_elements.parquet",
        materials_parquet_path=PROCESSED / "ifc_materials.parquet",
        output_path=PROCESSED / "patches_real.parquet",
    )
    print("Loading IFC mesh …")
    mesh_data = np.load(PROCESSED / "ifc_mesh.npz")
    verts = mesh_data["vertices"]    # (V, 3)
    faces = mesh_data["faces"]       # (F, 3)
    tri_elem = np.load(PROCESSED / "triangle_element_map.npy")   # (F,)
    tri_patch_path = PROCESSED / "patches_real_triangle_map.npy"
    tri_patch = (
        np.load(tri_patch_path)
        if tri_patch_path.exists()
        else np.full(len(faces), -1, dtype=np.int64)
    )
    if len(tri_patch) != len(faces):
        raise ValueError("patch triangle map length does not match IFC face count")
    face_normals = compute_face_normals(verts, faces)            # (F, 3)

    elems_df = pd.read_parquet(PROCESSED / "ifc_elements.parquet")
    idx_to_class = dict(zip(elems_df["element_index"], elems_df["ifc_class"]))
    idx_to_guid  = dict(zip(elems_df["element_index"], elems_df["GlobalId"]))

    print(f"Mesh: {len(verts):,} verts, {len(faces):,} faces")
    print("Building RaycastingScene …")
    scene = build_raycasting_scene(verts, faces)

    scanner_positions = [
        np.array([x, y, SCANNER_Z], dtype=np.float64)
        for x, y in SCANNER_XY
    ]

    all_rows: list[dict] = []

    for scan_id, pos in enumerate(scanner_positions):
        print(f"  Scanner {scan_id+1}/{len(scanner_positions)} at ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) …")
        origins, dirs = generate_rays(pos)

        rays_t = o3d.core.Tensor(np.hstack([origins, dirs]))
        result = scene.cast_rays(rays_t)
        t_hit   = result["t_hit"].numpy()
        prim_id = result["primitive_ids"].numpy()

        valid = (
            np.isfinite(t_hit)
            & (t_hit > MIN_RANGE_M)
            & (t_hit < MAX_RANGE_M)
        )

        if not valid.any():
            print("    → no hits")
            continue

        hit_t    = t_hit[valid]
        hit_tri  = prim_id[valid]
        hit_orig = origins[valid]
        hit_dir  = dirs[valid]

        hit_pts  = hit_orig + hit_dir * hit_t[:, None]
        hit_norm = face_normals[hit_tri]         # (N, 3) per-hit face normal
        hit_elem = tri_elem[hit_tri]             # (N,)  element_index

        # frontality: cos(angle between -ray_dir and face_normal)
        neg_dir = -hit_dir
        frontality = np.clip(
            np.sum(neg_dir * hit_norm, axis=1), 0.0, 1.0
        )

        # base semantic codes from IFC class
        codes = np.array(
            [_base_label(int(ei), idx_to_class.get(int(ei), ""), face_normals[ht])
             for ei, ht in zip(hit_elem, hit_tri)],
            dtype=np.int32,
        )
        codes_noisy = _inject_noise(codes, NOISE_FRAC)

        for i in range(len(hit_pts)):
            ei = int(hit_elem[i])
            all_rows.append({
                "scanner_id":     scan_id,
                "x":              float(hit_pts[i, 0]),
                "y":              float(hit_pts[i, 1]),
                "z":              float(hit_pts[i, 2]),
                "classification": int(codes_noisy[i]),
                "true_label":     int(codes[i]),
                "distance_m":     float(hit_t[i]),
                "frontality":     float(frontality[i]),
                "face_normal_x":  float(hit_norm[i, 0]),
                "face_normal_y":  float(hit_norm[i, 1]),
                "face_normal_z":  float(hit_norm[i, 2]),
                "element_index":  ei,
                "triangle_index": int(hit_tri[i]),
                "patch_id":       int(tri_patch[hit_tri[i]]),
                "element_guid":   idx_to_guid.get(ei, ""),
                "ifc_class":      idx_to_class.get(ei, ""),
            })

        print(f"    → {valid.sum():,} hits across "
              f"{len(np.unique(hit_elem))} elements")

    print(f"\nTotal hit points: {len(all_rows):,}")

    pts_df = pd.DataFrame(all_rows)
    out_pts = OUT_DIR / "points.parquet"
    pts_df.to_parquet(out_pts, index=False)
    print(f"Saved: {out_pts}")

    # ── per-element statistics ─────────────────────────────────────────────────
    print("Computing per-element statistics …")
    elem_stats: list[dict] = []
    for ei, grp in pts_df.groupby("element_index"):
        ei = int(ei)
        sem_counts: dict[int, int] = grp["classification"].value_counts().to_dict()
        elem_stats.append({
            "element_index":   ei,
            "element_guid":    idx_to_guid.get(ei, ""),
            "ifc_class":       idx_to_class.get(ei, ""),
            "point_count":     len(grp),
            "mean_frontality": float(grp["frontality"].mean()),
            "best_frontality": float(grp["frontality"].max()),
            "valid_views":     int((grp["frontality"] > 0.34).sum()),
            "scanner_count": int(grp["scanner_id"].nunique()),
            "valid_scanner_count": int(
                grp.loc[grp["frontality"] > 0.34, "scanner_id"].nunique()
            ),
            "mean_distance_m": float(grp["distance_m"].mean()),
            "semantic_counts": json.dumps(sem_counts),
        })
    stats_df = pd.DataFrame(elem_stats)
    out_stats = OUT_DIR / "element_stats.csv"
    stats_df.to_csv(out_stats, index=False)
    print(f"Saved: {out_stats}")
    print(f"Elements with hits: {len(stats_df)}/{len(elems_df)}")
    print(f"Elements gt_missing (no hits): {len(elems_df) - len(stats_df)}")

    valid_patch_points = pts_df[pts_df["patch_id"] >= 0]
    patch_stats: list[dict] = []
    for patch_id, grp in valid_patch_points.groupby("patch_id"):
        sem_counts = grp["classification"].value_counts().to_dict()
        patch_stats.append({
            "patch_id": int(patch_id),
            "element_index": int(grp["element_index"].iloc[0]),
            "element_guid": str(grp["element_guid"].iloc[0]),
            "ifc_class": str(grp["ifc_class"].iloc[0]),
            "point_count": int(len(grp)),
            "mean_frontality": float(grp["frontality"].mean()),
            "best_frontality": float(grp["frontality"].max()),
            "scanner_count": int(grp["scanner_id"].nunique()),
            "valid_scanner_count": int(
                grp.loc[grp["frontality"] > 0.34, "scanner_id"].nunique()
            ),
            "mean_distance_m": float(grp["distance_m"].mean()),
            "semantic_counts": json.dumps(sem_counts),
        })
    patch_stats_df = pd.DataFrame(patch_stats)
    out_patch_stats = OUT_DIR / "patch_stats.csv"
    patch_stats_df.to_csv(out_patch_stats, index=False)
    print(f"Saved: {out_patch_stats}")
    print(f"Patches with hits: {len(patch_stats_df)}")

    # ── scanner positions log ─────────────────────────────────────────────────
    scan_pos_df = pd.DataFrame(
        [{"scanner_id": i, "x": p[0], "y": p[1], "z": p[2]}
         for i, p in enumerate(scanner_positions)]
    )
    scan_pos_df.to_csv(OUT_DIR / "scanner_positions.csv", index=False)

    print("\nDone. Synthetic scan ready at", OUT_DIR)


if __name__ == "__main__":
    main()
