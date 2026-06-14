from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from patent_gap.data.audit import audit_ifc_text


def inspect_ifc(path: str | Path) -> dict[str, Any]:
    return audit_ifc_text(path)


def triangulate_ifc(path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"path": str(path), "output_dir": str(output_dir), "triangulated": False}
    cache_paths = {
        "mesh": output_dir / "ifc_mesh.npz",
        "elements": output_dir / "ifc_elements.parquet",
        "materials": output_dir / "ifc_materials.parquet",
        "triangle_element_map": output_dir / "triangle_element_map.npy",
        "triangle_element_guid": output_dir / "triangle_element_guid.npy",
        "failures": output_dir / "ifc_triangulation_failures.json",
    }
    if all(p.exists() for p in cache_paths.values()):
        mesh = np.load(cache_paths["mesh"])
        elements = pd.read_parquet(cache_paths["elements"])
        materials = pd.read_parquet(cache_paths["materials"])
        tri_map = np.load(cache_paths["triangle_element_map"])
        failures = json.loads(cache_paths["failures"].read_text(encoding="utf-8"))
        return {
            **result,
            "triangulated": True,
            "cached": True,
            "element_count": int(len(elements)),
            "material_count": int(len(materials)),
            "vertex_count": int(mesh["vertices"].shape[0]),
            "triangle_count": int(tri_map.shape[0]),
            "failure_count": int(len(failures)),
            "outputs": {k: str(v) for k, v in cache_paths.items()},
        }
    try:
        import ifcopenshell  # type: ignore
        import ifcopenshell.geom  # type: ignore
        import ifcopenshell.util.element  # type: ignore
    except Exception as exc:
        result["error"] = f"ifcopenshell unavailable: {exc}"
        return result

    model = ifcopenshell.open(str(path))
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    vertices_chunks: list[np.ndarray] = []
    faces_chunks: list[np.ndarray] = []
    triangle_element_map: list[np.ndarray] = []
    element_rows: list[dict[str, Any]] = []
    material_rows: list[dict[str, Any]] = []
    material_name_to_id: dict[str, int] = {}
    failures: list[dict[str, str]] = []
    vertex_offset = 0

    def material_names(element: Any) -> list[str]:
        try:
            material = ifcopenshell.util.element.get_material(element)
        except Exception:
            return []
        if material is None:
            return []
        items = material if isinstance(material, (list, tuple)) else [material]
        names: list[str] = []
        for item in items:
            if hasattr(item, "Name") and item.Name:
                names.append(str(item.Name))
            elif hasattr(item, "MaterialLayers"):
                for layer in item.MaterialLayers or []:
                    mat = getattr(layer, "Material", None)
                    if mat is not None and getattr(mat, "Name", None):
                        names.append(str(mat.Name))
            elif hasattr(item, "Materials"):
                for mat in item.Materials or []:
                    if getattr(mat, "Name", None):
                        names.append(str(mat.Name))
        return sorted(set(names))

    for element_index, product in enumerate(model.by_type("IfcProduct")):
        guid = str(getattr(product, "GlobalId", "") or "")
        if not getattr(product, "Representation", None):
            continue
        try:
            shape = ifcopenshell.geom.create_shape(settings, product)
            geom = shape.geometry
            verts = np.asarray(geom.verts, dtype=np.float64).reshape(-1, 3)
            faces = np.asarray(geom.faces, dtype=np.int64).reshape(-1, 3)
            if len(verts) == 0 or len(faces) == 0:
                failures.append({"guid": guid, "ifc_class": product.is_a(), "reason": "empty geometry"})
                continue
            faces_chunks.append(faces + vertex_offset)
            vertices_chunks.append(verts)
            triangle_element_map.append(np.full(len(faces), len(element_rows), dtype=np.int64))
            vertex_offset += len(verts)

            names = material_names(product)
            material_ids = []
            for name in names:
                if name not in material_name_to_id:
                    material_name_to_id[name] = len(material_name_to_id)
                    material_rows.append({"material_id": material_name_to_id[name], "material_name": name, "source": "IFC"})
                material_ids.append(material_name_to_id[name])
            bbox_min = verts.min(axis=0)
            bbox_max = verts.max(axis=0)
            try:
                psets = ifcopenshell.util.element.get_psets(product)
            except Exception:
                psets = {}
            try:
                container = ifcopenshell.util.element.get_container(product)
                container_guid = getattr(container, "GlobalId", None) if container else None
                container_name = getattr(container, "Name", None) if container else None
            except Exception:
                container_guid = None
                container_name = None
            element_rows.append(
                {
                    "element_index": len(element_rows),
                    "GlobalId": guid,
                    "Name": str(getattr(product, "Name", "") or ""),
                    "ObjectType": str(getattr(product, "ObjectType", "") or ""),
                    "ifc_class": product.is_a(),
                    "container_guid": str(container_guid or ""),
                    "container_name": str(container_name or ""),
                    "material_ids": json.dumps(material_ids),
                    "material_names": json.dumps(names, ensure_ascii=False),
                    "psets_json": json.dumps(psets, default=str, ensure_ascii=False),
                    "bbox_min_x": float(bbox_min[0]),
                    "bbox_min_y": float(bbox_min[1]),
                    "bbox_min_z": float(bbox_min[2]),
                    "bbox_max_x": float(bbox_max[0]),
                    "bbox_max_y": float(bbox_max[1]),
                    "bbox_max_z": float(bbox_max[2]),
                    "vertex_count": int(len(verts)),
                    "triangle_count": int(len(faces)),
                }
            )
        except Exception as exc:
            failures.append({"guid": guid, "ifc_class": product.is_a(), "reason": str(exc)})

    if not vertices_chunks or not faces_chunks:
        result["error"] = "no product geometry could be triangulated"
        result["failures"] = failures
        return result

    vertices = np.vstack(vertices_chunks)
    faces = np.vstack(faces_chunks)
    tri_map = np.concatenate(triangle_element_map)
    elements = pd.DataFrame(element_rows)
    materials = pd.DataFrame(material_rows)

    np.savez_compressed(output_dir / "ifc_mesh.npz", vertices=vertices, faces=faces)
    elements.to_parquet(output_dir / "ifc_elements.parquet", index=False)
    materials.to_parquet(output_dir / "ifc_materials.parquet", index=False)
    np.save(output_dir / "triangle_element_map.npy", tri_map)
    np.save(output_dir / "triangle_element_guid.npy", elements["GlobalId"].to_numpy()[tri_map])
    (output_dir / "ifc_triangulation_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")

    result.update(
        {
            "triangulated": True,
            "schema": model.schema,
            "element_count": int(len(elements)),
            "material_count": int(len(materials)),
            "vertex_count": int(len(vertices)),
            "triangle_count": int(len(faces)),
            "failure_count": int(len(failures)),
            "outputs": {
                "mesh": str(output_dir / "ifc_mesh.npz"),
                "elements": str(output_dir / "ifc_elements.parquet"),
                "materials": str(output_dir / "ifc_materials.parquet"),
                "triangle_element_map": str(output_dir / "triangle_element_map.npy"),
                "triangle_element_guid": str(output_dir / "triangle_element_guid.npy"),
                "failures": str(output_dir / "ifc_triangulation_failures.json"),
            },
        }
    )
    return result
