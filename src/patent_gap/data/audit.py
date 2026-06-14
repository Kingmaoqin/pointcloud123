from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


def file_md5(path: str | Path, chunk_size: int = 1024 * 1024) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_ifc_text(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    out: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8", errors="ignore")
    out["size_bytes"] = path.stat().st_size
    schema = re.search(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']+)'", text, flags=re.IGNORECASE)
    out["ifc_schema"] = schema.group(1) if schema else "UNKNOWN"
    entities = re.findall(r"=\s*(IFC[A-Z0-9_]+)\s*\(", text, flags=re.IGNORECASE)
    counter = Counter(e.upper() for e in entities)
    out["entity_count"] = int(sum(counter.values()))
    out["ifcproduct_like_count"] = int(sum(v for k, v in counter.items() if k.startswith("IFC") and k not in {"IFCPERSON", "IFCORGANIZATION"}))
    out["top_entities"] = dict(counter.most_common(25))
    out["material_count_text"] = int(counter.get("IFCMATERIAL", 0))
    out["material_properties_count_text"] = int(counter.get("IFCMATERIALPROPERTIES", 0))
    out["has_ifcopenshell"] = False
    try:
        import ifcopenshell  # type: ignore

        model = ifcopenshell.open(str(path))
        products = model.by_type("IfcProduct")
        materials = model.by_type("IfcMaterial")
        out["has_ifcopenshell"] = True
        out["ifcopenshell_product_count"] = len(products)
        out["ifcopenshell_material_count"] = len(materials)
    except Exception as exc:
        out["ifcopenshell_error"] = str(exc)
    return out


def audit_zip(path: str | Path, limit: int = 100) -> dict[str, Any]:
    path = Path(path)
    out: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return out
    out["size_bytes"] = path.stat().st_size
    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            out["file_count"] = len(infos)
            out["first_files"] = [i.filename for i in infos[:limit]]
            out["total_uncompressed_bytes"] = int(sum(i.file_size for i in infos))
            suffixes = Counter(Path(i.filename).suffix.lower() for i in infos)
            out["suffix_counts"] = dict(suffixes)
    except Exception as exc:
        out["zip_error"] = str(exc)
    return out


def write_data_audit(config: dict[str, Any], out_path: str | Path) -> dict[str, Any]:
    ifc_path = config.get("ifc_path", "data/raw/craslabbim.ifc")
    zip_path = config.get("pointcloud_zip", "data/raw/craslabannotated.zip")
    edf_dir = Path(config.get("edf_dir", "data/raw/edf"))
    ts40k_repo = Path(config.get("ts40k_repo", "external/TS40K"))
    audit = {
        "ifc": audit_ifc_text(ifc_path),
        "pointcloud_zip": audit_zip(zip_path),
        "md5": {
            "ifc": file_md5(ifc_path),
            "pointcloud_zip": file_md5(zip_path),
        },
        "edf": {
            "path": str(edf_dir),
            "exists": edf_dir.exists(),
            "note": "EDF Challenge requires user login after download; continuing CRAS primary experiment." if not edf_dir.exists() else "EDF directory exists.",
        },
        "ts40k": {
            "path": str(ts40k_repo),
            "exists": ts40k_repo.exists(),
            "note": "TS40K repository is optional and should not download full large data before CRAS smoke completion.",
        },
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Data Audit", "", "```json", json.dumps(audit, indent=2, ensure_ascii=False), "```", ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return audit

