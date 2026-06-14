from __future__ import annotations

from pathlib import Path
from typing import Any

from patent_gap.data.audit import audit_ifc_text


def inspect_ifc(path: str | Path) -> dict[str, Any]:
    return audit_ifc_text(path)


def triangulate_ifc(path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"path": str(path), "output_dir": str(output_dir), "triangulated": False}
    try:
        import ifcopenshell  # type: ignore
        import ifcopenshell.geom  # type: ignore
    except Exception as exc:
        result["error"] = f"ifcopenshell unavailable: {exc}"
        return result
    result["error"] = "Triangulation hook is available, but full CRAS mesh export is not run in smoke mode."
    return result

