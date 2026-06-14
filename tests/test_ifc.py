from pathlib import Path

from patent_gap.ifc.reader import inspect_ifc


def test_ifc_text_audit_handles_missing_file(tmp_path: Path):
    result = inspect_ifc(tmp_path / "missing.ifc")
    assert result["exists"] is False


def test_ifc_text_audit_reads_schema(tmp_path: Path):
    p = tmp_path / "mini.ifc"
    p.write_text(
        "ISO-10303-21;\nHEADER;\nFILE_SCHEMA(('IFC4'));\nENDSEC;\nDATA;\n#1=IFCMATERIAL('steel',$,$);\nENDSEC;\nEND-ISO-10303-21;",
        encoding="utf-8",
    )
    result = inspect_ifc(p)
    assert result["exists"] is True
    assert result["ifc_schema"] == "IFC4"
    assert result["material_count_text"] == 1

