from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from patent_gap.data.pointcloud import associate_points_to_ifc, iter_cras_asc_chunks


def test_iter_cras_asc_chunks_reads_valid_rows(tmp_path: Path):
    zpath = tmp_path / "mini.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr(
            "mini.asc",
            "//X Y Z R G B Intensity Classification\n"
            "0 0 0 1 2 3 0.1 7\n"
            "bad row\n"
            "1 0 0 4 5 6 0.2 8\n"
            "2 0 0 7 8 9 0.3 9\n",
        )
    chunks = list(iter_cras_asc_chunks(zpath, chunk_size=2))
    assert len(chunks) == 2
    assert chunks[0][0] == 0
    assert len(chunks[0][1]) == 2
    assert chunks[1][0] == 1
    assert len(chunks[1][1]) == 1


def test_associate_points_to_ifc_maps_triangle_to_guid(tmp_path: Path):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    np.savez_compressed(tmp_path / "ifc_mesh.npz", vertices=vertices, faces=faces)
    np.save(tmp_path / "triangle_element_map.npy", np.array([0], dtype=np.int64))
    pd.DataFrame([{"GlobalId": "GUID_A", "ifc_class": "IfcSlab"}]).to_parquet(tmp_path / "ifc_elements.parquet", index=False)
    points = pd.DataFrame(
        [
            {"x": 0.1, "y": 0.1, "z": 0.001, "r": 1, "g": 2, "b": 3, "intensity": 0.5, "classification": 1},
            {"x": 0.1, "y": 0.1, "z": 1.0, "r": 1, "g": 2, "b": 3, "intensity": 0.5, "classification": 1},
        ]
    )
    summary = associate_points_to_ifc(
        points,
        tmp_path / "ifc_mesh.npz",
        tmp_path / "triangle_element_map.npy",
        tmp_path / "ifc_elements.parquet",
        tmp_path,
        association_distance=0.01,
        enable_translation_calibration=False,
    )
    out = pd.read_parquet(tmp_path / "cras_point_sample_associations.parquet")
    assert summary["matched_points"] == 1
    assert out.loc[0, "element_guid"] == "GUID_A"
    assert out.loc[0, "ifc_class"] == "IfcSlab"
    assert out.loc[1, "ifc_class"] == "unmatched"

