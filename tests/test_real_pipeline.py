import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_real_pipeline.py"
_SPEC = importlib.util.spec_from_file_location("run_real_pipeline", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
run_real_pipeline = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_real_pipeline)


def test_real_pipeline_preserves_precomputed_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(run_real_pipeline, "TABLES", tmp_path)
    patches = pd.DataFrame(
        [
            {
                "patch_id": 1,
                "element_guid": "A",
                "ifc_class": "IfcWall",
                "name": "wall",
                "centroid": (0.0, 0.0, 0.0),
                "normal": (1.0, 0.0, 0.0),
                "area": 2.0,
                "engineering_importance": 0.5,
            }
        ]
    )
    evidence = pd.DataFrame(
        [
            {
                "patch_id": 1,
                "directly_observed": True,
                "number_of_views": 0,
                "number_of_valid_views": 0,
                "best_frontality": np.nan,
                "mean_frontality": np.nan,
                "angular_diversity": np.nan,
                "coverage_ratio": 0.4,
                "density_ratio": 0.4,
                "projected_resolution_quality": np.nan,
                "registration_confidence": 0.9,
                "point_density": 200.0,
                "source_type": "MEASURED",
                "D_obs": 0.2,
                "D_ang": np.nan,
                "D_geo": 0.6,
            }
        ]
    )
    semantics = pd.DataFrame(
        [{"patch_id": 1, "p_bim": np.array([1.0]), "p_obs": np.array([1.0])}]
    )
    materials = pd.DataFrame(
        [
            {
                "patch_id": 1,
                "material_present": True,
                "material_name": "concrete",
                "material_conflict": False,
                "visual_material_category": "concrete",
                "D_mat_missing": 0.0,
                "D_mat_conflict": 0.0,
            }
        ]
    )

    scores = run_real_pipeline.step6_gap_scores(
        patches,
        evidence,
        semantics,
        materials,
        {},
    )

    assert scores.loc[0, "D_obs"] == 0.2
    assert scores.loc[0, "D_geo"] == 0.6
    assert np.isnan(scores.loc[0, "D_ang"])
