from pathlib import Path

import numpy as np
import pandas as pd

from patent_gap.webapp import _get_state, _load_uploaded_state, _save_state, build_app, run_scan_steps


def test_webapp_builds_and_imported_mesh_can_run_scan(tmp_path: Path):
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    model_path = tmp_path / "triangle.npz"
    np.savez_compressed(model_path, vertices=vertices, faces=faces)
    score_path = tmp_path / "scores.csv"
    pd.DataFrame({"patch_id": [0], "G_gap": [0.8]}).to_csv(score_path, index=False)

    app = build_app()
    assert app is not None

    state = _load_uploaded_state(model_path, score_path)
    before = float(state["scores"]["G_gap"].iloc[0])
    state_key = _save_state(state)
    outputs = run_scan_steps(state_key, 0.8, 0.55, 3, 1)
    updated_state = _get_state(outputs[0])
    after = float(updated_state["scores"]["G_gap"].iloc[0])

    assert len(updated_state["history"]) == 1
    assert after < before
