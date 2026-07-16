import pandas as pd
import pytest

from patent_gap.gap.scoring import compute_patch_scores, rank_components
from patent_gap.simulation.synthetic_cube import synthetic_scene


def test_gap_scores_find_required_smoke_conditions():
    scene = synthetic_scene(seed=0)
    scores = compute_patch_scores(scene)
    by_patch = scores.set_index("patch_id")
    assert by_patch.loc[0, "D_obs"] > 0.8
    assert by_patch.loc[1, "D_sem"] > 0.5
    assert by_patch.loc[1, "D_mat_conflict"] == 1.0
    assert by_patch.loc[2, "D_mat_missing"] == 1.0
    assert by_patch.loc[3, "D_ang"] > 0.7
    assert by_patch.loc[4, "D_geo"] > 0.5


def test_component_ranking_prioritizes_occluded_component():
    scene = synthetic_scene(seed=0)
    scores = compute_patch_scores(scene)
    components = rank_components(scores)
    assert components.iloc[0]["element_guid"] == "GUID_OCCLUDED_WALL"


def test_missing_observation_row_preserves_patch_and_marks_observation_gap():
    scene = synthetic_scene(seed=0)
    scene["observations"] = scene["observations"].query("patch_id != 0")
    scores = compute_patch_scores(scene).set_index("patch_id")
    assert len(scores) == 6
    assert scores.loc[0, "D_obs"] == 1.0


def test_duplicate_evidence_patch_ids_are_rejected():
    scene = synthetic_scene(seed=0)
    scene["observations"] = pd.concat(
        [scene["observations"], scene["observations"].iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="duplicate patch_id"):
        compute_patch_scores(scene)
