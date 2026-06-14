from patent_gap.gap.scoring import compute_patch_scores
from patent_gap.simulation.synthetic_cube import synthetic_scene
from patent_gap.viewpoints.ranking import generate_candidates, score_candidates


def test_candidate_can_see_occluded_high_gap_surface():
    scene = synthetic_scene(seed=0)
    scores = compute_patch_scores(scene)
    ranked = score_candidates(scores, generate_candidates(scores))
    assert not ranked.empty
    assert int(ranked.iloc[0]["target_patch_id"]) == 0
    assert "0" in str(ranked.iloc[0]["visible_patch_ids"]).split(";")

