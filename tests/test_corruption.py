from patent_gap.corruption.synthetic import apply_synthetic_corruption
from patent_gap.gap.scoring import compute_patch_scores
from patent_gap.simulation.synthetic_cube import synthetic_scene


def test_spatial_corruption_marks_occluded_patch():
    scene = synthetic_scene(seed=0)
    corrupted = apply_synthetic_corruption(
        scene,
        {"seed": 0, "corruptions": [{"type": "contiguous_patch_removal"}, {"type": "quality_degradation"}]},
    )
    scores = compute_patch_scores(corrupted)
    patch0 = scores[scores["patch_id"] == 0].iloc[0]
    patch4 = scores[scores["patch_id"] == 4].iloc[0]
    assert patch0["D_obs"] > 0.8
    assert patch0["D_geo"] > 0.9
    assert patch4["D_geo"] > 0.5

