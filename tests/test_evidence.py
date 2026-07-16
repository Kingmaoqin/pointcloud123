import pandas as pd

from patent_gap.evidence import (
    compute_controlled_withheld_evidence,
    compute_evidence_from_synthetic,
)


def test_synthetic_evidence_uses_scanner_count_not_hit_count():
    patches = pd.DataFrame(
        [
            {
                "patch_id": 1,
                "element_guid": "A",
                "area": 2.0,
            }
        ]
    )
    elements = pd.DataFrame([{"GlobalId": "A", "element_index": 3}])
    stats = pd.DataFrame(
        [
            {
                "element_index": 3,
                "point_count": 10000,
                "mean_frontality": 0.6,
                "best_frontality": 0.9,
                "valid_views": 9000,
                "scanner_count": 2,
                "valid_scanner_count": 2,
            }
        ]
    )
    evidence = compute_evidence_from_synthetic(patches, elements, stats)
    assert int(evidence.loc[0, "number_of_views"]) == 2
    assert int(evidence.loc[0, "number_of_valid_views"]) == 2


def test_patch_level_stats_do_not_fall_back_to_observed_element():
    patches = pd.DataFrame(
        [
            {"patch_id": 1, "element_guid": "A", "area": 1.0},
            {"patch_id": 2, "element_guid": "A", "area": 1.0},
        ]
    )
    elements = pd.DataFrame([{"GlobalId": "A", "element_index": 3}])
    element_stats = pd.DataFrame(
        [
            {
                "element_index": 3,
                "point_count": 1000,
                "mean_frontality": 0.7,
                "best_frontality": 0.9,
                "valid_views": 10,
                "scanner_count": 2,
                "valid_scanner_count": 2,
            }
        ]
    )
    patch_stats = pd.DataFrame(
        [
            {
                "patch_id": 1,
                "point_count": 500,
                "mean_frontality": 0.7,
                "best_frontality": 0.9,
                "scanner_count": 2,
                "valid_scanner_count": 2,
            }
        ]
    )
    evidence = compute_evidence_from_synthetic(
        patches,
        elements,
        element_stats,
        patch_stats_df=patch_stats,
    ).set_index("patch_id")
    assert bool(evidence.loc[1, "directly_observed"])
    assert not bool(evidence.loc[2, "directly_observed"])
    assert int(evidence.loc[2, "point_count"]) == 0


def test_controlled_withheld_evidence_uses_predeclared_labels():
    patches = pd.DataFrame(
        [
            {"patch_id": 1, "area": 1.0, "gt_missing": False},
            {"patch_id": 2, "area": 1.0, "gt_missing": True},
        ]
    )
    evidence = compute_controlled_withheld_evidence(
        patches,
        seed=0,
        config={"degradation_fraction": 0.0, "missing_leakage_fraction": 0.0},
    ).set_index("patch_id")
    assert bool(evidence.loc[1, "directly_observed"])
    assert evidence.loc[1, "coverage_ratio"] > 0.7
    assert not bool(evidence.loc[2, "directly_observed"])
    assert evidence.loc[2, "coverage_ratio"] == 0.0
