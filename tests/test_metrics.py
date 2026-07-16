import math

import pandas as pd

from patent_gap.evaluation.metrics import patch_detection_metrics


def test_primary_f1_uses_declared_threshold_and_reports_oracle_separately():
    scores = pd.DataFrame(
        {
            "gt_missing": [0, 1],
            "G_gap": [0.40, 0.45],
        }
    )
    metrics = patch_detection_metrics(scores, decision_threshold=0.5)
    assert metrics["F1"] == 0.0
    assert metrics["F1_threshold"] == 0.5
    assert metrics["F1_oracle"] > metrics["F1"]


def test_single_class_auc_is_not_fabricated_as_zero():
    scores = pd.DataFrame(
        {
            "gt_missing": [0, 0],
            "G_gap": [0.1, 0.2],
        }
    )
    metrics = patch_detection_metrics(scores)
    assert math.isnan(metrics["AUROC"])
    assert math.isnan(metrics["AUPRC"])
