import math

import pandas as pd

from patent_gap.semantics import compute_semantic_scores


def test_unknown_classification_is_missing_semantic_evidence():
    patches = pd.DataFrame(
        [
            {
                "patch_id": 1,
                "element_guid": "A",
                "ifc_class": "IfcWall",
                "normal_z": 0.0,
            }
        ]
    )
    assoc = pd.DataFrame(
        [
            {
                "matched": True,
                "element_guid": "A",
                "classification": 0,
            }
        ]
    )
    scores = compute_semantic_scores(patches, assoc)
    assert math.isnan(float(scores.loc[0, "D_sem"]))
