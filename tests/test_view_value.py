from patent_gap.gap.scoring import compute_patch_scores
from patent_gap.simulation.closed_loop import run_closed_loop
from patent_gap.simulation.synthetic_cube import synthetic_scene
from patent_gap.viewpoints.ranking import generate_candidates, greedy_sequential_ranking, score_candidates


def test_view_value_ranks_gap_view_first():
    scene = synthetic_scene(seed=0)
    scores = compute_patch_scores(scene)
    candidates = generate_candidates(scores)
    ranked = score_candidates(scores, candidates)
    assert ranked.iloc[0]["value"] > ranked.iloc[-1]["value"]
    assert int(ranked.iloc[0]["target_patch_id"]) == 0


def test_greedy_ranking_returns_unique_views():
    scene = synthetic_scene(seed=0)
    scores = compute_patch_scores(scene)
    ranked = greedy_sequential_ranking(scores, generate_candidates(scores), k=3)
    assert len(ranked["view_id"]) == len(set(ranked["view_id"]))


def test_closed_loop_reduces_remaining_gap_area():
    history, _, _ = run_closed_loop({"seed": 0, "simulation": {"steps": 5, "recovery_per_view": 0.85}})
    assert len(history) == 5
    assert history["remaining_gap_area"].iloc[-1] < history["remaining_gap_area"].iloc[0]

