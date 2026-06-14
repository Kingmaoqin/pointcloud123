from patent_gap.simulation.synthetic_cube import synthetic_scene


def test_synthetic_cube_has_six_surface_patches():
    scene = synthetic_scene(seed=0)
    patches = scene["patches"]
    assert len(patches) == 6
    assert patches["area"].sum() == 24.0
    assert set(patches["patch_id"]) == {0, 1, 2, 3, 4, 5}

