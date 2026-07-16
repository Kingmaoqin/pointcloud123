"""PR1 验收: Schema 往返测试; 非法字段拒绝。"""

import numpy as np
import pytest

from patent_gap.io.station_npz import StationScan, load_station, save_station
from patent_gap.sensors.model import DEFAULT_TLS, SensorModel


def _scan(n=100):
    rng = np.random.default_rng(0)
    return StationScan(
        points=rng.normal(size=(n, 3)),
        origin=np.array([1.0, 2.0, 2.0]),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        station_id="s0",
        sensor=DEFAULT_TLS,
        intensity=rng.random(n),
        hit_patch=rng.integers(-1, 5, n),
    )


def test_roundtrip(tmp_path):
    scan = _scan()
    p = tmp_path / "station_0.npz"
    save_station(p, scan)
    back = load_station(p)
    np.testing.assert_allclose(back.points, scan.points, atol=1e-6)
    np.testing.assert_allclose(back.origin, scan.origin)
    assert back.station_id == "s0"
    assert back.frame == "global"
    assert back.sensor == DEFAULT_TLS
    np.testing.assert_array_equal(back.hit_patch, scan.hit_patch)


def test_reject_bad_frame():
    with pytest.raises(ValueError, match="frame"):
        StationScan(points=np.zeros((1, 3)), origin=np.zeros(3),
                    quat_wxyz=np.array([1.0, 0, 0, 0]), station_id="x",
                    sensor=DEFAULT_TLS, frame="local")


def test_reject_bad_quat():
    with pytest.raises(ValueError, match="quat"):
        StationScan(points=np.zeros((1, 3)), origin=np.zeros(3),
                    quat_wxyz=np.array([2.0, 0, 0, 0]), station_id="x",
                    sensor=DEFAULT_TLS)


def test_reject_unknown_key(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, points=np.zeros((1, 3), dtype=np.float32), origin=np.zeros(3),
             quat_wxyz=np.array([1.0, 0, 0, 0]), station_id="x",
             timestamp=0.0, frame="global", sensor=DEFAULT_TLS.to_json(),
             rogue_field=np.arange(3))
    with pytest.raises(ValueError, match="unknown keys"):
        load_station(p)


def test_sensor_validation():
    with pytest.raises(ValueError):
        SensorModel(r_min=5, r_opt=3, r_max=1, dtheta=0.001, dphi=0.001,
                    sigma_r=0.005, f_pulse=1e5)
