"""Tests cho adaptive reward (Lagrangian + curriculum).

Test trực tiếp logic reward bằng stub kế thừa SumoEnv KHÔNG khởi động SUMO/TraCI.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.env import SumoEnv  # noqa: E402


def _make_stub_env(adaptive: bool):
    """Tạo env mà KHÔNG gọi __init__ (tránh khởi động SUMO)."""
    env = SumoEnv.__new__(SumoEnv)
    # Hằng số reward mà _calculate_reward đọc qua self.*
    env.MAX_SPEED = 55.6
    env.TARGET_SPEED_RATIO = 1.0
    env.MIN_DESIRED_SPEED = 5.0
    env.JUNCTION_APPROACH_DIST = 15.0
    env.JUNCTION_TARGET_SPEED = 2.0
    env.JUNCTION_MIN_SPEED = 0.0
    env.adaptive_reward_enabled = adaptive
    # veh_data giả: xe chạy thẳng, không leader, không TLS
    env.veh_data = {
        "speed": 10.0,
        "lane_id": "",
        "road_id": "edgeA",
        "elec": 1.0,
        "leader": None,
        "tls": None,
    }
    # trạng thái stateful mà hàm dùng
    env.prev_dist = 100.0
    env.prev_action = np.array([0.0, 0.0], dtype=np.float32)
    # tránh TraCI: override 2 helper gọi traci
    env._get_junction_proximity = lambda: 999.0
    env._get_dist_to_destination = lambda: 95.0
    return env


def _scalar_from_components(env, action):
    """Tính lại reward gốc (9 thành phần + W_TIME) độc lập để so đẳng thức.

    Lấy weights đúng như trong env._calculate_reward.
    """
    rt, cs, cc, cr = _calc_tuple(env, action)
    W_COMFORT = -0.0287413370864677
    W_SAFETY = -0.019335217679737792
    W_RED_LIGHT = -25.0
    return rt + cc * W_COMFORT + cs * W_SAFETY + cr * W_RED_LIGHT


def _calc_tuple(env, action):
    """Bật adaptive tạm thời để lấy tuple thành phần."""
    saved = env.adaptive_reward_enabled
    env.adaptive_reward_enabled = True
    out = env._calculate_reward(action)
    env.adaptive_reward_enabled = saved
    return out


def test_calculate_reward_returns_tuple_when_adaptive():
    env = _make_stub_env(adaptive=True)
    out = env._calculate_reward(np.array([0.0, 0.0], dtype=np.float32))
    assert isinstance(out, tuple)
    assert len(out) == 4
    reward_task, cost_safety, cost_comfort, cost_redlight = out
    assert isinstance(reward_task, float)
    assert cost_safety >= 0.0
    assert cost_comfort >= 0.0
    assert cost_redlight >= 0.0


def test_calculate_reward_returns_scalar_when_not_adaptive():
    env = _make_stub_env(adaptive=False)
    out = env._calculate_reward(np.array([0.0, 0.0], dtype=np.float32))
    assert isinstance(out, float)


def test_scalar_equals_original_formula():
    """Backward-compat: nhánh scalar phải BẰNG HỆT công thức gốc 9 phần + W_TIME."""
    action = np.array([0.4, 0.7], dtype=np.float32)

    env = _make_stub_env(adaptive=False)
    env.prev_action = np.array([0.0, 0.0], dtype=np.float32)
    scalar = env._calculate_reward(action)

    env2 = _make_stub_env(adaptive=True)
    env2.prev_action = np.array([0.0, 0.0], dtype=np.float32)
    expected = _scalar_from_components(env2, action)

    assert abs(scalar - expected) < 1e-9
