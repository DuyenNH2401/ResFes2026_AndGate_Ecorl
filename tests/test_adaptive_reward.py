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


def test_split_reward_for_step_tuple_and_scalar():
    """_split_reward_for_step chuẩn hoá output về (reward_task, cs, cc, cr)."""
    env = _make_stub_env(adaptive=True)
    # tuple-4 (adaptive) → giữ nguyên
    rt, cs, cc, cr = env._split_reward_for_step((1.5, 0.3, 0.2, 0.1))
    assert (rt, cs, cc, cr) == (1.5, 0.3, 0.2, 0.1)
    # scalar (baseline) → cost = 0
    rt, cs, cc, cr = env._split_reward_for_step(2.7)
    assert rt == 2.7
    assert cs == 0.0 and cc == 0.0 and cr == 0.0


# ---------------------------------------------------------------------------
#  Task 4 — Lagrangian dual update
# ---------------------------------------------------------------------------
from ppg.core import PPGAgent  # noqa: E402


def _make_lagrangian_agent():
    """Agent tối giản, KHÔNG dựng network thật — chỉ test logic λ."""
    agent = PPGAgent.__new__(PPGAgent)
    agent.lagrangian_enabled = True
    agent.lambda_safety = 1.0
    agent.lambda_comfort = 0.1
    agent.lambda_redlight = 25.0
    agent.lambda_lr = 0.1
    agent.cost_limit_safety = 0.02
    agent.cost_limit_comfort = 0.30
    agent.cost_limit_redlight = 0.0
    agent.lambda_max = 50.0
    return agent


def test_update_lambdas_moves_correct_direction():
    agent = _make_lagrangian_agent()
    # cost_safety mean=0.5 >> limit 0.02 → λ_safety TĂNG
    # cost_comfort mean=0.0 < limit 0.30 → λ_comfort GIẢM
    # cost_redlight mean=0.1 > limit 0.0 → λ_redlight TĂNG
    out = agent.update_lambdas(mean_cost_safety=0.5, mean_cost_comfort=0.0,
                               mean_cost_redlight=0.1)
    assert out["lambda_safety"] > 1.0
    assert out["lambda_comfort"] < 0.1
    assert out["lambda_comfort"] >= 0.0     # luôn không âm
    assert out["lambda_redlight"] > 25.0


def test_update_lambdas_clamps_at_zero_and_max():
    agent = _make_lagrangian_agent()
    agent.lambda_comfort = 0.001
    agent.lambda_lr = 1.0
    # cost rất thấp → λ_comfort muốn âm nhưng phải clamp tại 0
    out = agent.update_lambdas(mean_cost_safety=0.0, mean_cost_comfort=0.0,
                               mean_cost_redlight=0.0)
    assert out["lambda_comfort"] == 0.0
    # cost rất cao → λ_safety đụng trần lambda_max
    agent.lambda_safety = 49.9
    out = agent.update_lambdas(mean_cost_safety=1000.0, mean_cost_comfort=0.0,
                               mean_cost_redlight=0.0)
    assert out["lambda_safety"] == 50.0


def test_compute_gae_subtracts_effective_cost():
    """compute_gae trừ λ·cost khi lagrangian bật (dùng λ hiện tại của rollout)."""
    agent = PPGAgent.__new__(PPGAgent)
    agent.lagrangian_enabled = True
    agent.lambda_safety = 2.0
    agent.lambda_comfort = 1.0
    agent.lambda_redlight = 10.0
    agent.gamma = 0.99
    agent.lam = 0.95

    class _Mem:
        rewards = [1.0, 1.0, 1.0]
        dones = [0.0, 0.0, 1.0]
        value_vals = [0.0, 0.0, 0.0]
        cost_safety = [0.5, 0.0, 0.0]
        cost_comfort = [0.0, 0.5, 0.0]
        cost_redlight = [0.0, 0.0, 0.1]
    agent.policy_memory = _Mem()

    adv, ret = agent.compute_gae(0.0)
    # r_eff[0] = 1 - 2*0.5 = 0 ; r_eff[1] = 1 - 1*0.5 = 0.5 ; r_eff[2] = 1 - 10*0.1 = 0
    assert adv.shape == (3,)
    assert ret.shape == (3,)


def test_compute_gae_no_subtract_when_disabled():
    agent = PPGAgent.__new__(PPGAgent)
    agent.lagrangian_enabled = False
    agent.gamma = 0.99
    agent.lam = 0.95

    class _Mem:
        rewards = [1.0, 1.0]
        dones = [0.0, 1.0]
        value_vals = [0.0, 0.0]
        cost_safety = [9.0, 9.0]
        cost_comfort = [9.0, 9.0]
        cost_redlight = [9.0, 9.0]
    agent.policy_memory = _Mem()
    adv, ret = agent.compute_gae(0.0)
    # returns không bị ảnh hưởng bởi cost khi tắt → return[1] = reward[1] = 1.0
    assert abs(ret[1] - 1.0) < 1e-6


