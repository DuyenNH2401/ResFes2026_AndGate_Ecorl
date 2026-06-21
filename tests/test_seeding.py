"""Tests cho seeding (config 3-lớp + SUMO --seed + cudnn deterministic)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.env import SumoEnv  # noqa: E402


def test_build_sumo_cmd_adds_seed_when_set():
    env = SumoEnv.__new__(SumoEnv)
    env.TRAFFIC_SCALE = 0.8
    env.delay = 0
    env.sim_seed = 123
    cmd = env._build_sumo_cmd("sumo", "map.sumocfg", [])
    assert "--seed" in cmd
    assert cmd[cmd.index("--seed") + 1] == "123"


def test_build_sumo_cmd_no_seed_when_none():
    env = SumoEnv.__new__(SumoEnv)
    env.TRAFFIC_SCALE = 0.8
    env.delay = 0
    env.sim_seed = None
    cmd = env._build_sumo_cmd("sumo", "map.sumocfg", [])
    assert "--seed" not in cmd


def test_build_sumo_cmd_keeps_route_arg():
    env = SumoEnv.__new__(SumoEnv)
    env.TRAFFIC_SCALE = 0.8
    env.delay = 0
    env.sim_seed = None
    cmd = env._build_sumo_cmd("sumo-gui", "m.sumocfg", ["-a", "r.rou.xml"])
    assert cmd[0] == "sumo-gui"
    assert "-a" in cmd and "r.rou.xml" in cmd


def test_set_seed_enables_cudnn_deterministic():
    import torch
    import train_ppg
    train_ppg.set_seed(7)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
