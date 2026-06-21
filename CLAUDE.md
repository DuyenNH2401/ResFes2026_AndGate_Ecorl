# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ResFes2026_AndGate_Ecorl** — Reinforcement learning for adaptive traffic signal control and autonomous eco-driving in SUMO. Uses **Phasic Policy Gradient (PPG)** with a pluggable backbone architecture. The backbone interface (`BaseBackbone` + `BACKBONE_REGISTRY` + `create_backbone()`) is in place, but **only the Mamba SSM backbone is currently implemented/registered** (`backbones/mamba.py`). DNN/LSTM/biLSTM/GRU/RNN are roadmap items — not yet written. Agents navigate 3×3 grid intersections with intelligent traffic lights.

## Critical Setup

Requires `SUMO_HOME` environment variable set to a SUMO (Simulation of Urban MObility) installation path. The simulation environment (`simulation/env.py`) will raise `sys.exit` if not set.

No `requirements.txt` or `setup.py` exists. Key dependencies: `torch`, `numpy`, `gymnasium`, `traci` (from SUMO), `pyyaml`.

## Backbone Status

The backbone interface exists and works:
- `backbones/__init__.py` — exports `BACKBONE_REGISTRY` (dict) and `create_backbone()` (factory).
- `backbones/base.py` — exports `BaseBackbone` (base class used by `mamba.py`).

**Only `mamba` is registered** in `BACKBONE_REGISTRY`. The CLI `--backbone` choices are derived from the registry, so currently only `--backbone mamba` is valid. Adding DNN/LSTM/biLSTM/GRU/RNN requires writing the corresponding files and registering them.

## Project Architecture

```
ecorl_adaptive_shaping/
├── train_ppg.py          # Main training script (CLI entry point)
├── test_ppg.py           # Evaluation script — tests trained model on ALL routes
├── ppg/                  # PPG algorithm implementation
│   ├── core.py           # PPGAgent, PolicyModel, ValueModel
│   ├── memory.py         # PPGMemory (on-policy buffer), AuxiliaryBuffer
│   ├── distributions.py   # Continuous action distribution (Gaussian)
│   └── ppg_config.py     # Default hyperparameters
├── backbones/            # Neural network backbones (pluggable)
│   └── mamba.py          # Mamba SSM backbone + BaseBackbone base class
├── simulation/           # SUMO environment
│   ├── env.py            # SumoEnv — custom Gymnasium environment (45-dim obs, 2-dim action)
│   └── config.py         # Vehicle physics constants (VF9 EV parameters)
├── configs/              # YAML config files
│   └── ppg_default.yaml  # Default training hyperparameters
└── maps/                 # SUMO road network configurations
    ├── grid_3_3_intelligent_tls/        # 3×3 grid map (base)
    └── grid_3_3_intelligent_tls_1800/   # 3×3 grid map (1800s variant)
```

## Core Algorithm: Phasic Policy Gradient (PPG)

**Phase 1 (Policy Phase):** Standard PPO with dual critics. `PolicyModel` has a shared backbone with Actor + Auxiliary Critic heads. `ValueModel` has its own independent backbone + critic head. PPO clipped surrogate objective with separate networks.

**Phase 2 (Auxiliary Phase):** Triggered every `N_aux` policy updates. Joint loss: (1) value distillation from separate Value Network to Policy Network's auxiliary critic, (2) direct return regression, (3) **adaptive KL** divergence constraint (coefficient `β` weighted, `self.beta_kl * l_kl`) to prevent policy drift from old parameters stored in `AuxiliaryBuffer`. After each auxiliary phase, the measured mean KL adapts `β` toward target `d_targ`: `kl > d_targ·1.5 → β×2`; `kl < d_targ/1.5 → β÷2`; clamped to `[beta_kl_min, beta_kl_max]`.

Key PPG parameters: `n_aux=5`, `k_aux=10`, `beta_kl=5.0` (initial, adaptive), `d_targ=0.03` (KL target), `clip_val=10.0`.

## Observation & Action Space

- **Observation:** 45-dim vector — 6 ego physics (speed, accel, energy, lane, slope, lat_offset) + 2 leader (dist, rel_speed) + 3 infra (speed_limit, tls_dist, tls_state) + 2 lane availability + 32 surroundings (8 closest vehicles × 4 features: dx, dy, rel_speed, rel_heading)
- **Action:** 2-dim continuous `[-1, 1]` — `action[0]` = lane change intent (|val| > 0.3 triggers change), `action[1]` = speed control mapped to `[0, MAX_SPEED]`
- Surrounding vehicle detection switches between two modes: **context subscription** (near junctions, TLS-aware filtering) and **getNeighbors** (straight roads, 4-directional)

## Reward Function

Multi-objective reward with 10 components:
| Component | Weight | Description |
|-----------|--------|-------------|
| Speed target | +0.3 | Bell-curve around speed limit |
| Too slow | -0.1 | Penalty when below minimum speed (disabled near junctions) |
| Progress | +0.187 | Distance toward destination per step |
| Energy | -0.012 | Wh/m efficiency penalty |
| Comfort | -0.029 | Acceleration jerk between steps |
| Lane change | -0.1 | Quadratic penalty on lane-change action magnitude |
| Safety | -0.019 | TTC-based following distance penalty |
| Red light | -25.0 | Violation penalty (1-time per crossing) |
| Junction approach | -0.258 | Speed excess approaching junctions |
| Time | -0.044 | Per-step living penalty |

The action planner (`_apply_action_planner`) overrides agent actions using **RSS (Responsibility-Sensitive Safety)** longitudinal and lateral safety checks.

## Key Commands

```bash
# Train (only mamba is currently registered)
python train_ppg.py --backbone mamba
python train_ppg.py --backbone mamba --hidden-size 256 --num-layers 3

# Train with config file
python train_ppg.py --config configs/ppg_default.yaml --backbone mamba

# Train with adaptive reward arms
python train_ppg.py --backbone mamba --lagrangian-enabled true     # Lagrangian constrained shaping
python train_ppg.py --backbone mamba --curriculum-enabled true     # curriculum energy weighting

# Test/evaluate a trained model
python test_ppg.py --backbone mamba -m Mamba_PPG/runs/<run_id>/models/Mamba_PPG_best --no-gui

# Test specific map/route
python test_ppg.py --backbone mamba -m Mamba_PPG/models/Mamba_PPG_best -map maps/grid_3_3_intelligent_tls_1800/run.sumocfg
```

Output goes to `{BACKBONE}_PPG/runs/{exp_name}_seed{seed}_{timestamp}/` with:
- `models/` — saved checkpoints (`_policy.tar`, `_value.tar` pairs)
- `training_log.csv` — per-episode metrics
- `run_config.yaml` — full hyperparameter dump
- `eval/` — per-route test results
