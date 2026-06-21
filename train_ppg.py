"""
Unified PPG Training Script — Multi-Backbone
=============================================
Train PPG with any backbone via CLI argument:

    python train_ppg.py --backbone dnn
    python train_ppg.py --backbone lstm
    python train_ppg.py --backbone mamba
    python train_ppg.py --backbone bilstm --hidden-size 256 --num-layers 3

Output automatically goes to {BACKBONE}_PPG/ folder:
    DNN_PPG/models/    DNN_PPG/reports/
    LSTM_PPG/models/   LSTM_PPG/reports/
    Mamba_PPG/models/  Mamba_PPG/reports/
    ...
"""

import os
import sys
import csv
import time
import argparse
import torch
import numpy as np
import datetime as dt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ppg.core import PPGAgent
from simulation.env import SumoEnv
from backbones import BACKBONE_REGISTRY

# ============================================================
#  DEFAULT CONFIGURATION
# ============================================================
MAP_CONFIGS = [
    "maps/grid_3_3_intelligent_tls_1800/run.sumocfg"
]

CSV_HEADER = [
    "episode", "steps", "ep_reward", "avg_speed", "total_energy",
    "wiggle", "safety", "success", "reason", "route", "override_rate", "avg_jerk"
]


def set_seed(seed=55):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
#  CLI ARGUMENT PARSER
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified PPG training with selectable backbone architecture.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_ppg.py --backbone dnn
  python train_ppg.py --config ppg_config.yaml --backbone lstm
        """
    )

    # Config file
    parser.add_argument("--config", type=str, default="configs/ppg_default.yaml", help="Path to YAML config file")

    # Required
    parser.add_argument(
        "--backbone", required=True,
        choices=list(BACKBONE_REGISTRY.keys()),
        help="Backbone architecture to use"
    )

    # PPO/PPG hyperparameters
    parser.add_argument("--entropy-coef", type=float, default=None, help="Entropy coefficient (default: loads from ppg_config.py)")
    parser.add_argument("--vf-loss-coef", type=float, default=None, help="Value function loss coefficient (default: loads from ppg_config.py)")
    parser.add_argument("--batchsize", type=int, default=None, help="Mini-batch size (default: loads from ppg_config.py)")
    parser.add_argument("--ppo-epochs", type=int, default=None, help="PPO update epochs (default: loads from ppg_config.py)")
    parser.add_argument("--n-update", type=int, default=None, help="Steps between updates (default: loads from ppg_config.py)")
    parser.add_argument("--gamma", type=float, default=None, help="Discount factor (default: loads from ppg_config.py)")
    parser.add_argument("--lam", type=float, default=None, help="GAE lambda (default: loads from ppg_config.py)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: loads from ppg_config.py)")
    parser.add_argument("--use-history", type=str, default="auto", choices=["auto", "true", "false"], help="Whether to use rolling temporal history (default: auto)")

    # PPG-specific hyperparameters
    parser.add_argument("--lr-policy", type=float, default=None, help="Policy learning rate (default: loads from ppg_config.py)")
    parser.add_argument("--lr-value", type=float, default=None, help="Separate Value learning rate (default: loads from ppg_config.py)")
    parser.add_argument("--n-aux", type=int, default=None, help="Auxiliary phase interval in terms of policy phases (default: loads from ppg_config.py)")
    parser.add_argument("--k-aux", type=int, default=None, help="Auxiliary update epochs (default: loads from ppg_config.py)")
    parser.add_argument("--beta-kl", type=float, default=None, help="Target KL penalty coefficient (default: loads from ppg_config.py)")
    parser.add_argument("--d-targ", type=float, default=None, help="Target KL divergence value (default: loads from ppg_config.py)")
    parser.add_argument("--clip-val", type=float, default=None, help="Value clipping range in PPG (default: loads from ppg_config.py)")
    parser.add_argument("--clip-eps", type=float, default=None, help="Policy clipping range in PPG (default: loads from ppg_config.py)")

    # Backbone hyperparameters (shared)
    parser.add_argument("--d-model", type=int, default=128, help="Feature dimension (default: 128)")
    parser.add_argument("--hidden-size", type=int, default=128, help="RNN/GRU/LSTM hidden size (default: 128)")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256], help="DNN hidden layer sizes (default: 256 256)")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers (default: 2)")
    parser.add_argument("--seq-len", type=int, default=5, help="Sequence length for reshaping (default: 5)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (default: 0.1)")

    # Mamba-specific
    parser.add_argument("--d-state", type=int, default=16, help="Mamba SSM state dimension (default: 16)")
    parser.add_argument("--d-conv", type=int, default=4, help="Mamba convolution kernel size (default: 4)")
    parser.add_argument("--n-layers", type=int, default=2, help="Mamba number of blocks (default: 2)")

    # Training control
    parser.add_argument("--seed", type=int, default=55, help="Random seed (default: 55)")
    parser.add_argument("--exp-name", type=str, default="run", help="Experiment name for logging (default: run)")
    parser.add_argument("--n-episode", type=int, default=5000, help="Total episodes (default: 5000)")
    parser.add_argument("--n-saved", type=int, default=100, help="Checkpoint interval (default: 100)")
    parser.add_argument("--load-model", type=str, default="", help="Path prefix to load model weights")
    parser.add_argument("--render", type=str, default="false", help="Render SUMO GUI (default: false)")
    parser.add_argument("--map", type=str, nargs="+", default=["maps/grid_3_3_intelligent_tls_1800/run.sumocfg"], help="Map config file(s)")

    # First parse to get the config file path (if provided)
    temp_args, _ = parser.parse_known_args()
    if temp_args.config:
        if os.path.exists(temp_args.config):
            import yaml
            with open(temp_args.config, 'r') as f:
                config_data = yaml.safe_load(f)
                
            # Flatten the YAML structure to match argparse arguments
            flat_config = {}
            for section, params in config_data.items():
                if isinstance(params, dict):
                    for k, v in params.items():
                        flat_config[k.replace("-", "_")] = v
                else:
                    flat_config[section.replace("-", "_")] = params
                    
            parser.set_defaults(**flat_config)
            print(f"Loaded configuration from {temp_args.config}")
        else:
            print(f"Warning: Configuration file '{temp_args.config}' not found. Using defaults.")

    return parser.parse_args()


def get_backbone_kwargs(args):
    """Build backbone-specific kwargs from CLI args."""
    name = args.backbone.lower()
    kwargs = {"dropout": args.dropout}

    if name == "dnn":
        kwargs["hidden_sizes"] = args.hidden_sizes
    elif name in ("rnn", "gru", "lstm", "bilstm"):
        kwargs["hidden_size"] = args.hidden_size
        kwargs["num_layers"] = args.num_layers
        kwargs["seq_len"] = args.seq_len
    elif name == "mamba":
        kwargs["d_model"] = args.d_model
        kwargs["n_layers"] = args.n_layers
        kwargs["d_state"] = args.d_state
        kwargs["d_conv"] = args.d_conv
        kwargs["seq_len"] = args.seq_len

    return kwargs


# ============================================================
#  CSV LOGGING
# ============================================================
def init_csv_logging(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not os.path.exists(filepath):
        with open(filepath, 'w', newline='') as f:
            csv.writer(f).writerow(CSV_HEADER)
        print(f"Created new log file: {filepath}")
    else:
        print(f"Appending to existing log file: {filepath}")


def get_episode_count(filepath):
    if not os.path.exists(filepath):
        return 0
    try:
        with open(filepath, 'r') as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return 0


def log_episode(filepath, row):
    try:
        with open(filepath, 'a', newline='') as f:
            csv.writer(f).writerow(row)
    except Exception as e:
        print(f"Logging Error: {e}")


# ============================================================
#  TRAINING LOOP
# ============================================================
def train(args):
    set_seed(args.seed)
    backbone_name = args.backbone.lower()
    backbone_upper = {
        "dnn": "DNN", "rnn": "RNN", "gru": "GRU",
        "lstm": "LSTM", "bilstm": "biLSTM", "mamba": "Mamba",
    }[backbone_name]

    timestamp = dt.datetime.now().strftime('%d%m%Y_%H%M%S')
    exp_name = args.exp_name
    run_id = f"{exp_name}_seed{args.seed}_{timestamp}"
    
    output_dir = os.path.join(f"{backbone_upper}_PPG", "runs", run_id)
    model_dir = os.path.join(output_dir, "models")
    csv_path = os.path.join(output_dir, "training_log.csv")
    config_dump_path = os.path.join(output_dir, "run_config.yaml")
    best_save_path = os.path.join(model_dir, f"{backbone_upper}_PPG_best")

    init_csv_logging(csv_path)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save a copy of the hyperparameters used for this run
    import yaml
    with open(config_dump_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    # Environment
    map_configs = args.map if args.map else MAP_CONFIGS
    render = args.render.lower() == "true"

    env = SumoEnv(render=render, map_config=map_configs, test_mode=False)
    state_dim = env.observation_space.shape[0]   # 45
    action_dim = env.action_space.shape[0]       # 2

    # Backbone kwargs
    backbone_kwargs = get_backbone_kwargs(args)
    backbone_kwargs.pop("d_model", None)  # Prevent duplicate parameter error in PPGAgent

    print(f"\n{'='*60}")
    print(f"  PPG Agent — {backbone_upper} Backbone")
    print(f"  Algorithm: Phasic Policy Gradient (PPG)")
    print(f"  State Dim: {state_dim}, Action Dim: {action_dim}")
    print(f"  Backbone: {backbone_upper} | d_model={args.d_model}")
    print(f"  Backbone kwargs: {backbone_kwargs}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"{'='*60}\n")

    # Create agent
    agent = PPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        backbone_name=backbone_name,
        is_training_mode=True,
        entropy_coef=args.entropy_coef,
        vf_loss_coef=args.vf_loss_coef,
        batchsize=args.batchsize,
        PPO_epochs=args.ppo_epochs,
        gamma=args.gamma,
        lam=args.lam,
        learning_rate=args.lr,
        d_model=args.d_model,
        use_history=args.use_history,
        # PPG-specific
        lr_policy=args.lr_policy,
        lr_value=args.lr_value,
        n_aux=args.n_aux,
        k_aux=args.k_aux,
        beta_kl=args.beta_kl,
        d_targ=args.d_targ,
        clip_val=args.clip_val,
        clip_eps=args.clip_eps,
        **backbone_kwargs,
    )

    policy_params_count, value_params_count = agent.get_param_count()
    print(f"  Policy Model Params: {policy_params_count:,}")
    print(f"  Value Model Params:  {value_params_count:,}")
    print(f"  Total Params:        {policy_params_count + value_params_count:,}\n")

    if args.load_model:
        try:
            agent.load_weights(args.load_model)
            print(f"  [INFO] Loaded model weights from: {args.load_model}\n")
        except Exception as e:
            print(f"  [ERROR] Failed to load weights: {e}\n")

    print(f"  Logging to: {csv_path}")
    print(f"  Model dir:  {model_dir}")
    print(f"  Press Ctrl+C to stop and save.\n")

    # Defaults or overridden n_update step interval
    n_update_val = args.n_update if args.n_update is not None else agent.policy_memory.clear_memory() or 2048
    # Clear memory side-effect clean-up:
    agent.policy_memory.clear_memory()

    # Counters
    global_ep_cnt = get_episode_count(csv_path)
    best_reward = -float('inf')  # Tracks best moving average reward
    reward_window = []           # Rolling window for moving average
    window_size = 50             # Window size (e.g. 50 episodes)
    t_updates = 0
    start_time = time.time()
    max_action = 1.0

    try:
        for i_episode in range(1, args.n_episode + 1):
            result = env.reset()
            state = result[0] if isinstance(result, tuple) else result
            agent.reset_history(state)
            done = False

            ep_reward = 0.0
            ep_energy = 0.0
            ep_speed_sum = 0.0
            ep_safety_sum = 0.0
            ep_jerk_sum = 0.0
            ep_steps = 0

            for _ in range(5000):
                action, log_prob, value, value_val = agent.act(state)
                action_clipped = np.clip(action, -1.0, 1.0) * max_action

                result = env.step(action_clipped)
                if len(result) == 5:
                    next_state, reward, terminated, truncated, info = result
                    done = terminated or truncated
                else:
                    next_state, reward, done, info = result

                ep_steps += 1
                t_updates += 1
                ep_reward += reward
                ep_energy += info.get("real_energy", 0.0)
                ep_speed_sum += info.get("real_speed", 0.0)
                ep_safety_sum += info.get("safety", 0.0)
                ep_jerk_sum += info.get("comfort", info.get("wiggle", 0.0))

                agent.save_eps(
                    state.tolist(), action.tolist(), reward, float(done),
                    next_state.tolist(), log_prob, value, value_val
                )

                state = next_state

                # PPG Update
                if t_updates == n_update_val:
                    update_results = agent.update()
                    t_updates = 0
                    if update_results and update_results.get("aux_executed"):
                        print(f"  [AUX PHASE] Executed auxiliary phase update. Loss: {update_results['aux_loss']:.4f}")

                if done:
                    break

            # Logging
            global_ep_cnt += 1
            avg_speed = ep_speed_sum / max(1, ep_steps)
            avg_safety = ep_safety_sum / max(1, ep_steps)
            avg_jerk = ep_jerk_sum / max(1, ep_steps)
            success = info.get("is_success", 0)
            reason = info.get("reason", "unknown")
            route_str = info.get("route", "") if reason in ["stuck_too_long", "timeout", "teleport"] else ""
            override_rate = info.get("override_rate", 0.0)
            avg_physical_jerk = info.get("avg_jerk", 0.0)

            row = [
                global_ep_cnt, ep_steps,
                f"{ep_reward:.2f}", f"{avg_speed:.2f}", f"{ep_energy:.2f}",
                f"{avg_jerk:.4f}", f"{avg_safety:.4f}",
                success, reason, route_str,
                f"{override_rate:.4f}", f"{avg_physical_jerk:.4f}"
            ]
            log_episode(csv_path, row)

            print(f"Ep {global_ep_cnt:>5d} | "
                  f"Reward: {ep_reward:>8.2f} | "
                  f"Speed: {avg_speed:>5.2f} | "
                  f"Energy: {ep_energy:>8.1f} | "
                  f"Comfort: {avg_jerk:>6.4f} | "
                  f"PhysJerk: {avg_physical_jerk:>6.4f} | "
                  f"Override: {override_rate:>5.2%} | "
                  f"Steps: {ep_steps:>4d} | "
                  f"{'✓' if success else '✗'} {reason}")

            # Save best based on moving average reward (requires at least 10 episodes to prevent early noise)
            reward_window.append(ep_reward)
            if len(reward_window) > window_size:
                reward_window.pop(0)
            moving_reward = np.mean(reward_window)

            if len(reward_window) >= min(10, window_size):
                if moving_reward > best_reward:
                    best_reward = moving_reward
                    agent.save_weights(best_save_path)
                    print(f"  [BEST] New best moving average reward {best_reward:.2f} (over last {len(reward_window)} eps)! Model saved.")

            # Checkpoint
            if i_episode % args.n_saved == 0:
                cp_path = os.path.join(model_dir, f"{backbone_upper}_PPG_ep{global_ep_cnt}")
                agent.save_weights(cp_path)
                print(f"  [CHECKPOINT] Episode {global_ep_cnt}")

    except KeyboardInterrupt:
        print("\n\n!!! User Interrupted Training (Ctrl+C) !!!")
        agent.save_weights(os.path.join(model_dir, f"{backbone_upper}_PPG_emergency"))

    except Exception as e:
        print(f"\n!!! Critical Error: {e} !!!")
        agent.save_weights(os.path.join(model_dir, f"{backbone_upper}_PPG_crash"))
        raise e

    finally:
        elapsed = time.time() - start_time
        print(f"\nTotal Training Time: {dt.timedelta(seconds=int(elapsed))}")
        agent.save_weights(os.path.join(model_dir, f"{backbone_upper}_PPG_final"))
        print(f"Final model saved to {model_dir}")
        env.close()
        print("Training Closed.")


if __name__ == "__main__":
    args = parse_args()
    train(args)
