"""
Unified PPG Test Script — Multi-Backbone
==========================================
Test PPG models with any backbone on ALL routes:

    python test_ppg.py --backbone lstm -m LSTM_PPG/models/.../LSTM_PPG_best
    python test_ppg.py --backbone mamba -m Mamba_PPG/models/.../Mamba_PPG_best
    python test_ppg.py --backbone dnn -m DNN_PPG/models/.../DNN_PPG_best --no-gui

Based on test_ppo.py but adapted for PPGAgent.
"""

import os
import sys
import csv
import argparse
import random
import time
import xml.etree.ElementTree as ET
import torch
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ppg.core import PPGAgent
from simulation.env import SumoEnv
from backbones import BACKBONE_REGISTRY

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CSV_HEADER = [
    "route_idx", "flow_id", "route_edges", "edge_count", "route_length_m",
    "steps", "ep_reward",
    "avg_speed", "total_energy",
    "avg_safety", "avg_comfort",
    "success", "reason",
    "override_rate", "avg_jerk"
]


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTE PARSER
# ═══════════════════════════════════════════════════════════════════════════════
def parse_routes_from_xml(rou_xml_path: str) -> list:
    tree = ET.parse(rou_xml_path)
    root = tree.getroot()
    all_routes = []
    seen_edge_tuples = set()

    for flow in root.findall(".//flow"):
        flow_id = flow.get("id", "unknown")
        route_elem = flow.find("route")
        if route_elem is not None:
            edges_str = route_elem.get("edges", "")
            edges = edges_str.split()
            edge_tuple = tuple(edges)
            if edge_tuple not in seen_edge_tuples and len(edges) >= 2:
                seen_edge_tuples.add(edge_tuple)
                all_routes.append({"flow_id": flow_id, "edges": edges})

    for vehicle in root.findall(".//vehicle"):
        veh_id = vehicle.get("id", "unknown")
        route_elem = vehicle.find("route")
        if route_elem is not None:
            edges_str = route_elem.get("edges", "")
            edges = edges_str.split()
            edge_tuple = tuple(edges)
            if edge_tuple not in seen_edge_tuples and len(edges) >= 2:
                seen_edge_tuples.add(edge_tuple)
                all_routes.append({"flow_id": veh_id, "edges": edges})

    print(f"Parsed {len(all_routes)} unique routes from {rou_xml_path}")
    return all_routes


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM RESET
# ═══════════════════════════════════════════════════════════════════════════════
def reset_with_specific_route(env, route_edges: list):
    import traci
    import traci.constants as tc

    try:
        traci.close()
    except Exception:
        pass
    time.sleep(0.5)

    active_map = env.maps[0]
    if isinstance(active_map, list):
        active_map = active_map[0]

    env.step_count = 0
    env.veh_data = None
    env._route_edge_lengths = {}
    env._turn_info_cache = (0.0, 1.0, 0.0)
    env._prev_tls_was_red = False
    env._context_subscribed = False
    env._current_junction_tls_id = None
    env._last_recognized_vehicles = set()
    env._prev_was_in_junction = False

    SumoBinary = "sumo-gui" if env.render_mode else "sumo"
    SumoCMD = [SumoBinary, "-c", active_map,
               "--start", "--quit-on-end",
               "--device.emissions.probability", "1.0",
               "--scale", str(env.TRAFFIC_SCALE),
               "--delay", str(env.delay),
               "--no-step-log", "true",
               "--time-to-teleport", "-1",
               "--collision.action", "remove",
               "--collision.check-junctions", "true",
               "--no-warnings", "true"]

    traci.start(SumoCMD)
    env._success = False

    if hasattr(env, "prev_action"): del env.prev_action
    if hasattr(env, "prev_dist"): del env.prev_dist

    for _ in range(1000):
        traci.simulationStep()

    try:
        existing_types = traci.vehicletype.getIDList()
        source_type = "DEFAULT_VEHTYPE"
        if source_type not in existing_types and len(existing_types) > 0:
            source_type = existing_types[0]

        traci.vehicletype.copy(source_type, env.VTYPE_ID)
        traci.vehicletype.setVehicleClass(env.VTYPE_ID, "passenger")
        traci.vehicletype.setColor(env.VTYPE_ID, (0, 255, 0))
        traci.vehicletype.setParameter(env.VTYPE_ID, "mass", "2911")
        traci.vehicletype.setLength(env.VTYPE_ID, "5.1181")
        traci.vehicletype.setEmissionClass(env.VTYPE_ID, "MMPEVEM")

        traci.vehicletype.setParameter(env.VTYPE_ID, "has.battery.device", "true")
        traci.vehicletype.setParameter(env.VTYPE_ID, "device.battery.capacity", "123000.00")
        traci.vehicletype.setParameter(env.VTYPE_ID, "device.battery.chargeLevel", "123000.00")
        traci.vehicletype.setParameter(env.VTYPE_ID, "device.battery.rechargeEfficiency", "0.8")
        traci.vehicletype.setParameter(env.VTYPE_ID, "device.battery.maxRegenerationAcceleration", "2.0")

        for v_type in existing_types:
            traci.vehicletype.setParameter(v_type, "sigma", str(env.imperfection))
            traci.vehicletype.setParameter(v_type, "impatience", str(env.impatience))
    except Exception:
        pass

    spawned = False
    route_length = 0.0
    for edge in route_edges:
        try:
            route_length += traci.lane.getLength(f"{edge}_0")
        except Exception:
            pass

    try:
        route_id = f"test_route_{random.randint(0, 999999)}"
        traci.route.add(route_id, route_edges)
        env.current_route_edges = route_edges
        traci.vehicle.add(env.VEH_ID, route_id, departPos="free", typeID=env.VTYPE_ID)

        for _ in range(50):
            traci.simulationStep()
            if env.VEH_ID in traci.vehicle.getIDList():
                spawned = True
                break

        if spawned:
            traci.vehicle.setSpeedMode(env.VEH_ID, 0)
            traci.vehicle.setLaneChangeMode(env.VEH_ID, 768)  # 256+512: cho phép đổi làn qua TraCI
            env.last_known_dist = route_length
            env.prev_dist = route_length
    except Exception as e:
        print(f"  ⚠ Error spawning vehicle: {e}")

    if not spawned:
        print(f"  ⚠ Failed to spawn, retrying...")
        return reset_with_specific_route(env, route_edges)

    if env.render_mode and spawned:
        if env.VEH_ID in traci.vehicle.getIDList():
            try:
                traci.gui.setSchema("View #0", "real world")
            except Exception:
                pass
            traci.gui.trackVehicle("View #0", env.VEH_ID)
            traci.gui.setZoom("View #0", 1001)

    env.stuck_time = 0
    env._update_cache()

    obs = env._get_obs()
    return obs, {"route_length": route_length}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def init_csv(filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        csv.writer(f).writerow(CSV_HEADER)


def append_csv(filepath: str, row: list):
    with open(filepath, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified PPG test script for all backbones on ALL routes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_ppg.py --backbone lstm -m LSTM_PPG/models/.../LSTM_PPG_best
  python test_ppg.py --backbone mamba -m Mamba_PPG/models/.../Mamba_PPG_best --no-gui
        """
    )

    parser.add_argument("--config", type=str, default="", help="Path to YAML config file")
    parser.add_argument("--backbone", choices=list(BACKBONE_REGISTRY.keys()), help="Backbone architecture")
    parser.add_argument("-m", "--model", required=True, help="Path prefix to saved model (without _policy.tar)")
    parser.add_argument("--no-gui", action="store_true", help="Disable SUMO GUI")
    parser.add_argument("--delay", type=int, default=0, help="SUMO GUI delay (ms)")
    parser.add_argument("-map", "--map", type=str, default="maps/grid_3_3_intelligent_tls/run.sumocfg")
    parser.add_argument("-r", "--route-file", type=str, default="maps/grid_3_3_intelligent_tls/grid_3_3_intelligent_tls.rou.xml")

    # Backbone hyperparams (must match training)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--use-history", type=str, default="auto", choices=["auto", "true", "false"], help="Whether to use rolling temporal history (default: auto)")

    # First parse to get the model path and config path
    temp_args, _ = parser.parse_known_args()
    
    config_file = temp_args.config
    
    # Try to auto-detect run_config.yaml in the model's run folder
    if not config_file and temp_args.model:
        model_dir = os.path.dirname(temp_args.model)
        # Check in the model directory or one level up
        for base_path in [model_dir, os.path.dirname(model_dir)]:
            candidate = os.path.join(base_path, "run_config.yaml")
            if os.path.exists(candidate):
                config_file = candidate
                print(f"[!] Auto-detected run configuration: {config_file}")
                break

    if config_file:
        import yaml
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
            
        flat_config = {}
        for section, params in config_data.items():
            if isinstance(params, dict):
                for k, v in params.items():
                    flat_config[k.replace("-", "_")] = v
            else:
                flat_config[section.replace("-", "_")] = params
                
        parser.set_defaults(**flat_config)
        print(f"Loaded configuration from {config_file}")

    args = parser.parse_args()
    if not args.backbone:
        parser.error("the following arguments are required: --backbone (or it must be specified in the loaded config)")
        
    return args


def get_backbone_kwargs(args):
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


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TEST LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def run_test(args):
    backbone_name = args.backbone.lower()
    backbone_upper = {
        "dnn": "DNN", "rnn": "RNN", "gru": "GRU",
        "lstm": "LSTM", "bilstm": "biLSTM", "mamba": "Mamba",
    }[backbone_name]

    render = not args.no_gui
    routes = parse_routes_from_xml(args.route_file)
    n_routes = len(routes)

    if n_routes == 0:
        print("ERROR: No routes found!")
        return

    model_name = os.path.basename(args.model)
    # Put eval results in the same run directory as the model
    model_dir = os.path.dirname(args.model)
    run_dir = os.path.dirname(model_dir) if "models" in model_dir else model_dir
    report_dir = os.path.join(run_dir, "eval")
    
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    csv_path = os.path.join(report_dir, f"test_results_{model_name}_{timestamp}.csv")
    init_csv(csv_path)

    # Build agent
    backbone_kwargs = get_backbone_kwargs(args)
    backbone_kwargs.pop("d_model", None)  # Prevent duplicate parameter error in PPGAgent

    env = SumoEnv(
        render=render,
        map_config=args.map,
        test_mode=False,
        delay=args.delay,
    )
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        backbone_name=backbone_name,
        is_training_mode=False,
        use_history=args.use_history,
        d_model=args.d_model,
        **backbone_kwargs,
    )

    # Load weights
    prefix = args.model
    if prefix.endswith(".tar"):
        prefix = prefix.replace("_policy.tar", "").replace("_value.tar", "")

    print(f"Loading {backbone_upper}_PPG model from: {prefix}")
    agent.load_weights(prefix)
    print("Loaded weights successfully.")

    env.MAX_EPISODE_STEPS = 2000

    # Tracking
    all_rewards = []
    all_speeds = []
    all_energies = []
    all_comforts = []
    all_override_rates = []
    all_physical_jerks = []
    success_count = 0

    print(f"\n{'='*70}")
    print(f"  Model        : {model_name}")
    print(f"  Backbone     : {backbone_upper}")
    print(f"  Algorithm    : PPG")
    print(f"  Total routes : {n_routes}")
    print(f"  GUI          : {'ON' if render else 'OFF'}")
    print(f"{'='*70}\n")

    for route_idx, route_info in enumerate(routes, 1):
        flow_id = route_info["flow_id"]
        route_edges = route_info["edges"]

        print(f"\n--- Route {route_idx}/{n_routes} (flow={flow_id}) ---")

        obs, reset_info = reset_with_specific_route(env, route_edges)
        agent.reset_history(obs)
        route_length = reset_info.get("route_length", 0.0)

        ep_reward = 0.0
        ep_energy = 0.0
        ep_speed_sum = 0.0
        ep_safety_sum = 0.0
        ep_jerk_sum = 0.0
        prev_accel = env.veh_data["accel"] if (env.veh_data and "accel" in env.veh_data) else 0.0
        ep_steps = 0
        done = False

        while not done:
            action, _, _, _ = agent.act(obs)

            result = env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result

            ep_reward += reward
            ep_energy += info.get("real_energy", 0.0)
            ep_speed_sum += info.get("real_speed", 0.0)
            ep_safety_sum += info.get("safety", 0.0)
            
            # Jerk calculation (Comfort Index)
            curr_accel = env.veh_data["accel"] if (env.veh_data and "accel" in env.veh_data) else 0.0
            jerk = (curr_accel - prev_accel) / 0.5
            ep_jerk_sum += abs(jerk)
            prev_accel = curr_accel
            
            ep_steps += 1

        avg_speed = ep_speed_sum / max(1, ep_steps)
        avg_safety = ep_safety_sum / max(1, ep_steps)
        avg_comfort = ep_jerk_sum / max(1, ep_steps)
        success = info.get("is_success", 0)
        reason = info.get("reason", "unknown")
        override_rate = info.get("override_rate", 0.0)
        avg_physical_jerk = info.get("avg_jerk", 0.0)

        if success:
            success_count += 1

        all_rewards.append(ep_reward)
        all_speeds.append(avg_speed)
        all_energies.append(ep_energy)
        all_comforts.append(avg_comfort)
        all_override_rates.append(override_rate)
        all_physical_jerks.append(avg_physical_jerk)

        row = [
            route_idx, flow_id, " -> ".join(route_edges), len(route_edges),
            f"{route_length:.1f}", ep_steps, f"{ep_reward:.2f}",
            f"{avg_speed:.2f}", f"{ep_energy:.2f}", f"{avg_safety:.4f}", f"{avg_comfort:.4f}",
            success, reason,
            f"{override_rate:.4f}", f"{avg_physical_jerk:.4f}"
        ]
        append_csv(csv_path, row)

        status = "✓" if success else "✗"
        print(f"  [{route_idx:>3}/{n_routes}] {status}  "
              f"steps={ep_steps:>4}  reward={ep_reward:>8.2f}  "
              f"speed={avg_speed:>5.2f} m/s  energy={ep_energy:>8.2f}  "
              f"comfort={avg_comfort:>8.4f} m/s³  "
              f"override={override_rate:>5.2%}  "
              f"phys_jerk={avg_physical_jerk:>8.4f} m/s³  "
              f"reason={reason}")

    env.close()

    print(f"\n{'='*70}")
    print(f"  SUMMARY  ({n_routes} routes)")
    print(f"  Backbone     : {backbone_upper}")
    print(f"{'='*70}")
    print(f"  Success rate : {success_count}/{n_routes}  ({100*success_count/n_routes:.1f} %)")
    print(f"  Avg reward   : {np.mean(all_rewards):.2f}  ± {np.std(all_rewards):.2f}")
    print(f"  Avg speed    : {np.mean(all_speeds):.2f} m/s")
    print(f"  Avg energy   : {np.mean(all_energies):.2f}")
    print(f"  Avg comfort  : {np.mean(all_comforts):.4f} m/s³")
    print(f"  Avg override : {np.mean(all_override_rates):.2%}")
    print(f"  Avg phys jerk: {np.mean(all_physical_jerks):.4f} m/s³")
    print(f"  Results CSV  : {csv_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    args = parse_args()
    run_test(args)
