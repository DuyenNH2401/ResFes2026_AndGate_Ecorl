"""
PPG Memory & Auxiliary Buffer
===============================
Replay buffers for Phasic Policy Gradient (PPG) training:
- PPGMemory: On-policy memory Dataset for storing transitions during the Policy Phase.
- AuxiliaryBuffer: Stores states and returns over multiple policy updates to use in the Auxiliary Phase.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.distributions.normal import Normal

class PPGMemory(Dataset):
    """
    On-policy memory dataset for PPG.
    Stores states, actions, rewards, next_states, values (from policy network's
    auxiliary critic head), value_vals (from separate value network), log_probs,
    advantages, and returns.
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.log_probs = []
        self.values = []
        self.value_vals = []
        self.advantages = []
        self.returns = []
        self.old_values = []
        self.old_value_vals = []
        # Lagrangian cost (raw, ≥0). Mặc định rỗng; chỉ dùng khi lagrangian bật.
        self.cost_safety = []
        self.cost_comfort = []
        self.cost_redlight = []

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        return (
            np.array(self.states[idx], dtype=np.float32),
            np.array(self.actions[idx], dtype=np.float32),
            np.array([self.rewards[idx]], dtype=np.float32),
            np.array([self.dones[idx]], dtype=np.float32),
            np.array(self.next_states[idx], dtype=np.float32),
            np.array([self.log_probs[idx]], dtype=np.float32),
            np.array([self.advantages[idx]], dtype=np.float32),
            np.array([self.returns[idx]], dtype=np.float32),
            np.array([self.old_values[idx]], dtype=np.float32),
            np.array([self.old_value_vals[idx]], dtype=np.float32),
        )

    def save_eps(self, state, action, reward, done, next_state, log_prob, value, value_val,
                 cost_safety=0.0, cost_comfort=0.0, cost_redlight=0.0):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.value_vals.append(value_val)
        self.cost_safety.append(cost_safety)
        self.cost_comfort.append(cost_comfort)
        self.cost_redlight.append(cost_redlight)

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]
        del self.log_probs[:]
        del self.values[:]
        del self.value_vals[:]
        del self.advantages[:]
        del self.returns[:]
        del self.old_values[:]
        del self.old_value_vals[:]
        del self.cost_safety[:]
        del self.cost_comfort[:]
        del self.cost_redlight[:]


class AuxiliaryBuffer:
    """
    Auxiliary Buffer for storing states and GAE returns for the Periodical Auxiliary Phase.
    Automatically computes value targets and target policy parameters on transition to Phase 2.
    """

    def __init__(self):
        self.states = []
        self.returns = []
        self.target_values = []
        self.target_means = []
        self.target_stds = []

    def store_states_and_returns(self, states_array, returns_array):
        """
        Store states and returns from a completed Policy Phase.
        Args:
            states_array: numpy array of states (N, state_dim)
            returns_array: numpy array of GAE returns (N,)
        """
        self.states.append(states_array)
        self.returns.append(returns_array)

    def prepare_targets(self, policy_net, value_net, device):
        """
        Computes value targets and target policy parameter distributions to constrain policy drift.
        Invoked prior to starting Phase 2 updates.
        """
        if len(self.states) == 0:
            return

        # Concatenate all stored runs
        all_states = np.concatenate(self.states, axis=0)  # (Total_N, state_dim)
        all_returns = np.concatenate(self.returns, axis=0)  # (Total_N,)
        num_samples = all_states.shape[0]

        batch_size = 256
        target_values_list = []
        target_means_list = []
        target_stds_list = []

        policy_net.eval()
        value_net.eval()

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                states_batch = torch.tensor(
                    all_states[i : i + batch_size], device=device, dtype=torch.float32
                )

                # Target value estimate from separate value network V_phi
                v_target = value_net.get_value(states_batch)  # (B,)
                target_values_list.append(v_target.cpu().numpy())

                # Target policy mean & std for KL divergence constraint
                dist = policy_net.get_distribution(states_batch)
                target_means_list.append(dist.mean.cpu().numpy())
                target_stds_list.append(dist.stddev.cpu().numpy())

        self.states = all_states
        self.returns = all_returns
        self.target_values = np.concatenate(target_values_list, axis=0)
        self.target_means = np.concatenate(target_means_list, axis=0)
        self.target_stds = np.concatenate(target_stds_list, axis=0)

    def clear(self):
        self.states = []
        self.returns = []
        self.target_values = []
        self.target_means = []
        self.target_stds = []
