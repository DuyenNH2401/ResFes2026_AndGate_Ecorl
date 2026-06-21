"""
PPG Core Agent & Models
========================
Backbone-agnostic Phasic Policy Gradient (PPG) agent implementing separate
Policy and Value networks, Phase 1 (Policy Phase) updates, and Phase 2 (Auxiliary Phase) updates.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from backbones import create_backbone
from .memory import PPGMemory, AuxiliaryBuffer
from .distributions import Continous
from . import ppg_config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PolicyModel(nn.Module):
    """
    Policy Model - SHARED backbone with Actor + Auxiliary Critic heads.
    
    Args:
        state_dim: Dimension of the input state vector.
        action_dim: Dimension of the continuous action space.
        backbone_name: Name of the backbone from registry.
        d_model: Feature dimension from the backbone.
        **backbone_kwargs: Backbone-specific hyperparameters.
    """

    def __init__(self, state_dim, action_dim, backbone_name, d_model=128, **backbone_kwargs):
        super().__init__()

        self.backbone = create_backbone(
            backbone_name, input_dim=state_dim, output_dim=d_model, d_model=d_model, **backbone_kwargs
        )

        # Actor head: maps backbone features to action_dim (Tanh for continuous [-1, 1])
        self.actor_layer = nn.Sequential(
            nn.Linear(d_model, action_dim),
            nn.Tanh()
        )

        # Learnable log-std parameter for continuous action distribution
        self.action_log_std = nn.Parameter(torch.ones(action_dim) * -0.51)

        # Auxiliary Critic head (value estimate from Policy Network)
        self.critic_layer = nn.Linear(d_model, 1)

    def forward(self, states):
        """
        Args:
            states: (B, state_dim) or (state_dim,) input states.
        Returns:
            action_mean: (B, action_dim)
            action_std: (B, action_dim)
            value: (B,)
        """
        features = self.backbone(states)
        action_mean = self.actor_layer(features)
        action_std = torch.exp(self.action_log_std).expand_as(action_mean)
        value = self.critic_layer(features).squeeze(-1)
        return action_mean, action_std, value

    def get_distribution(self, states):
        """Returns the normal distribution of the actor."""
        features = self.backbone(states)
        action_mean = self.actor_layer(features)
        action_std = torch.exp(self.action_log_std).expand_as(action_mean)
        return Normal(action_mean, action_std)

    def get_action_and_value(self, states, action=None):
        """Computes action, log-prob, entropy, and auxiliary value prediction."""
        features = self.backbone(states)
        action_mean = self.actor_layer(features)
        action_std = torch.exp(self.action_log_std).expand_as(action_mean)
        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic_layer(features).squeeze(-1)

        return action, log_prob, entropy, value

    def get_value(self, states):
        """Returns only the auxiliary value prediction."""
        features = self.backbone(states)
        value = self.critic_layer(features).squeeze(-1)
        return value


class ValueModel(nn.Module):
    """
    Separate Value Model with its own independent backbone.
    
    Args:
        state_dim: Dimension of the input state vector.
        backbone_name: Name of the backbone from registry.
        d_model: Feature dimension from the backbone.
        **backbone_kwargs: Backbone-specific hyperparameters.
    """

    def __init__(self, state_dim, backbone_name, d_model=128, **backbone_kwargs):
        super().__init__()

        self.backbone = create_backbone(
            backbone_name, input_dim=state_dim, output_dim=d_model, d_model=d_model, **backbone_kwargs
        )
        self.critic_layer = nn.Linear(d_model, 1)

    def forward(self, states):
        """
        Args:
            states: (B, state_dim) or (state_dim,) input states.
        Returns:
            value: (B,)
        """
        features = self.backbone(states)
        value = self.critic_layer(features).squeeze(-1)
        return value

    def get_value(self, states):
        """Returns the value prediction."""
        features = self.backbone(states)
        value = self.critic_layer(features).squeeze(-1)
        return value


class PPGAgent:
    """
    Backbone-agnostic Phasic Policy Gradient (PPG) agent.
    
    Phase 1 (Policy Phase): Standard PPO updates with separate Policy and Value networks.
    Phase 2 (Auxiliary Phase): Periodic updates (every N_aux steps) utilizing target value
    distillation and a KL constraint on policy divergence.
    """

    def __init__(self, state_dim, action_dim, backbone_name="mamba",
                 is_training_mode=True, entropy_coef=None, vf_loss_coef=None,
                 batchsize=None, PPO_epochs=None, gamma=None, lam=None,
                 learning_rate=None, d_model=128, use_history=None,
                 # PPG-specific hyperparameters
                 lr_policy=None, lr_value=None,
                 beta_kl=None, d_targ=None,
                 n_aux=None, k_aux=None, clip_val=None, clip_eps=None,
                 # Lagrangian adaptive weighting
                 lagrangian_enabled=None, lambda_safety_init=None,
                 lambda_comfort_init=None, lambda_redlight_init=None,
                 lambda_lr=None, cost_limit_safety=None, cost_limit_comfort=None,
                 cost_limit_redlight=None, lambda_max=None,
                 **backbone_kwargs):

        self.backbone_name = backbone_name
        self.is_training_mode = is_training_mode
        self.action_dim = action_dim
        self.d_model = d_model

        # Hyperparameter defaults from ppg_config.py
        self.entropy_coef = entropy_coef if entropy_coef is not None else ppg_config.ENTROPY_COEF
        self.vf_loss_coef = vf_loss_coef if vf_loss_coef is not None else ppg_config.VF_LOSS_COEF
        self.batchsize = batchsize if batchsize is not None else ppg_config.BATCHSIZE
        self.PPO_epochs = PPO_epochs if PPO_epochs is not None else ppg_config.PPO_EPOCHS
        self.gamma = gamma if gamma is not None else ppg_config.GAMMA
        self.lam = lam if lam is not None else ppg_config.LAM

        # Learning rates
        base_lr = learning_rate if learning_rate is not None else ppg_config.LR_POLICY
        self.lr_policy = lr_policy if lr_policy is not None else base_lr
        self.lr_value = lr_value if lr_value is not None else ppg_config.LR_VALUE

        # PPG-specific configurations
        self.beta_kl = beta_kl if beta_kl is not None else ppg_config.BETA_KL
        self.d_targ = d_targ if d_targ is not None else ppg_config.D_TARG
        self.N_aux = n_aux if n_aux is not None else ppg_config.N_AUX
        self.K_aux = k_aux if k_aux is not None else ppg_config.K_AUX
        self.clip_val = clip_val if clip_val is not None else ppg_config.CLIP_VAL
        self.clip_eps = clip_eps if clip_eps is not None else getattr(ppg_config, 'CLIP_EPS', 0.2)

        # Lagrangian adaptive weighting (an toàn + êm ái + đèn đỏ là ràng buộc)
        self.lagrangian_enabled = lagrangian_enabled if lagrangian_enabled is not None else ppg_config.LAGRANGIAN_ENABLED
        self.lambda_safety = lambda_safety_init if lambda_safety_init is not None else ppg_config.LAMBDA_SAFETY_INIT
        self.lambda_comfort = lambda_comfort_init if lambda_comfort_init is not None else ppg_config.LAMBDA_COMFORT_INIT
        self.lambda_redlight = lambda_redlight_init if lambda_redlight_init is not None else ppg_config.LAMBDA_REDLIGHT_INIT
        self.lambda_lr = lambda_lr if lambda_lr is not None else ppg_config.LAMBDA_LR
        self.cost_limit_safety = cost_limit_safety if cost_limit_safety is not None else ppg_config.COST_LIMIT_SAFETY
        self.cost_limit_comfort = cost_limit_comfort if cost_limit_comfort is not None else ppg_config.COST_LIMIT_COMFORT
        self.cost_limit_redlight = cost_limit_redlight if cost_limit_redlight is not None else ppg_config.COST_LIMIT_REDLIGHT
        self.lambda_max = lambda_max if lambda_max is not None else ppg_config.LAMBDA_MAX

        # Setup history sequential wrapping matching PPO Agent
        self.history = []
        if use_history is None or use_history == "auto":
            # Mamba consumes flat state vectors; no rolling history in auto mode.
            self.is_sequential = False
        else:
            if isinstance(use_history, str):
                self.is_sequential = use_history.lower() == "true"
            else:
                self.is_sequential = bool(use_history)

        if self.is_sequential:
            self.seq_len = backbone_kwargs.get("seq_len", 5)
            self.agent_state_dim = state_dim * self.seq_len
        else:
            self.seq_len = 1
            self.agent_state_dim = state_dim

        # Networks
        self.policy = PolicyModel(
            self.agent_state_dim, action_dim, backbone_name, d_model, **backbone_kwargs
        ).to(device)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.lr_policy)

        self.value = ValueModel(
            self.agent_state_dim, backbone_name, d_model, **backbone_kwargs
        ).to(device)
        self.value_optimizer = Adam(self.value.parameters(), lr=self.lr_value)

        # PPG Memory buffers
        self.policy_memory = PPGMemory()
        self.aux_buffer = AuxiliaryBuffer()
        self.t_aux = 0  # Policy phase update counter

        self.distributions = Continous()

        if is_training_mode:
            self.policy.train()
            self.value.train()
        else:
            self.policy.eval()
            self.value.eval()

    def reset_history(self, state=None):
        """Reset the rolling state history (sequential backbones)."""
        if self.is_sequential:
            if state is not None:
                self.history = [np.array(state, dtype=np.float32).copy() for _ in range(self.seq_len)]
            else:
                self.history = []

    def act(self, state):
        """
        Select actions from the policy network and retrieve value predictions.
        
        Args:
            state: NumPy array of shape (state_dim,)
        Returns:
            action: NumPy array of shape (action_dim,)
            log_prob: float
            v_policy: float (auxiliary value estimate)
            v_value: float (separate value estimate)
        """
        if self.is_sequential:
            if not self.history:
                self.history = [np.array(state, dtype=np.float32).copy() for _ in range(self.seq_len)]
            else:
                self.history.append(np.array(state, dtype=np.float32).copy())
                self.history.pop(0)
            stacked_state = np.concatenate(self.history, axis=0)
            state_t = torch.FloatTensor(stacked_state).unsqueeze(0).to(device).detach()
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device).detach()

        action_mean, action_std, v_policy = self.policy(state_t)
        v_value = self.value(state_t)

        if self.is_training_mode:
            action = self.distributions.sample(action_mean, action_std)
            log_prob = self.distributions.logprob(action_mean, action_std, action)
        else:
            action = action_mean
            log_prob = self.distributions.logprob(action_mean, action_std, action)

        return (
            action.squeeze(0).detach().cpu().numpy(),
            log_prob.squeeze(0).detach().cpu().item(),
            v_policy.squeeze(0).detach().cpu().item(),
            v_value.squeeze(0).detach().cpu().item(),
        )

    def save_eps(self, state, action, reward, done, next_state, log_prob, value, value_val,
                 cost_safety=0.0, cost_comfort=0.0, cost_redlight=0.0):
        """Saves a transition step to rollout memory."""
        if self.is_sequential:
            stacked_state = np.concatenate(self.history, axis=0).tolist()

            next_history = self.history[1:] + [np.array(next_state, dtype=np.float32).copy()]
            stacked_next_state = np.concatenate(next_history, axis=0).tolist()

            self.policy_memory.save_eps(
                stacked_state, action, reward, done, stacked_next_state, log_prob, value, value_val,
                cost_safety=cost_safety, cost_comfort=cost_comfort, cost_redlight=cost_redlight
            )
        else:
            self.policy_memory.save_eps(
                state, action, reward, done, next_state, log_prob, value, value_val,
                cost_safety=cost_safety, cost_comfort=cost_comfort, cost_redlight=cost_redlight
            )

    def update_lambdas(self, mean_cost_safety, mean_cost_comfort, mean_cost_redlight):
        """Dual gradient ascent cho hệ số Lagrangian λ.

        λ ← clamp(λ + lr · (mean_cost − limit), 0, λ_max).
        Nhận mean cost của rollout VỪA THU (đã tính trong update() trước khi
        clear_memory). λ mới sẽ áp cho rollout KẾ TIẾP — đúng chuẩn PPO-Lagrangian.
        """
        self.lambda_safety = float(np.clip(
            self.lambda_safety + self.lambda_lr * (mean_cost_safety - self.cost_limit_safety),
            0.0, self.lambda_max))
        self.lambda_comfort = float(np.clip(
            self.lambda_comfort + self.lambda_lr * (mean_cost_comfort - self.cost_limit_comfort),
            0.0, self.lambda_max))
        self.lambda_redlight = float(np.clip(
            self.lambda_redlight + self.lambda_lr * (mean_cost_redlight - self.cost_limit_redlight),
            0.0, self.lambda_max))

        return {
            "lambda_safety": self.lambda_safety,
            "lambda_comfort": self.lambda_comfort,
            "lambda_redlight": self.lambda_redlight,
            "mean_cost_safety": float(mean_cost_safety),
            "mean_cost_comfort": float(mean_cost_comfort),
            "mean_cost_redlight": float(mean_cost_redlight),
        }

    def compute_gae(self, last_value_val):
        """
        Generalized Advantage Estimation (GAE).
        Computes advantages and return targets based on the separate Value Network estimates.
        """
        rewards = np.array(self.policy_memory.rewards, dtype=np.float32)

        # Lagrangian: reward hiệu dụng r_eff = r − λ_s·c_s − λ_c·c_c − λ_r·c_r.
        # Dùng λ HIỆN TẠI = λ của rollout này (cập nhật λ diễn ra SAU update()).
        if getattr(self, "lagrangian_enabled", False):
            cs = np.array(self.policy_memory.cost_safety, dtype=np.float32)
            cc = np.array(self.policy_memory.cost_comfort, dtype=np.float32)
            cr = np.array(self.policy_memory.cost_redlight, dtype=np.float32)
            if cs.size == rewards.size and cc.size == rewards.size and cr.size == rewards.size:
                rewards = (rewards
                           - self.lambda_safety * cs
                           - self.lambda_comfort * cc
                           - self.lambda_redlight * cr)

        dones = np.array(self.policy_memory.dones, dtype=np.float32)
        value_vals = np.array(self.policy_memory.value_vals, dtype=np.float32)
        N = len(rewards)

        advantages = np.zeros(N, dtype=np.float32)
        returns = np.zeros(N, dtype=np.float32)

        last_adv = 0
        for t in reversed(range(N)):
            next_val = last_value_val if t == N - 1 else value_vals[t + 1]
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * non_terminal - value_vals[t]
            advantages[t] = last_adv = delta + self.gamma * self.lam * non_terminal * last_adv
            returns[t] = advantages[t] + value_vals[t]

        # Normalized advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(self):
        """Execute Phase 1 (Policy Phase) updates and check if Phase 2 is triggered."""
        if len(self.policy_memory) == 0:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "aux_executed": False,
                "aux_loss": 0.0
            }

        # Bootstrap separate value network estimate for terminal state
        last_next_state = self.policy_memory.next_states[-1]
        last_next_state_t = torch.FloatTensor(last_next_state).unsqueeze(0).to(device)
        with torch.no_grad():
            last_value_val = self.value.get_value(last_next_state_t).item()

        advantages, returns = self.compute_gae(last_value_val)

        # Store calculated advantages, returns, and values
        self.policy_memory.advantages = advantages.tolist()
        self.policy_memory.returns = returns.tolist()
        self.policy_memory.old_values = self.policy_memory.values
        self.policy_memory.old_value_vals = self.policy_memory.value_vals

        states_all_t = torch.FloatTensor(np.array(self.policy_memory.states)).to(device)

        # Cache pre-update old policy distribution distributions
        self.policy.eval()
        with torch.no_grad():
            old_dist = self.policy.get_distribution(states_all_t)
            old_means = old_dist.mean.clone()
            old_stds = old_dist.stddev.clone()

        num_samples = len(self.policy_memory)
        policy_losses = []
        value_losses = []
        entropy_losses = []

        # Phase 1: Policy Phase Updates
        for epoch in range(self.PPO_epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            for start in range(0, num_samples, self.batchsize):
                end = start + self.batchsize
                batch_idx = indices[start:end]

                b_states = states_all_t[batch_idx]
                b_actions = torch.FloatTensor(np.array(self.policy_memory.actions)[batch_idx]).to(device)
                b_old_log_probs = torch.FloatTensor(np.array(self.policy_memory.log_probs)[batch_idx]).to(device)
                b_old_values = torch.FloatTensor(np.array(self.policy_memory.values)[batch_idx]).to(device)
                b_old_value_vals = torch.FloatTensor(np.array(self.policy_memory.value_vals)[batch_idx]).to(device)
                b_advantages = torch.FloatTensor(advantages[batch_idx]).to(device)
                b_returns = torch.FloatTensor(returns[batch_idx]).to(device)
                b_old_means = old_means[batch_idx]
                b_old_stds = old_stds[batch_idx]

                self.policy.train()
                self.value.train()

                # Policy Network forward pass
                new_actions, new_log_probs, entropy, v_policy = self.policy.get_action_and_value(b_states, b_actions)
                
                # Value Network forward pass
                v_value = self.value.get_value(b_states)

                # Ratios
                r_t = torch.exp(new_log_probs - b_old_log_probs)

                # Standard PPO Clipped Surrogate Objective
                surr1 = r_t * b_advantages
                surr2 = torch.clamp(r_t, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages
                pg_target = torch.min(surr1, surr2)

                # Policy Network auxiliary critic head loss (value clipping)
                if self.clip_val is not None:
                    v_policy_clip = b_old_values + torch.clamp(v_policy - b_old_values, -self.clip_val, self.clip_val)
                    l_vf_policy = 0.5 * torch.max((v_policy - b_returns)**2, (v_policy_clip - b_returns)**2).mean()
                else:
                    l_vf_policy = 0.5 * ((v_policy - b_returns)**2).mean()

                # Separate Value Network critic loss
                if self.clip_val is not None:
                    v_value_clip = b_old_value_vals + torch.clamp(v_value - b_old_value_vals, -self.clip_val, self.clip_val)
                    l_vf_value = 0.5 * torch.max((v_value - b_returns)**2, (v_value_clip - b_returns)**2).mean()
                else:
                    l_vf_value = 0.5 * ((v_value - b_returns)**2).mean()

                # Total Policy Network loss (minimize)
                loss_policy = self.vf_loss_coef * l_vf_policy - self.entropy_coef * entropy.mean() - pg_target.mean()

                # Optimizer steps
                self.policy_optimizer.zero_grad()
                loss_policy.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad()
                l_vf_value.backward()
                nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=0.5)
                self.value_optimizer.step()

                policy_losses.append(-pg_target.mean().item())
                value_losses.append(l_vf_value.item())
                entropy_losses.append(entropy.mean().item())

        # Archive states and return targets into the auxiliary phase buffer
        self.aux_buffer.store_states_and_returns(np.array(self.policy_memory.states), returns)
        self.t_aux += 1

        # Periodic Phase 2 check
        aux_executed = False
        aux_loss_val = 0.0
        if self.t_aux >= self.N_aux:
            aux_loss_val = self.update_auxiliary_phase()
            aux_executed = True

        # Mean cost của rollout này — đọc TRƯỚC clear_memory để train loop
        # cập nhật λ cho rollout kế tiếp (chuẩn PPO-Lagrangian).
        mean_cost_safety = mean_cost_comfort = mean_cost_redlight = 0.0
        if getattr(self, "lagrangian_enabled", False):
            cs = self.policy_memory.cost_safety
            cc = self.policy_memory.cost_comfort
            cr = self.policy_memory.cost_redlight
            if len(cs) > 0:
                mean_cost_safety = float(np.mean(cs))
            if len(cc) > 0:
                mean_cost_comfort = float(np.mean(cc))
            if len(cr) > 0:
                mean_cost_redlight = float(np.mean(cr))

        self.policy_memory.clear_memory()

        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropy_losses),
            "aux_executed": aux_executed,
            "aux_loss": aux_loss_val,
            "mean_cost_safety": mean_cost_safety,
            "mean_cost_comfort": mean_cost_comfort,
            "mean_cost_redlight": mean_cost_redlight,
        }

    def update_auxiliary_phase(self):
        """
        Phase 2: Auxiliary Phase (Value Distillation & KL Constraint).
        Updates only the shared backbone of Policy Network.
        """
        self.aux_buffer.prepare_targets(self.policy, self.value, device)

        states_t = torch.FloatTensor(self.aux_buffer.states).to(device)
        target_vals_t = torch.FloatTensor(self.aux_buffer.target_values).to(device)
        returns_t = torch.FloatTensor(self.aux_buffer.returns).to(device)
        old_means_t = torch.FloatTensor(self.aux_buffer.target_means).to(device)
        old_stds_t = torch.FloatTensor(self.aux_buffer.target_stds).to(device)

        N = states_t.size(0)
        aux_losses = []

        # Auxiliary optimization loops
        for epoch in range(self.K_aux):
            indices = np.arange(N)
            np.random.shuffle(indices)

            for start in range(0, N, self.batchsize):
                end = start + self.batchsize
                batch_idx = indices[start:end]

                b_states = states_t[batch_idx]
                b_targets = target_vals_t[batch_idx]
                b_returns = returns_t[batch_idx]
                b_old_means = old_means_t[batch_idx]
                b_old_stds = old_stds_t[batch_idx]

                self.policy.train()

                # Current outputs from the policy model
                dist = self.policy.get_distribution(b_states)
                v_policy = self.policy.get_value(b_states)

                # Distillation loss from Separate Value Network predictions
                l_distill = F.mse_loss(v_policy, b_targets)

                # Direct regression loss on buffer return targets
                l_value_buf = F.mse_loss(v_policy, b_returns)

                # KL constraint to prevent policy parameters drift
                old_dist = Normal(b_old_means, b_old_stds)
                l_kl = kl_divergence(old_dist, dist).sum(dim=-1).mean()

                # Joint Auxiliary loss
                loss_aux = self.vf_loss_coef * (l_distill + l_value_buf) + 1.0 * l_kl

                # Update Policy Network parameters (backbone and auxiliary critic)
                self.policy_optimizer.zero_grad()
                loss_aux.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.policy_optimizer.step()

                aux_losses.append(loss_aux.item())

        self.aux_buffer.clear()
        self.t_aux = 0

        return np.mean(aux_losses)

    def save_weights(self, path_prefix='PPG'):
        """Save model checkpoints to disk."""
        os.makedirs(os.path.dirname(path_prefix) if os.path.dirname(path_prefix) else '.', exist_ok=True)

        # Policy Network (backbone + actor + auxiliary critic)
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.policy_optimizer.state_dict(),
            'backbone_name': self.backbone_name,
        }, f'{path_prefix}_policy.tar')

        # Separate Value Network (independent backbone + critic)
        torch.save({
            'model_state_dict': self.value.state_dict(),
            'optimizer_state_dict': self.value_optimizer.state_dict(),
        }, f'{path_prefix}_value.tar')

    def load_weights(self, path_prefix='PPG'):
        """Load model checkpoints from disk."""
        policy_checkpoint = torch.load(f'{path_prefix}_policy.tar', map_location=device)
        self.policy.load_state_dict(policy_checkpoint['model_state_dict'])
        self.policy_optimizer.load_state_dict(policy_checkpoint['optimizer_state_dict'])

        value_checkpoint = torch.load(f'{path_prefix}_value.tar', map_location=device)
        self.value.load_state_dict(value_checkpoint['model_state_dict'])
        self.value_optimizer.load_state_dict(value_checkpoint['optimizer_state_dict'])

    def get_param_count(self):
        """Returns total trainable parameters for Policy and Value networks."""
        policy_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        value_params = sum(p.numel() for p in self.value.parameters() if p.requires_grad)
        return policy_params, value_params
