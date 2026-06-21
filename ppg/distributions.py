"""
PPG Distributions
=================
Probability distribution utilities for continuous actions (Gaussian).
Adapted from ppo/distributions.py.
"""

import torch
from torch.distributions import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Continous:
    """Continuous action distribution utilities (Normal distribution)."""

    def sample(self, mean, std):
        distribution = Normal(mean, std)
        return distribution.sample().float().to(device)

    def entropy(self, mean, std):
        distribution = Normal(mean, std)
        # Sum over action dimension for joint entropy
        return distribution.entropy().sum(dim=-1, keepdim=True).float().to(device)

    def logprob(self, mean, std, value_data):
        distribution = Normal(mean, std)
        # Sum over action dimension for joint log probability
        return distribution.log_prob(value_data).sum(dim=-1, keepdim=True).float().to(device)
