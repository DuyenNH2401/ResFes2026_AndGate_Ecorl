"""
Mamba Backbone — Selective State Space Model
=============================================
Extracted from ppg/ppg_mamba.py.

This module implements a Mamba-inspired selective state space model backbone.

Key Ideas:
  - Input-dependent discretization: Δ, B, and C are functions of the input
  - Causal depthwise 1D convolution for local context
  - Linear recurrent selective scan
  - Pure PyTorch reference-style scan implementation

Architecture:
  Input (B, input_dim)
    → Pad & Reshape to (B, seq_len, token_dim)
    → Input Projection + Positional Embedding
    → [MambaBlock × N]
    → LayerNorm + Mean Pool
    → Output Head → (B, output_dim)

Each MambaBlock:
  Input
    → LayerNorm
    → SelectiveSSM
    → Dropout
    → Residual Add
    → Output

Reference:
  - Mamba: https://arxiv.org/abs/2312.00752
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseBackbone


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model - Mamba/S6-style core mechanism.

    Implements input-dependent discretization and selective scan:

        h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
        y_t = C_t * h_t + D * x_t

    where:

        A_bar_t = exp(delta_t * A)
        B_bar_t ≈ delta_t * B_t

    Δ, B, and C are input-dependent, allowing the model to selectively
    propagate, forget, or expose information along the sequence.

    This is a pure PyTorch reference-style implementation, not the official
    fused CUDA/Triton Mamba scan.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)

        # Input projection:
        # Projects to 2 * d_inner, then splits into x and z branches.
        self.in_proj = nn.Linear(
            self.d_model,
            self.d_inner * 2,
            bias=False,
        )

        # Causal depthwise 1D convolution for local context.
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )

        # Input-dependent SSM parameter projections.
        # Produces:
        #   dt: low-rank delta representation
        #   B:  input-dependent input matrix
        #   C:  input-dependent output matrix
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + self.d_state * 2,
            bias=False,
        )

        self.dt_proj = nn.Linear(
            self.dt_rank,
            self.d_inner,
            bias=True,
        )

        # Initialize dt projection weight like Mamba.
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(
            self.dt_proj.weight,
            -dt_init_std,
            dt_init_std,
        )

        # Initialize dt bias so softplus(dt_bias) starts in a stable range.
        dt = torch.exp(
            torch.rand(self.d_inner)
            * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)

        # Inverse of softplus.
        inv_dt = dt + torch.log(-torch.expm1(-dt))

        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # Some training frameworks respect this convention and avoid
        # accidentally reinitializing this bias.
        self.dt_proj.bias._no_reinit = True

        # S4D-style real initialization for diagonal A.
        # A is stored in log-space and made negative in forward().
        A = torch.arange(
            1,
            self.d_state + 1,
            dtype=torch.float32,
        ).unsqueeze(0).expand(self.d_inner, -1)

        self.A_log = nn.Parameter(torch.log(A.contiguous()))
        self.A_log._no_weight_decay = True

        # D skip parameter.
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # Output projection back to d_model.
        self.out_proj = nn.Linear(
            self.d_inner,
            self.d_model,
            bias=False,
        )

        self.act = nn.SiLU()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, L, d_model)

        Returns:
            Tensor of shape (B, L, d_model)
        """
        if x.dim() != 3:
            raise ValueError(
                f"SelectiveSSM expected input of shape (B, L, D), "
                f"got {tuple(x.shape)}"
            )

        batch, seqlen, dim = x.shape

        if dim != self.d_model:
            raise ValueError(
                f"SelectiveSSM expected last dimension {self.d_model}, "
                f"got {dim}"
            )

        # Project and split into x and z branches.
        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x_branch, z = xz.chunk(2, dim=-1)  # each: (B, L, d_inner)

        # Causal depthwise convolution.
        x_branch = x_branch.transpose(1, 2)  # (B, d_inner, L)
        x_branch = self.conv1d(x_branch)[..., :seqlen]  # causal trim
        x_branch = self.act(x_branch)
        x_branch = x_branch.transpose(1, 2)  # (B, L, d_inner)

        # Project to input-dependent SSM parameters.
        x_dbl = self.x_proj(x_branch)

        dt, B, C = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )

        # Project low-rank dt to full inner dimension.
        dt = self.dt_proj(dt)
        dt = F.softplus(dt)

        # Negative diagonal state matrix.
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # Selective scan.
        y = self._selective_scan(
            x=x_branch,
            dt=dt,
            A=A,
            B=B,
            C=C,
        )

        # D skip connection and z gate.
        y = y + x_branch * self.D.unsqueeze(0).unsqueeze(0)
        y = y * self.act(z)

        # Project back to model dimension.
        output = self.out_proj(y)

        return output

    def _selective_scan(self, x, dt, A, B, C):
        """
        Pure PyTorch reference-style selective scan.

        Args:
            x:  Tensor of shape (B, L, d_inner)
            dt: Tensor of shape (B, L, d_inner)
            A:  Tensor of shape (d_inner, d_state)
            B:  Tensor of shape (B, L, d_state)
            C:  Tensor of shape (B, L, d_state)

        Returns:
            y: Tensor of shape (B, L, d_inner)
        """
        batch, seqlen, d_inner = x.shape
        d_state = A.shape[1]

        original_dtype = x.dtype

        # Use fp32 internally for better numerical stability.
        x = x.float()
        dt = dt.float()
        A = A.float()
        B = B.float()
        C = C.float()

        h = torch.zeros(
            batch,
            d_inner,
            d_state,
            device=x.device,
            dtype=torch.float32,
        )

        ys = []

        for t in range(seqlen):
            # dA_t: (B, d_inner, d_state)
            dA_t = torch.exp(
                dt[:, t].unsqueeze(-1) * A.unsqueeze(0)
            )

            # dB_x_t: (B, d_inner, d_state)
            dB_x_t = (
                dt[:, t].unsqueeze(-1)
                * B[:, t].unsqueeze(1)
                * x[:, t].unsqueeze(-1)
            )

            # Recurrent state update.
            h = dA_t * h + dB_x_t

            # Project state to output.
            # C[:, t]: (B, d_state)
            # h:       (B, d_inner, d_state)
            # y_t:     (B, d_inner)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)

            ys.append(y_t)

        y = torch.stack(ys, dim=1)

        return y.to(dtype=original_dtype)


class MambaBlock(nn.Module):
    """
    Single Mamba-style block with pre-normalization and residual connection.

    Structure:
        Input
          → LayerNorm
          → SelectiveSSM
          → Dropout
          → Residual Add
          → Output
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.mamba = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, L, d_model)

        Returns:
            Tensor of shape (B, L, d_model)
        """
        residual = x

        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)

        return x + residual


class MambaBackbone(BaseBackbone):
    """
    Stacked Mamba-style backbone.

    This backbone is designed for flat input vectors. It pads the input if
    needed, reshapes it into a short sequence of tokens, processes those tokens
    with Mamba-style blocks, then mean-pools the sequence.

    Args:
        input_dim: Dimension of input state.
        output_dim: Dimension of output.
        d_model: Internal model dimension.
        n_layers: Number of stacked Mamba blocks.
        d_state: SSM state dimension.
        d_conv: Causal convolution kernel size.
        expand: Expansion factor for inner dimension.
        seq_len: Number of pseudo-tokens to reshape the input into.
        dropout: Dropout rate.
        
    """


    # def __init__( từ ngày 27/05, số layer của Mamba thay đổi từ 4 thành 2
    #     self,
    #     input_dim,
    #     output_dim,
    #     d_model=None,
    #     n_layers=4,
    #     d_state=16,
    #     d_conv=5,
    #     expand=2,
    #     seq_len=7,
    #     dropout=0.2952068146366997,
    #     **kwargs,
    # ): 
    def __init__(
        self,
        input_dim,
        output_dim,
        d_model=None,
        n_layers=2,
        d_state=16,
        d_conv=4,
        expand=2,
        seq_len=5,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(input_dim, output_dim)

        if d_model is None:
            d_model = output_dim
        self.d_model = d_model
        self.seq_len = seq_len

        # Compute padded input dimension so it divides evenly into seq_len.
        self.padded_dim = math.ceil(input_dim / seq_len) * seq_len
        self.token_dim = self.padded_dim // seq_len

        # Token projection: token_dim -> d_model.
        self.input_proj = nn.Sequential(
            nn.Linear(self.token_dim, d_model),
            nn.SiLU(),
        )

        # Learnable positional embedding for pseudo-token positions.
        self.pos_embedding = nn.Parameter(
            torch.randn(1, seq_len, d_model) * 0.02
        )

        # Stacked Mamba-style blocks.
        self.mamba_layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # Output head.
        self.output_norm = nn.LayerNorm(d_model)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, input_dim) or (input_dim,)

        Returns:
            Tensor of shape (B, output_dim)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        if x.dim() != 2:
            raise ValueError(
                f"MambaBackbone expected input of shape "
                f"(B, input_dim) or (input_dim,), got {tuple(x.shape)}"
            )

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"MambaBackbone expected input_dim {self.input_dim}, "
                f"got {x.shape[-1]}"
            )

        batch_size = x.shape[0]

        # Pad to make input divisible by seq_len.
        if self.input_dim != self.padded_dim:
            x = F.pad(
                x,
                (0, self.padded_dim - self.input_dim),
            )

        # Reshape flat input vector into pseudo-token sequence:
        # (B, input_dim) -> (B, seq_len, token_dim)
        x = x.reshape(
            batch_size,
            self.seq_len,
            self.token_dim,
        )

        # Project tokens and add positional embedding.
        x = self.input_proj(x)
        x = x + self.pos_embedding

        # Process through Mamba-style blocks.
        for layer in self.mamba_layers:
            x = layer(x)

        # Normalize and pool over sequence dimension.
        x = self.output_norm(x)
        x = x.mean(dim=1)

        # Project to output.
        output = self.output_head(x)

        return output