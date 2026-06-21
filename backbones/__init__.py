"""
Backbone registry and factory.

Only the Mamba backbone is registered (PPG-Mamba research focus).
"""

from .base import BaseBackbone
from .mamba import MambaBackbone

# Maps CLI backbone names to their classes.
BACKBONE_REGISTRY = {
    "mamba": MambaBackbone,
}


def create_backbone(name, input_dim, output_dim, **kwargs):
    """
    Instantiate a backbone by registry name.

    Args:
        name: Registry key (e.g. "mamba").
        input_dim: Dimension of the input state vector.
        output_dim: Dimension of the produced feature vector.
        **kwargs: Backbone-specific hyperparameters.

    Returns:
        An instance of the requested backbone.
    """
    key = name.lower()
    if key not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{name}'. "
            f"Available: {list(BACKBONE_REGISTRY.keys())}"
        )
    return BACKBONE_REGISTRY[key](
        input_dim=input_dim,
        output_dim=output_dim,
        **kwargs,
    )


__all__ = ["BaseBackbone", "MambaBackbone", "BACKBONE_REGISTRY", "create_backbone"]
