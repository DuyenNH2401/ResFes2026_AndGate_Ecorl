# ppg_config.py - Default configurations for Phasic Policy Gradient (PPG)

# PPO / Policy Phase Hyperparameters
ENTROPY_COEF = 0.005  # c2 coefficient
VF_LOSS_COEF = 0.25   # c1 coefficient for auxiliary critic head in policy model
BATCHSIZE = 256
PPO_EPOCHS = 10       # K_PPO (epochs of policy phase updates)
N_UPDATE = 2048       # Steps collected between updates
GAMMA = 0.99
LAM = 0.95
LR_POLICY = 3e-4
LR_VALUE = 1e-3

# PPG Auxiliary Phase Hyperparameters
N_AUX = 5             # Auxiliary update after this many policy updates
K_AUX = 10            # Number of epochs for auxiliary update (K_aux)
BETA_KL = 5.0         # Adaptive/target KL penalty coefficient
D_TARG = 0.03         # Target KL divergence
CLIP_VAL = 10.0       # Value function clip range for separate value network
CLIP_EPS = 0.2        # PPO policy clipping epsilon

