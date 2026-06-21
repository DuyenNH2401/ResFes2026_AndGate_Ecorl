# ppg_config.py - Default configurations for Phasic Policy Gradient (PPG)

# Reproducibility
SEED = 55             # Random seed mặc định (python/numpy/torch/cuda + SUMO)

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
BETA_KL = 5.0         # Adaptive KL penalty coefficient (β khởi tạo, tự điều chỉnh)
D_TARG = 0.03         # Target KL divergence (mục tiêu để adapt β)
BETA_KL_MIN = 0.5     # chặn dưới β
BETA_KL_MAX = 100.0   # chặn trên β
KL_ADAPT_THRESH = 1.5 # ngưỡng nhân/chia: kl>d_targ*thresh→β×2; kl<d_targ/thresh→β÷2
CLIP_VAL = 10.0       # Value function clip range for separate value network
CLIP_EPS = 0.2        # PPO policy clipping epsilon

# ============================================================
#  Adaptive Reward — Lagrangian (an toàn + êm ái + đèn đỏ là ràng buộc)
# ============================================================
LAGRANGIAN_ENABLED   = False   # bật/tắt Lagrangian adaptive weighting
LAMBDA_SAFETY_INIT   = 1.0     # λ khởi tạo cho cost an toàn (following distance)
LAMBDA_COMFORT_INIT  = 0.1     # λ khởi tạo cho cost êm ái (jerk)
LAMBDA_REDLIGHT_INIT = 25.0    # λ khởi tạo cho cost vượt đèn đỏ (~|W_RED_LIGHT| baseline)
LAMBDA_LR            = 0.01    # learning rate cho dual gradient ascent của λ
COST_LIMIT_SAFETY    = 0.02    # ngưỡng cost an toàn trung bình cho phép (d_safety)
COST_LIMIT_COMFORT   = 0.30    # ngưỡng cost êm ái trung bình cho phép (d_comfort)
COST_LIMIT_REDLIGHT  = 0.0     # ngưỡng vượt đèn đỏ trung bình (lý tưởng 0 vi phạm)
LAMBDA_MAX           = 50.0    # chặn trên λ tránh nổ gradient

# ============================================================
#  Adaptive Reward — Curriculum theo lịch episode (baseline so sánh)
# ============================================================
CURRICULUM_ENABLED          = False   # bật/tắt curriculum weighting
CURRICULUM_WARMUP_EPISODES  = 1000    # số episode để W_ENERGY tăng tuyến tính start→end
CURRICULUM_ENERGY_W_START   = 0.0     # |W_ENERGY| đầu training (chưa phạt năng lượng)
CURRICULUM_ENERGY_W_END     = 0.01197655138923567  # |W_ENERGY| cuối = giá trị gốc

