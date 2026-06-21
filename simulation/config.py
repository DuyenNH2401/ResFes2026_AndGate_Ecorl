import os

# Đường dẫn map
SUMOCFG_PATH = os.path.join(os.path.dirname(__file__), "map", "map", "run.sumocfg")

# ============================================================
# VF9 Physics Parameters (dùng cho MMPEVEM & fallback model)
# ============================================================
VF9_MASS = 2911.0           # kg (ECO trim curb weight)
VF9_CD = 0.35               # Drag coefficient (ước tính SUV lớn, chưa công bố chính thức)
VF9_FRONTAL_AREA = 3.1      # m² (ước tính từ kích thước 2.0m rộng x 1.78m cao)
VF9_CR = 0.01               # Rolling resistance coefficient (lốp EV low-RR)
VF9_WHEEL_RADIUS = 0.375    # m (lốp 20-inch, ~265/50R20)
VF9_ETA_DRIVE = 0.90        # Hiệu suất powertrain khi kéo (motor + inverter)
VF9_ETA_REGEN = 0.75        # Hiệu suất thu hồi năng lượng (regen braking)
VF9_P_AUX = 300.0           # W — Phụ tải phụ trợ (A/C, hệ thống điện, etc.)
VF9_MAX_REGEN_POWER = 70000.0  # W — Giới hạn công suất regen tối đa (~70kW)
AIR_DENSITY = 1.225         # kg/m³ (mật độ không khí ở mực nước biển, 15°C)
GRAVITY = 9.81              # m/s²