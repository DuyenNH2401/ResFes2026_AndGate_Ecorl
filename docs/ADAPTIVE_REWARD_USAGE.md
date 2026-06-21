# Adaptive Reward — Hướng dẫn chạy & verify

## Chạy unit test (không cần SUMO)

Môi trường này có một pytest plugin ROS hỏng (`launch_testing`) làm crash collection,
nên cần TẮT autoload plugin:

```bash
cd /home/duyennh/AI_projects/ecorl_adaptive_shaping
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_adaptive_reward.py -v
```

`tests/conftest.py` tự inject một module `traci` giả + `SUMO_HOME` giả để import được
`simulation/env.py` mà không cần cài SUMO. KHÔNG ảnh hưởng code production.

## Smoke train 3 chế độ (CẦN SUMO_HOME thật)

Môi trường review không có SUMO → người dùng tự chạy 3 lệnh sau để xác nhận end-to-end:

```bash
# Baseline (cả hai cờ tắt — reward y hệt gốc)
python train_ppg.py --backbone mamba --n-episode 2 --exp-name smoke_baseline

# Lagrangian
python train_ppg.py --backbone mamba --n-episode 2 --exp-name smoke_lagrangian \
    --lagrangian-enabled true

# Curriculum
python train_ppg.py --backbone mamba --n-episode 2 --exp-name smoke_curriculum \
    --curriculum-enabled true --curriculum-warmup 2
```

Kỳ vọng: cả 3 chạy không crash; chế độ Lagrangian in dòng `[LAGRANGIAN] λ_s=...`;
CSV có 15 cột (thêm `lambda_safety, lambda_comfort, lambda_redlight`);
`run_config.yaml` chứa các tham số adaptive mới.

## Thí nghiệm so sánh paper

```bash
# Arm 0 — Baseline
python train_ppg.py --backbone mamba --exp-name baseline

# Arm 1 — Lagrangian (phương pháp chính)
python train_ppg.py --backbone mamba --exp-name lagrangian \
    --lagrangian-enabled true --lambda-lr 0.01 \
    --cost-limit-safety 0.02 --cost-limit-comfort 0.30 --cost-limit-redlight 0.0

# Arm 2 — Curriculum (baseline so sánh)
python train_ppg.py --backbone mamba --exp-name curriculum \
    --curriculum-enabled true --curriculum-warmup 1000

# Arm 3 — Kết hợp (ablation)
python train_ppg.py --backbone mamba --exp-name combined \
    --lagrangian-enabled true --curriculum-enabled true
```

## Tinh chỉnh sau (lúc chạy thí nghiệm)
- Chạy baseline trước, đọc `mean_cost_safety/comfort/redlight` (in qua `[LAGRANGIAN]`)
  → đặt `cost_limit_*` ngay dưới mức trung bình baseline để tạo áp lực ràng buộc.
- λ dao động mạnh → giảm `--lambda-lr` (vd 0.005); λ phản ứng chậm → tăng (vd 0.02).
- `red_light`: baseline phạt rất nặng (W=-25). λ_redlight khởi tạo 25.0 để xuất phát
  gần baseline; `cost_limit_redlight=0` ép về 0 vi phạm.
