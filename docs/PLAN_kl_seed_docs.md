# Plan: Adaptive KL (PPG paper) + Full Seeding + Docs Sync

Hai việc người dùng đã chốt. Cả hai đổi hành vi/cấu hình → có test + đồng bộ 3-lớp config.

## Bối cảnh đã xác nhận
- Người dùng đã có baseline RL (PPO, PPO-Mamba, PPG-Mamba, PPG-Mamba-Lagrange), train 3 seed, có SUMO+GPU.
- `set_seed()` đã seed đủ random/numpy/torch/cuda; `--seed` CLI có (default 55); NHƯNG seed CHƯA vào YAML/ppg_config, CHƯA truyền xuống SUMO, CHƯA bật cudnn deterministic.
- `beta_kl=5.0`, `d_targ=0.03` hiện là DEAD param: KL coef hard-code `1.0` tại `core.py:566`. Người dùng chọn **adaptive KL đầy đủ** (chấp nhận train lại).

---

## Task A — Adaptive KL trong auxiliary phase (PPG paper)

**Cơ chế (Cobbe et al. PPG / PPO adaptive-KL):** sau mỗi auxiliary phase, đo mean KL thực tế; điều chỉnh hệ số phạt β:
- nếu `kl > d_targ * 1.5` → `β *= 2` (phạt mạnh hơn)
- nếu `kl < d_targ / 1.5` → `β /= 2` (nới lỏng)
- clamp β trong `[β_min, β_max]` (vd 0.5 .. 100) để ổn định.

**Files:**
- `ppg/core.py`:
  - `update_auxiliary_phase` (`:518-579`): đổi `loss_aux = vf_loss_coef*(l_distill+l_value_buf) + 1.0*l_kl` → dùng `self.beta_kl * l_kl`; thu thập `l_kl.item()` mỗi batch.
  - Cuối hàm: tính `mean_kl`; cập nhật `self.beta_kl` theo luật trên với `self.d_targ`; clamp `[self.beta_kl_min, self.beta_kl_max]`. Trả về `(mean_aux_loss, mean_kl, self.beta_kl)` (hoặc dict) để log.
  - `__init__`: thêm `beta_kl_min`, `beta_kl_max` (mặc định từ config), giữ `beta_kl` là state thay đổi được.
  - `update()` (`:429-431`): khi aux chạy, nhận thêm mean_kl/beta để đưa vào return dict.
- `ppg/ppg_config.py`: thêm `BETA_KL_MIN=0.5`, `BETA_KL_MAX=100.0` (KL_ADAPT_FACTOR=1.5, BETA_MULT=2.0 có thể hard-code hoặc config).
- `train_ppg.py`: thêm CLI `--beta-kl-min/--beta-kl-max`; log `beta_kl` + `aux_kl` ra console khi aux chạy (CSV optional — cân nhắc cột `beta_kl`).

**Test (`tests/test_adaptive_kl.py` hoặc thêm vào file cũ):** test thuần logic `_adapt_beta(kl, beta, d_targ)` (tách thành staticmethod để test không cần network):
- kl >> d_targ → beta tăng (×2), clamp max.
- kl << d_targ → beta giảm (÷2), clamp min.
- kl ≈ d_targ → beta giữ.

> Tách hàm điều chỉnh β ra staticmethod `PPGAgent._adapt_beta_kl(kl, beta, d_targ, lo, hi)` để test được mà không dựng torch model.

---

## Task B — Seed đầy đủ (config 3-lớp + SUMO + cudnn)

**Files:**
- `ppg/ppg_config.py`: thêm `SEED = 55`.
- `configs/ppg_default.yaml`: thêm `seed: 55` trong section `training:`.
- `train_ppg.py`:
  - `--seed` đổi `default=None` (để 3-lớp: CLI→YAML→config); resolve `seed = args.seed if not None else cfg.SEED`. (Hiện default=55 cứng — đổi sang None + fallback.)
  - `set_seed()`: thêm `torch.backends.cudnn.deterministic=True`, `torch.backends.cudnn.benchmark=False`.
  - Truyền seed xuống env: `env.sim_seed = seed` (hoặc tham số constructor) trước vòng train.
- `simulation/env.py`:
  - `__init__`: nhận/định `self.sim_seed` (mặc định None).
  - `reset` (`:1041`): nếu `self.sim_seed is not None`, thêm `["--seed", str(self.sim_seed)]` vào `SumoCMD` → traffic tái lập được. (Cân nhắc: cộng `step_count`/episode để mỗi episode khác nhau nhưng tái lập theo seed gốc — đơn giản nhất: dùng seed cố định, traffic giống nhau mỗi episode theo seed. Hoặc `--seed seed` + SUMO tự diễn tiến. Chốt: dùng seed cố định truyền vào, đủ cho reproducibility giữa các lần chạy cùng seed.)

**Test:** `set_seed` set cudnn flags (kiểm tra `torch.backends.cudnn.deterministic is True` sau gọi). Reproducibility env khó test không SUMO → chỉ verify SumoCMD chứa `--seed` khi `sim_seed` set (refactor build SumoCMD thành helper `_build_sumo_cmd()` trả list để test).

---

## Task C — Đồng bộ docs (CLAUDE.md + comment)

- CLAUDE.md: sửa "6 backbone (dnn/lstm/...)" → hiện chỉ `mamba` đăng ký (hoặc ghi rõ "các backbone khác là roadmap"). Sửa mô tả `beta_kl/d_targ` → giờ là adaptive KL THẬT (sau Task A đúng rồi). Sửa red_light "−5.0" → −25.0.
- `simulation/env.py:898` comment "× 1.0 = -5.0 / lần" → sửa thành −25.0.

> KHÔNG đụng phần mô tả khác của CLAUDE.md ngoài các điểm sai.

---

## Verify
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v` PASS toàn bộ (cũ + mới).
- `python -c "import train_ppg"` (với fake traci) OK.
- Người dùng tự chạy train lại 3 seed cho arm dùng adaptive KL.

## Commit
Mỗi task 1 commit, trailer Co-Authored-By.

## NGOÀI phạm vi lần này (đề xuất plan riêng sau)
- Non-RL baseline (IDM/Krauss) — tác động cao cho journal.
- Script gộp multi-seed + thống kê (t-test/CI) + plotting.
- Tách metric an toàn/năng lượng độc lập trong eval (collision rate, redlight violation rate, Wh/km, travel time).
