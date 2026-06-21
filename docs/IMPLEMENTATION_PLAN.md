# Adaptive Reward — Implementation Plan (đã chỉnh theo quyết định người dùng)

Dựa trên `docs/2026-06-21-adaptive-reward.md`, với **3 thay đổi** so với docs gốc:

1. **Tách `cost_redlight` thành cost thứ 3 với λ riêng** (`λ_redlight`) — đồng bộ đầy đủ qua config/CLI/CSV/info. → 3 cost: `safety`, `comfort`, `redlight`.
2. **Theo bản chuẩn PPO-Lagrangian**: `compute_gae` dùng λ của *rollout vừa thu* (λ cũ); `update_lambdas()` cập nhật λ *cho rollout kế tiếp*. Tránh việc áp λ-mới lên chính rollout sinh ra nó.
3. Giữ nguyên mọi nguyên tắc khác của docs: backward-compat tuyệt đối khi tắt cờ, config 3-lớp, test-driven, curriculum cho W_ENERGY.

## Mapping cost (đã chốt)
- `reward_task` = speed_target, too_slow, progress, energy, lane_change, junction, W_TIME.
- `cost_safety`  = `safety_penalty` (∈[0,1], raw).
- `cost_comfort` = `accel_jerk` (raw).
- `cost_redlight`= `red_light_penalty` (1.0/lần vượt, raw).
- `r_eff = reward_task − λ_s·cost_s − λ_c·cost_c − λ_r·cost_r`.

> Lưu ý scale: baseline nhân red_light với −25. Khi Lagrangian bật, λ_redlight tự học mức phạt từ `LAMBDA_REDLIGHT_INIT` (mặc định đặt cao hơn để xuất phát gần baseline — xem Task 3).

## Thứ tự gọi trong train loop (bản chuẩn)
```
if t_updates == n_update_val:
    update_results = agent.update()        # compute_gae dùng λ HIỆN TẠI (của rollout này) → clear_memory
    if lagrangian_enabled:
        lam_info = agent.update_lambdas_from(update_results)  # dùng mean cost đã thu trong update()
```
→ `update()` (khi lagrangian bật) sẽ **tự đọc mean cost trước khi clear_memory** và trả về trong dict; `update_lambdas` chỉ nhận mean cost đó để dịch λ cho rollout sau. Nhờ vậy không phụ thuộc thứ tự đọc/clear, và λ áp cho GAE luôn là λ của rollout hiện tại.

---

## Task 1 — Tách `_calculate_reward` (env.py)
- Thêm vào `__init__` (cạnh JUNCTION_*, ~dòng 48): `self.adaptive_reward_enabled = False` + attrs curriculum (Task 5).
- Sửa return cuối `_calculate_reward` (dòng 947–956):
  - Tính `reward_task` (6 phần + W_TIME), `cost_safety=safety_penalty`, `cost_comfort=accel_jerk`, `cost_redlight=red_light_penalty`.
  - Nếu `getattr(self,'adaptive_reward_enabled',False)`: return tuple `(reward_task, cost_safety, cost_comfort, cost_redlight)`.
  - Ngược lại: return scalar = `reward_task + accel_jerk*W_COMFORT + safety_penalty*W_SAFETY + red_light_penalty*W_RED_LIGHT` (= y hệt công thức gốc 9 phần + W_TIME). **Phải verify bằng test bằng số.**
- Test `tests/test_adaptive_reward.py`: stub env không gọi SUMO (override `_get_junction_proximity`, `_get_dist_to_destination`); kiểm tra tuple-4 khi bật, scalar khi tắt, và **scalar == công thức gốc** (test đẳng thức số học).

## Task 2 — Expose cost qua `info` (env.py `step()`)
- Helper `_split_reward_for_step(reward_out)` → `(reward_task, cost_safety, cost_comfort, cost_redlight)`; scalar → costs=0.
- Tại dòng 1456: thay bằng đọc tuple, cộng `reward_task/SIM_STEPS` vào `reward`, lưu `self._step_cost_safety/comfort/redlight`.
- info dict (1494–1506): thêm `cost_safety`, `cost_comfort`, `cost_redlight` (getattr default 0.0).

## Task 3 — Hằng số config (ppg_config.py, append)
```
LAGRANGIAN_ENABLED=False
LAMBDA_SAFETY_INIT=1.0; LAMBDA_COMFORT_INIT=0.1; LAMBDA_REDLIGHT_INIT=25.0
LAMBDA_LR=0.01
COST_LIMIT_SAFETY=0.02; COST_LIMIT_COMFORT=0.30; COST_LIMIT_REDLIGHT=0.0
LAMBDA_MAX=50.0
CURRICULUM_ENABLED=False; CURRICULUM_WARMUP_EPISODES=1000
CURRICULUM_ENERGY_W_START=0.0; CURRICULUM_ENERGY_W_END=0.01197655138923567
```
> `LAMBDA_REDLIGHT_INIT=25.0` để khởi điểm xấp xỉ baseline (−25). `COST_LIMIT_REDLIGHT=0.0`: lý tưởng 0 vi phạm → λ_redlight chỉ giảm khi thực sự không vi phạm.

## Task 4 — λ state + dual update + GAE (core.py, memory.py)
- `memory.py`: thêm list `cost_safety/cost_comfort/cost_redlight`; `save_eps(...)` nhận 3 kwarg default 0.0; append; `clear_memory` xoá.
- `core.py __init__`: thêm params (sau `clip_eps=None`): `lagrangian_enabled, lambda_safety_init, lambda_comfort_init, lambda_redlight_init, lambda_lr, cost_limit_safety, cost_limit_comfort, cost_limit_redlight, lambda_max` (đều `=None`); resolve fallback về ppg_config.
- `PPGAgent.save_eps(...)`: thêm 3 kwarg, forward xuống cả 2 nhánh memory.
- `compute_gae`: ngay sau `rewards=...`, nếu lagrangian bật và size khớp → `rewards = rewards - λ_s·cs - λ_c·cc - λ_r·cr` (dùng λ HIỆN TẠI = λ của rollout này).
- `update()`: khi lagrangian bật, **trước `clear_memory()`** tính `mean_cost_safety/comfort/redlight`, đưa vào dict trả về (keys: `mean_cost_safety`,...). Không thay đổi hành vi khi tắt.
- `update_lambdas(mean_cs, mean_cc, mean_cr)`: dual ascent `λ←clip(λ+lr·(mean−limit),0,λ_max)` cho cả 3; trả dict λ + mean. (Đổi chữ ký so với docs: nhận mean cost làm tham số thay vì đọc memory — để gọi *sau* `update()` đã clear.)
- Test: λ_safety tăng khi cost>limit, λ_comfort giảm khi cost<limit, λ luôn ≥0; thêm test λ_redlight.

## Task 5 — Curriculum W_ENERGY (env.py)
- `__init__`: `curriculum_enabled=False; curriculum_warmup=1000; curriculum_energy_w_start=0.0; curriculum_energy_w_end=0.01197655138923567; current_episode=0`.
- `set_episode(ep)`, `_curriculum_energy_weight()` (nội suy tuyến tính, clamp; tắt→trả `end`).
- Trong `_calculate_reward`: `W_ENERGY = -self._curriculum_energy_weight()` (dòng 778).
- Test: ramp 0/500/5000 + disabled→base.

## Task 6 — Nối dây train_ppg.py + YAML
- CLI flags: `--lagrangian-enabled, --lambda-safety-init, --lambda-comfort-init, --lambda-redlight-init, --lambda-lr, --cost-limit-safety, --cost-limit-comfort, --cost-limit-redlight, --lambda-max, --curriculum-enabled, --curriculum-warmup, --curriculum-energy-w-start, --curriculum-energy-w-end` (bool dùng lambda parser, default=None).
- Sau tạo env: resolve `_lag_enabled/_cur_enabled` với fallback config; set `env.adaptive_reward_enabled`, các attr curriculum.
- `PPGAgent(...)`: truyền 9 param Lagrangian.
- Đầu episode (sau `reset_history`): `env.set_episode(global_ep_cnt + 1)`.
- `agent.save_eps(...)`: thêm `cost_safety/comfort/redlight=info.get(...,0.0)`.
- Khối update: `update()` trước → nếu lagrangian bật, gọi `update_lambdas` bằng mean cost trong `update_results`, lưu `last_lambda_*` để log + print.
- CSV: thêm cột `lambda_safety, lambda_comfort, lambda_redlight` (header + row). Khởi tạo `last_lambda_*=0.0`.
- YAML `configs/ppg_default.yaml`: section `adaptive_reward:` (mặc định tắt) với đủ key kể cả redlight.

## Task 7 — Verify
- `python -m pytest tests/test_adaptive_reward.py -v` PASS toàn bộ (gồm test đẳng thức backward-compat).
- `python -c "import train_ppg"` OK.
- (Nếu có SUMO) smoke 3 arm — ghi chú để người dùng tự chạy nếu môi trường thiếu SUMO_HOME.

## Commit
Mỗi task 1 commit (như docs), trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Làm trên nhánh mới (đang ở nhánh mặc định thì branch trước).
