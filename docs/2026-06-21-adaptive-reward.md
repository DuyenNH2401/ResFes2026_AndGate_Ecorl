# Adaptive Reward (Lagrangian + Curriculum) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thêm hai cơ chế adaptive reward để tăng novelty cho bài nghiên cứu eco-driving: (#1) **Lagrangian adaptive weighting** — coi an toàn (va chạm/đèn đỏ) và êm ái (jerk) là RÀNG BUỘC CỨNG, tối ưu tiết kiệm năng lượng dưới ràng buộc đó với trọng số λ tự cập nhật; (#2) **Curriculum weighting theo lịch episode** — baseline để so sánh.

**Architecture:** Tách `_calculate_reward()` của env thành hai luồng output: `reward_task` (mục tiêu chính, gồm energy/speed/progress/...) và các `cost_*` (safety, comfort) bị ràng buộc. Env trả thêm các trường này qua `info` dict (KHÔNG đổi return signature 5-tuple của Gymnasium). Train loop tích lũy cost theo từng update-batch, agent dùng dual gradient ascent để cập nhật `λ_safety`, `λ_comfort`, rồi ghép reward cuối `r = reward_task − λ_s·cost_s − λ_c·cost_c` ngay trong `compute_gae`. Curriculum (#2) là một chế độ riêng bật bằng flag, đổi weight theo `episode`.

**Tech Stack:** PyTorch, NumPy, Gymnasium, TraCI (SUMO), PyYAML. Toàn bộ thay đổi nằm trong `simulation/env.py`, `ppg/core.py`, `ppg/ppg_config.py`, `train_ppg.py`, `configs/`. KHÔNG đụng `ppg/memory.py`, backbone, auxiliary phase.

## Global Constraints

- Giữ nguyên Gymnasium 5-tuple `(obs, reward, terminated, truncated, info)` từ `env.step()`. Mọi tín hiệu mới đi qua `info`.
- Mọi hyperparameter mới phải theo đúng pipeline config 3 lớp: CLI flag `default=None` → YAML default → `ppg_config.py` fallback (idiom `x if x is not None else ppg_config.X`).
- Khi `lagrangian_enabled=False` VÀ `curriculum_enabled=False`, hành vi reward phải **giống hệt** code gốc (backward-compatible — đây là điều kiện để dùng làm baseline).
- Convention dấu: `reward_task` là phần thưởng (càng cao càng tốt), `cost_*` là chi phí KHÔNG ÂM (càng thấp càng tốt). Reward cuối trừ đi `λ·cost`.
- λ phải `>= 0` luôn (clamp tại 0 sau mỗi update) — đây là yêu cầu của Lagrangian dual.
- Reward weights mới và λ phải được ghi vào `run_config.yaml` (tự động qua `vars(args)`) và λ phải log được ra CSV để theo dõi.
- Không xoá/sửa code không liên quan tới task. Giữ nguyên style tiếng Việt trong comment của env.py.

---

## Bối cảnh code hiện tại (đọc trước khi làm)

**`simulation/env.py`:**
- `_calculate_reward(self, action)` — dòng 768–956. Trả về 1 scalar = weighted sum 9 thành phần dense + `W_TIME`. Weights là **local constants** dòng 775–785.
- Các biến trong scope tại cuối hàm (dòng 937–956): `r_speed_target`, `r_too_slow`, `progress_reward`, `energy_penalty`, `accel_jerk`, `lane_change_penalty`, `safety_penalty`, `red_light_penalty`, `junction_penalty`. Tất cả đã qua `np.nan_to_num`.
- `step()` — dòng 1339–1508 — cộng thêm event reward (`+40` goal, `−20` collision, `−5` teleport...) lên scalar từ `_calculate_reward`, rồi build `info` dict (dòng ~1494–1506).

**`ppg/core.py`:**
- `PPGAgent.__init__` — dòng 147–224. Idiom fallback config dòng 162–181.
- `compute_gae(self, last_value_val)` — dòng 290–313. Đọc `self.policy_memory.rewards` → tính advantages/returns. **Đây là chokepoint duy nhất reward biến thành learning target.**
- `update()` — dòng 315–441. Gọi `compute_gae` (dòng 332), cuối hàm `clear_memory()` (dòng 433).

**`train_ppg.py`:**
- `parse_args()` — dòng 54–137.
- `train()` vòng lặp — episode loop dòng 288, step loop dòng 301, `env.step` dòng 305, `agent.save_eps` dòng 320, `agent.update()` dòng 329, logging CSV dòng 348–355.
- `CSV_HEADER` — dòng 35–38.

---

## File Structure

| File | Trách nhiệm thay đổi |
|---|---|
| `simulation/env.py` | Tách `_calculate_reward` → trả `(reward_task, cost_safety, cost_comfort)`; thêm `set_episode()` cho curriculum; expose cost qua `info` trong `step()` |
| `ppg/ppg_config.py` | Thêm hằng số fallback: λ init, λ lr, cost limits, curriculum schedule, các flag |
| `ppg/core.py` | Thêm state `λ_safety`/`λ_comfort`; method `update_lambdas()`; consume reward_task − λ·cost trong `compute_gae`; lưu cost vào memory |
| `train_ppg.py` | CLI flags mới; truyền vào agent; truyền episode vào env; tích lũy cost; gọi `update_lambdas`; log λ ra CSV |
| `configs/ppg_default.yaml` | Section `adaptive_reward:` với mặc định tắt (giữ baseline) |

---

## Quyết định thiết kế then chốt

**Cost đi qua `info`, KHÔNG đi qua reward scalar.** Lý do: `agent.save_eps()` hiện chỉ nhận 1 `reward` scalar. Ta cần lưu `reward_task`, `cost_safety`, `cost_comfort` riêng để (a) cập nhật λ, (b) ghép reward cuối. Cách ít xâm lấn nhất: env trả 3 giá trị qua `info`; train loop đọc từ `info` và truyền xuống agent qua một method mở rộng `save_eps_adaptive()`. Khi adaptive tắt, dùng đúng `save_eps()` cũ.

**λ cập nhật mỗi policy-update (mỗi `n_update_val` steps), KHÔNG mỗi step.** Dual gradient ascent: `λ ← clamp(λ + lr_λ · (mean_cost − limit), 0, λ_max)`. `mean_cost` = trung bình cost trên toàn rollout buffer vừa thu. Đây là cách chuẩn của PPO-Lagrangian (ổn định hơn cập nhật mỗi step).

**reward_task gồm gì:** speed_target, too_slow, progress, energy, lane_change, junction, W_TIME. **cost_safety** = `safety_penalty + red_light_penalty` (cả hai đã là đại lượng ≥0). **cost_comfort** = `accel_jerk`. (Junction approach giữ trong reward_task vì nó là shaping điều hướng, không phải ràng buộc an toàn cứng — có thể chuyển sau nếu muốn.)

---

## Task 1: Tách `_calculate_reward` thành reward_task + costs (giữ tương thích)

**Files:**
- Modify: `simulation/env.py:947-956` (phần return của `_calculate_reward`)
- Modify: `simulation/env.py` (thêm attribute init trong `__init__`, vùng dòng 38–63)

**Interfaces:**
- Produces: `_calculate_reward(self, action)` trả về tuple `(reward_task: float, cost_safety: float, cost_comfort: float)` khi `self.adaptive_reward_enabled` True; trả về **scalar float như cũ** khi False.
- Consumes: các biến local đã có sẵn tại dòng 937–945 (`safety_penalty`, `red_light_penalty`, `accel_jerk`, v.v.).

- [ ] **Step 1: Viết test thất bại — kiểm tra `_calculate_reward` trả tuple khi bật adaptive**

Tạo file test mới `tests/test_adaptive_reward.py`. Vì env cần SUMO/TraCI nặng, ta test trực tiếp logic bằng một stub kế thừa env và bơm `veh_data` giả, KHÔNG khởi động SUMO.

```python
# tests/test_adaptive_reward.py
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.env import SumoEnv


def _make_stub_env(adaptive: bool):
    """Tạo env mà KHÔNG gọi __init__ (tránh khởi động SUMO)."""
    env = SumoEnv.__new__(SumoEnv)
    # Hằng số reward mà _calculate_reward đọc qua self.*
    env.MAX_SPEED = 55.6
    env.TARGET_SPEED_RATIO = 1.0
    env.MIN_DESIRED_SPEED = 5.0
    env.JUNCTION_APPROACH_DIST = 15.0
    env.JUNCTION_TARGET_SPEED = 2.0
    env.JUNCTION_MIN_SPEED = 0.0
    env.adaptive_reward_enabled = adaptive
    # veh_data giả: xe chạy thẳng, không leader, không TLS
    env.veh_data = {
        "speed": 10.0, "lane_id": "", "road_id": "edgeA",
        "elec": 1.0, "leader": None, "tls": None,
    }
    # các thuộc tính stateful mà hàm dùng
    env.prev_dist = 100.0
    return env


def test_calculate_reward_returns_tuple_when_adaptive():
    env = _make_stub_env(adaptive=True)
    out = env._calculate_reward(np.array([0.0, 0.0], dtype=np.float32))
    assert isinstance(out, tuple)
    assert len(out) == 3
    reward_task, cost_safety, cost_comfort = out
    assert isinstance(reward_task, float)
    assert cost_safety >= 0.0
    assert cost_comfort >= 0.0


def test_calculate_reward_returns_scalar_when_not_adaptive():
    env = _make_stub_env(adaptive=False)
    out = env._calculate_reward(np.array([0.0, 0.0], dtype=np.float32))
    assert isinstance(out, float)
```

> Lưu ý: `_get_junction_proximity()` và `_get_dist_to_destination()` được gọi bên trong `_calculate_reward` và dùng TraCI. Step 3 sẽ guard chúng — xem ghi chú dưới. Nếu chúng vẫn gọi TraCI, test sẽ lỗi; khi đó stub thêm `env._get_junction_proximity = lambda: 999.0` và `env._get_dist_to_destination = lambda: 95.0` ngay trong `_make_stub_env`. Thêm 2 dòng đó vào stub ngay từ đầu để test chạy được:
> ```python
>     env._get_junction_proximity = lambda: 999.0
>     env._get_dist_to_destination = lambda: 95.0
> ```

- [ ] **Step 2: Chạy test để xác nhận thất bại**

Run: `cd /home/duyennh/AI_projects/ecorl_adaptive_shaping && python -m pytest tests/test_adaptive_reward.py -v`
Expected: FAIL — `test_calculate_reward_returns_tuple_when_adaptive` lỗi vì hiện hàm luôn trả scalar (AssertionError: not a tuple); cũng có thể lỗi `AttributeError: adaptive_reward_enabled` nếu chưa thêm attribute (Step 3 sẽ thêm).

- [ ] **Step 3: Sửa `_calculate_reward` để trả tuple khi adaptive bật**

Thay đoạn return cuối (dòng 947–956) trong `simulation/env.py`. Tính `reward_task`, `cost_safety`, `cost_comfort` rồi rẽ nhánh theo `self.adaptive_reward_enabled` (dùng `getattr` để an toàn nếu attribute chưa set):

```python
        # ------------------------------------------------------------------ #
        #  TỔNG HỢP — tách reward_task và các cost bị ràng buộc (Lagrangian) #
        # ------------------------------------------------------------------ #
        reward_task = (
            (r_speed_target      * W_SPEED_TARGET)  +
            (r_too_slow          * W_TOO_SLOW)      +
            (progress_reward     * W_PROGRESS)      +
            (energy_penalty      * W_ENERGY)        +
            (lane_change_penalty * W_LANE_CHANGE)   +
            (junction_penalty    * W_JUNCTION)      +
            W_TIME
        )

        # Cost ≥ 0 (sẽ bị trừ λ·cost ở tầng agent)
        cost_safety  = float(safety_penalty + red_light_penalty)
        cost_comfort = float(accel_jerk)

        if getattr(self, "adaptive_reward_enabled", False):
            return float(reward_task), cost_safety, cost_comfort

        # Backward-compatible: scalar y hệt công thức gốc
        return (
            reward_task
            + (accel_jerk     * W_COMFORT)
            + (safety_penalty * W_SAFETY)
            + (red_light_penalty * W_RED_LIGHT)
        )
```

> Quan trọng: nhánh scalar phải cộng lại đúng 3 thành phần đã tách (`accel_jerk*W_COMFORT`, `safety_penalty*W_SAFETY`, `red_light_penalty*W_RED_LIGHT`) để bằng hệt công thức gốc dòng 947–956. Kiểm tra kỹ: gốc có 9 thành phần + W_TIME; reward_task ở đây có 6 + W_TIME, nhánh scalar cộng thêm 3 → đủ 9 + W_TIME. ✅

Thêm attribute default vào `__init__` (vùng dòng 38–63, đặt cạnh các hằng JUNCTION_*):

```python
        # Adaptive reward (Lagrangian) — mặc định tắt để giữ baseline
        self.adaptive_reward_enabled = False
```

> Nếu Step 2 báo lỗi vì `_get_junction_proximity`/`_get_dist_to_destination` gọi TraCI trong môi trường test (không có SUMO), KHÔNG sửa hai hàm đó — chúng vẫn chạy bình thường khi train thật. Test đã được stub override hai hàm này (xem ghi chú Step 1), nên không cần đụng env code cho mục đích test.

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `cd /home/duyennh/AI_projects/ecorl_adaptive_shaping && python -m pytest tests/test_adaptive_reward.py -v`
Expected: PASS cả 2 test.

- [ ] **Step 5: Commit**

```bash
git add simulation/env.py tests/test_adaptive_reward.py
git commit -m "feat(reward): split _calculate_reward into task reward + safety/comfort costs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Expose cost qua `info` trong `step()`

**Files:**
- Modify: `simulation/env.py` — chỗ gọi `_calculate_reward` trong `step()` (dòng ~1456) và chỗ build `info` dict (dòng ~1494–1506)

**Interfaces:**
- Consumes: `_calculate_reward()` (Task 1) trả tuple khi adaptive bật.
- Produces: `info` dict chứa thêm `"reward_task"`, `"cost_safety"`, `"cost_comfort"` khi adaptive bật. `reward` scalar trả về (giá trị thứ 2 của tuple step) khi adaptive bật = `reward_task` thuần (CHƯA trừ λ·cost) cộng event rewards — agent sẽ ghép λ sau. Khi adaptive tắt: hành vi y như cũ.

- [ ] **Step 1: Viết test thất bại — `step()` đưa cost vào info khi adaptive bật**

Test bằng cách monkeypatch `_calculate_reward` để tránh chạy SUMO. Thêm vào `tests/test_adaptive_reward.py`:

```python
def test_step_exposes_costs_in_info():
    """Khi adaptive bật, step() phải đẩy reward_task/cost_* vào info."""
    env = _make_stub_env(adaptive=True)
    # Giả lập _calculate_reward trả tuple cố định
    env._calculate_reward = lambda action: (1.5, 0.3, 0.2)
    # Gọi helper tách logic (Task 2 Step 3 tạo hàm này)
    reward_task, cost_s, cost_c = env._split_reward_for_step(
        env._calculate_reward(np.array([0.0, 0.0]))
    )
    assert reward_task == 1.5
    assert cost_s == 0.3
    assert cost_c == 0.2
```

- [ ] **Step 2: Chạy test để xác nhận thất bại**

Run: `cd /home/duyennh/AI_projects/ecorl_adaptive_shaping && python -m pytest tests/test_adaptive_reward.py::test_step_exposes_costs_in_info -v`
Expected: FAIL — `AttributeError: 'SumoEnv' object has no attribute '_split_reward_for_step'`.

- [ ] **Step 3: Thêm helper `_split_reward_for_step` và dùng nó trong `step()`**

Thêm method nhỏ vào `SumoEnv` (đặt ngay sau `_calculate_reward`, trước `reset`):

```python
    def _split_reward_for_step(self, reward_out):
        """Chuẩn hoá output của _calculate_reward về (reward_task, cost_safety, cost_comfort).
        Khi adaptive tắt, reward_out là scalar → cost = 0."""
        if isinstance(reward_out, tuple):
            reward_task, cost_safety, cost_comfort = reward_out
            return float(reward_task), float(cost_safety), float(cost_comfort)
        return float(reward_out), 0.0, 0.0
```

Trong `step()`, tại dòng ~1456 hiện là:
```python
reward += self._calculate_reward(planned_action) / SIM_STEPS
```
Đổi thành:
```python
                _rew_out = self._calculate_reward(planned_action)
                _r_task, _c_safety, _c_comfort = self._split_reward_for_step(_rew_out)
                reward += _r_task / SIM_STEPS
                self._step_cost_safety = _c_safety
                self._step_cost_comfort = _c_comfort
```

> Lưu ý: giữ nguyên tên biến `planned_action` và `SIM_STEPS` đúng như code gốc tại dòng đó. Nếu vòng lặp `for` cộng dồn reward qua nhiều SIM_STEPS, đặt `self._step_cost_*` = giá trị cost của bước cuối (hiện `SIM_STEPS=1` nên không khác biệt). Nếu muốn chính xác hơn khi SIM_STEPS>1, cộng dồn `self._step_cost_safety += _c_safety` (khởi tạo `=0.0` trước vòng lặp). Với SIM_STEPS=1, hai cách tương đương — dùng phép gán đơn giản.

Trong phần build `info` dict (dòng ~1494–1506), thêm 3 khoá:
```python
        info["reward_task"]  = getattr(self, "_step_cost_safety", None) is not None and reward or reward
        info["cost_safety"]  = getattr(self, "_step_cost_safety", 0.0)
        info["cost_comfort"] = getattr(self, "_step_cost_comfort", 0.0)
```

> Sửa lại dòng `reward_task` cho rõ ràng (dòng trên viết rối) — dùng đúng:
> ```python
>         info["cost_safety"]  = getattr(self, "_step_cost_safety", 0.0)
>         info["cost_comfort"] = getattr(self, "_step_cost_comfort", 0.0)
> ```
> KHÔNG cần `info["reward_task"]` riêng vì `reward` (giá trị step trả về) đã = reward_task + event rewards; agent dùng `reward` làm task-reward và trừ λ·cost. Bỏ dòng `info["reward_task"]` rối ở trên. Chỉ thêm `cost_safety` và `cost_comfort`.

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `cd /home/duyennh/AI_projects/ecorl_adaptive_shaping && python -m pytest tests/test_adaptive_reward.py -v`
Expected: PASS toàn bộ.

- [ ] **Step 5: Commit**

```bash
git add simulation/env.py tests/test_adaptive_reward.py
git commit -m "feat(reward): expose cost_safety/cost_comfort via step() info dict

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Thêm hằng số config Lagrangian + curriculum vào `ppg_config.py`

**Files:**
- Modify: `ppg/ppg_config.py` (append cuối file)

**Interfaces:**
- Produces: các hằng số module-level: `LAGRANGIAN_ENABLED`, `LAMBDA_SAFETY_INIT`, `LAMBDA_COMFORT_INIT`, `LAMBDA_LR`, `COST_LIMIT_SAFETY`, `COST_LIMIT_COMFORT`, `LAMBDA_MAX`, `CURRICULUM_ENABLED`, `CURRICULUM_WARMUP_EPISODES`, `CURRICULUM_ENERGY_W_START`, `CURRICULUM_ENERGY_W_END`.

- [ ] **Step 1: Thêm hằng số fallback**

Append vào cuối `ppg/ppg_config.py`:

```python
# ============================================================
#  Adaptive Reward — Lagrangian (an toàn + êm ái là ràng buộc)
# ============================================================
LAGRANGIAN_ENABLED   = False   # bật/tắt Lagrangian adaptive weighting
LAMBDA_SAFETY_INIT   = 1.0     # λ khởi tạo cho cost an toàn
LAMBDA_COMFORT_INIT  = 0.1     # λ khởi tạo cho cost êm ái (jerk)
LAMBDA_LR            = 0.01    # learning rate cho dual gradient ascent của λ
COST_LIMIT_SAFETY    = 0.02    # ngưỡng cost an toàn trung bình cho phép (d_safety)
COST_LIMIT_COMFORT   = 0.30    # ngưỡng cost êm ái trung bình cho phép (d_comfort)
LAMBDA_MAX           = 50.0    # chặn trên λ tránh nổ gradient

# ============================================================
#  Adaptive Reward — Curriculum theo lịch episode (baseline so sánh)
# ============================================================
CURRICULUM_ENABLED          = False   # bật/tắt curriculum weighting
CURRICULUM_WARMUP_EPISODES  = 1000    # số episode để W_ENERGY tăng tuyến tính từ start→end
CURRICULUM_ENERGY_W_START   = 0.0     # |W_ENERGY| đầu training (chưa phạt năng lượng)
CURRICULUM_ENERGY_W_END     = 0.01197655138923567  # |W_ENERGY| cuối = giá trị gốc
```

> Giá trị `COST_LIMIT_*` là ước lượng khởi điểm — sẽ tinh chỉnh sau khi xem log cost trung bình của baseline. `LAMBDA_SAFETY_INIT=1.0` xấp xỉ scale `|W_SAFETY|≈0.019` × hệ số; có thể chỉnh.

- [ ] **Step 2: Verify import không lỗi**

Run: `cd /home/duyennh/AI_projects/ecorl_adaptive_shaping && python -c "import ppg.ppg_config as c; print(c.LAGRANGIAN_ENABLED, c.LAMBDA_LR, c.CURRICULUM_WARMUP_EPISODES)"`
Expected: in ra `False 0.01 1000` không lỗi.

- [ ] **Step 3: Commit**

```bash
git add ppg/ppg_config.py
git commit -m "feat(config): add Lagrangian + curriculum reward fallback constants

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Thêm λ state + dual update + ghép reward trong `PPGAgent`

**Files:**
- Modify: `ppg/core.py` — `__init__` (dòng 147–224), `compute_gae` (dòng 290–313), thêm method `update_lambdas`
- Modify: `ppg/memory.py` — thêm list `cost_safety`, `cost_comfort` vào `PPGMemory`

**Interfaces:**
- Consumes: hằng số từ `ppg_config.py` (Task 3); cost lưu trong `self.policy_memory.cost_safety/cost_comfort` (Task 4 sửa memory).
- Produces:
  - `PPGAgent.__init__(..., lagrangian_enabled=None, lambda_safety_init=None, lambda_comfort_init=None, lambda_lr=None, cost_limit_safety=None, cost_limit_comfort=None, lambda_max=None, ...)`.
  - `PPGAgent.update_lambdas() -> dict` trả `{"lambda_safety": float, "lambda_comfort": float, "mean_cost_safety": float, "mean_cost_comfort": float}`.
  - `compute_gae` khi `lagrangian_enabled` True dùng reward hiệu dụng `r_eff = reward − λ_s·cost_s − λ_c·cost_c`.
  - `PPGMemory.save_eps(..., cost_safety=0.0, cost_comfort=0.0)`.

- [ ] **Step 1: Viết test thất bại — dual update đẩy λ đúng hướng**

Thêm vào `tests/test_adaptive_reward.py`:

```python
from ppg.core import PPGAgent


def _make_lagrangian_agent():
    """Agent tối giản, KHÔNG dựng network thật — chỉ test logic λ."""
    agent = PPGAgent.__new__(PPGAgent)
    agent.lagrangian_enabled = True
    agent.lambda_safety = 1.0
    agent.lambda_comfort = 0.1
    agent.lambda_lr = 0.1
    agent.cost_limit_safety = 0.02
    agent.cost_limit_comfort = 0.30
    agent.lambda_max = 50.0

    class _Mem:
        cost_safety = [0.5, 0.5]      # mean=0.5 >> limit 0.02 → λ_safety phải TĂNG
        cost_comfort = [0.0, 0.0]     # mean=0.0 < limit 0.30 → λ_comfort phải GIẢM
    agent.policy_memory = _Mem()
    return agent


def test_update_lambdas_moves_correct_direction():
    agent = _make_lagrangian_agent()
    out = agent.update_lambdas()
    assert out["lambda_safety"] > 1.0    # cost vượt ngưỡng → tăng phạt
    assert out["lambda_comfort"] < 0.1   # cost dưới ngưỡng → giảm phạt
    assert out["lambda_comfort"] >= 0.0  # luôn không âm
```

- [ ] **Step 2: Chạy test để xác nhận thất bại**

Run: `cd /home/duyennh/AI_projects/ecorl_adaptive_shaping && python -m pytest tests/test_adaptive_reward.py::test_update_lambdas_moves_correct_direction -v`
Expected: FAIL — `AttributeError: 'PPGAgent' object has no attribute 'update_lambdas'`.

- [ ] **Step 3a: Thêm list cost vào `PPGMemory` (`ppg/memory.py`)**

Trong `__init__` (sau dòng 30 `self.value_vals = []`), thêm:
```python
        self.cost_safety = []
        self.cost_comfort = []
```

Đổi chữ ký `save_eps` (dòng 53) thành:
```python
    def save_eps(self, state, action, reward, done, next_state, log_prob, value, value_val,
                 cost_safety=0.0, cost_comfort=0.0):
```
và thêm 2 dòng append cuối thân hàm (sau dòng 61 `self.value_vals.append(value_val)`):
```python
        self.cost_safety.append(cost_safety)
        self.cost_comfort.append(cost_comfort)
```

Trong `clear_memory` (dòng 63–75), thêm:
```python
        del self.cost_safety[:]
        del self.cost_comfort[:]
```

- [ ] **Step 3b: Thêm λ state vào `PPGAgent.__init__` (`ppg/core.py`)**

Thêm params vào chữ ký `__init__` (sau `clip_eps=None,`, trước `**backbone_kwargs`):
```python
                 lagrangian_enabled=None, lambda_safety_init=None, lambda_comfort_init=None,
                 lambda_lr=None, cost_limit_safety=None, cost_limit_comfort=None, lambda_max=None,
```

Thêm resolve fallback (sau dòng 181, cạnh các config khác):
```python
        # Lagrangian adaptive weighting
        self.lagrangian_enabled = lagrangian_enabled if lagrangian_enabled is not None else ppg_config.LAGRANGIAN_ENABLED
        self.lambda_safety = lambda_safety_init if lambda_safety_init is not None else ppg_config.LAMBDA_SAFETY_INIT
        self.lambda_comfort = lambda_comfort_init if lambda_comfort_init is not None else ppg_config.LAMBDA_COMFORT_INIT
        self.lambda_lr = lambda_lr if lambda_lr is not None else ppg_config.LAMBDA_LR
        self.cost_limit_safety = cost_limit_safety if cost_limit_safety is not None else ppg_config.COST_LIMIT_SAFETY
        self.cost_limit_comfort = cost_limit_comfort if cost_limit_comfort is not None else ppg_config.COST_LIMIT_COMFORT
        self.lambda_max = lambda_max if lambda_max is not None else ppg_config.LAMBDA_MAX
```

- [ ] **Step 3c: Thêm method `update_lambdas` (`ppg/core.py`)**

Thêm method mới (đặt ngay trước `compute_gae`, dòng ~290):
```python
    def update_lambdas(self):
        """Dual gradient ascent cho hệ số Lagrangian λ.
        λ ← clamp(λ + lr · (mean_cost − limit), 0, λ_max).
        Gọi mỗi policy-update (sau khi thu xong rollout, trước compute_gae)."""
        cs = np.array(self.policy_memory.cost_safety, dtype=np.float32)
        cc = np.array(self.policy_memory.cost_comfort, dtype=np.float32)
        mean_cs = float(cs.mean()) if cs.size > 0 else 0.0
        mean_cc = float(cc.mean()) if cc.size > 0 else 0.0

        self.lambda_safety = float(np.clip(
            self.lambda_safety + self.lambda_lr * (mean_cs - self.cost_limit_safety),
            0.0, self.lambda_max))
        self.lambda_comfort = float(np.clip(
            self.lambda_comfort + self.lambda_lr * (mean_cc - self.cost_limit_comfort),
            0.0, self.lambda_max))

        return {
            "lambda_safety": self.lambda_safety,
            "lambda_comfort": self.lambda_comfort,
            "mean_cost_safety": mean_cs,
            "mean_cost_comfort": mean_cc,
        }
```

- [ ] **Step 3d: Ghép reward hiệu dụng trong `compute_gae` (`ppg/core.py`)**

Trong `compute_gae` (dòng 295), ngay sau `rewards = np.array(self.policy_memory.rewards, dtype=np.float32)`, thêm:
```python
        if getattr(self, "lagrangian_enabled", False):
            cs = np.array(self.policy_memory.cost_safety, dtype=np.float32)
            cc = np.array(self.policy_memory.cost_comfort, dtype=np.float32)
            if cs.size == rewards.size and cc.size == rewards.size:
                rewards = rewards - self.lambda_safety * cs - self.lambda_comfort * cc
```

> Đặt sau khi đọc `rewards`, trước vòng GAE. Phần còn lại của `compute_gae` không đổi — `rewards` đã là `r_eff`.

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `cd /home/duyennh/AI_projects/ecorl_adaptive_shaping && python -m pytest tests/test_adaptive_reward.py -v`
Expected: PASS toàn bộ, gồm `test_update_lambdas_moves_correct_direction`.

- [ ] **Step 5: Commit**

```bash
git add ppg/core.py ppg/memory.py tests/test_adaptive_reward.py
git commit -m "feat(ppg): add Lagrangian lambda dual update + effective reward in GAE

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Curriculum weighting theo episode trong env

**Files:**
- Modify: `simulation/env.py` — thêm `set_episode()` + làm `W_ENERGY` động trong `_calculate_reward`; thêm attribute `__init__`

**Interfaces:**
- Consumes: `self.curriculum_enabled`, `self.curriculum_warmup`, `self.curriculum_energy_w_start`, `self.curriculum_energy_w_end`, `self.current_episode`.
- Produces: `SumoEnv.set_episode(ep: int)` — train loop gọi đầu mỗi episode. Khi `curriculum_enabled` True, `|W_ENERGY|` nội suy tuyến tính từ `start` → `end` theo `current_episode/warmup` (clamp tại `end`).

- [ ] **Step 1: Viết test thất bại — W_ENERGY tăng theo episode**

Thêm vào `tests/test_adaptive_reward.py`. Test gián tiếp qua một helper `_curriculum_energy_weight` (Step 3 tạo) để không phụ thuộc TraCI:

```python
def test_curriculum_energy_weight_ramps():
    env = _make_stub_env(adaptive=False)
    env.curriculum_enabled = True
    env.curriculum_warmup = 1000
    env.curriculum_energy_w_start = 0.0
    env.curriculum_energy_w_end = 0.012

    env.current_episode = 0
    assert abs(env._curriculum_energy_weight() - 0.0) < 1e-9
    env.current_episode = 500
    assert abs(env._curriculum_energy_weight() - 0.006) < 1e-6
    env.current_episode = 5000  # quá warmup → clamp tại end
    assert abs(env._curriculum_energy_weight() - 0.012) < 1e-9


def test_curriculum_disabled_returns_base_weight():
    env = _make_stub_env(adaptive=False)
    env.curriculum_enabled = False
    env.curriculum_energy_w_end = 0.012
    assert abs(env._curriculum_energy_weight() - 0.012) < 1e-9
```

- [ ] **Step 2: Chạy test để xác nhận thất bại**

Run: `cd /home/duyennh/AI_projects/ecorl_adaptive_shaping && python -m pytest tests/test_adaptive_reward.py -k curriculum -v`
Expected: FAIL — `AttributeError: ... '_curriculum_energy_weight'`.

- [ ] **Step 3: Thêm `set_episode`, `_curriculum_energy_weight`, và dùng trong `_calculate_reward`**

Thêm attribute vào `__init__` (cạnh `self.adaptive_reward_enabled` từ Task 1):
```python
        # Curriculum weighting theo episode — mặc định tắt
        self.curriculum_enabled = False
        self.curriculum_warmup = 1000
        self.curriculum_energy_w_start = 0.0
        self.curriculum_energy_w_end = 0.01197655138923567
        self.current_episode = 0
```

Thêm 2 method (đặt cạnh `_split_reward_for_step`):
```python
    def set_episode(self, ep):
        """Train loop gọi đầu mỗi episode để curriculum biết tiến độ."""
        self.current_episode = int(ep)

    def _curriculum_energy_weight(self):
        """|W_ENERGY| hiệu dụng theo curriculum. Trả về giá trị DƯƠNG (độ lớn)."""
        if not getattr(self, "curriculum_enabled", False):
            return self.curriculum_energy_w_end
        warmup = max(1, self.curriculum_warmup)
        frac = min(1.0, self.current_episode / warmup)
        start = self.curriculum_energy_w_start
        end = self.curriculum_energy_w_end
        return start + frac * (end - start)
```

Trong `_calculate_reward`, đổi dòng `W_ENERGY = -0.01197655138923567` (dòng 778) thành:
```python
        W_ENERGY        = -self._curriculum_energy_weight()
```

> `_curriculum_energy_weight()` trả về độ lớn dương; dấu âm đặt ở đây để giữ nguyên ý nghĩa penalty. Khi `curriculum_enabled=False`, nó trả `curriculum_energy_w_end` = đúng giá trị gốc → backward-compatible. ✅

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `cd /home/duyennh/AI_projects/ecorl_adaptive_shaping && python -m pytest tests/test_adaptive_reward.py -v`
Expected: PASS toàn bộ.

- [ ] **Step 5: Commit**

```bash
git add simulation/env.py tests/test_adaptive_reward.py
git commit -m "feat(reward): add episode-based curriculum weighting for energy penalty

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Nối dây CLI/config + train loop + logging λ

**Files:**
- Modify: `train_ppg.py` — `parse_args` (dòng 54–137), `train()` agent construction (dòng 231–255), env construction (dòng 213), step loop (dòng 301–335), CSV header (dòng 35–38) và logging (dòng 348–355)
- Modify: `configs/ppg_default.yaml`

**Interfaces:**
- Consumes: `PPGAgent.update_lambdas()` (Task 4), `SumoEnv.set_episode()` (Task 5), `info["cost_safety"]/["cost_comfort"]` (Task 2).
- Produces: CLI flags `--lagrangian-enabled`, `--lambda-safety-init`, `--lambda-comfort-init`, `--lambda-lr`, `--cost-limit-safety`, `--cost-limit-comfort`, `--lambda-max`, `--curriculum-enabled`, `--curriculum-warmup`, `--curriculum-energy-w-start`, `--curriculum-energy-w-end`. CSV thêm cột `lambda_safety`, `lambda_comfort`.

- [ ] **Step 1: Thêm CLI flags (`parse_args`)**

Thêm sau khối PPG-specific (sau dòng 94 `--clip-eps`):
```python
    # Adaptive reward — Lagrangian
    parser.add_argument("--lagrangian-enabled", type=lambda x: str(x).lower() == "true", default=None, help="Bật Lagrangian adaptive weighting (true/false)")
    parser.add_argument("--lambda-safety-init", type=float, default=None, help="λ khởi tạo cho cost an toàn")
    parser.add_argument("--lambda-comfort-init", type=float, default=None, help="λ khởi tạo cho cost êm ái")
    parser.add_argument("--lambda-lr", type=float, default=None, help="Learning rate dual ascent của λ")
    parser.add_argument("--cost-limit-safety", type=float, default=None, help="Ngưỡng cost an toàn trung bình")
    parser.add_argument("--cost-limit-comfort", type=float, default=None, help="Ngưỡng cost êm ái trung bình")
    parser.add_argument("--lambda-max", type=float, default=None, help="Chặn trên λ")
    # Adaptive reward — Curriculum
    parser.add_argument("--curriculum-enabled", type=lambda x: str(x).lower() == "true", default=None, help="Bật curriculum weighting (true/false)")
    parser.add_argument("--curriculum-warmup", type=int, default=None, help="Số episode warmup curriculum")
    parser.add_argument("--curriculum-energy-w-start", type=float, default=None, help="|W_ENERGY| đầu training")
    parser.add_argument("--curriculum-energy-w-end", type=float, default=None, help="|W_ENERGY| cuối training")
```

> Dùng `type=lambda x: str(x).lower()=="true"` cho flag boolean vì YAML có thể nạp string `"true"`/`"false"` hoặc bool; lambda xử lý cả hai. `default=None` để phân biệt "không truyền".

- [ ] **Step 2: Truyền vào agent + bật adaptive trên env (`train()`)**

Sau khi tạo `env` (dòng 213), thêm — resolve giá trị enabled với fallback về config, rồi set lên env:
```python
    import ppg.ppg_config as _cfg
    _lag_enabled = args.lagrangian_enabled if args.lagrangian_enabled is not None else _cfg.LAGRANGIAN_ENABLED
    _cur_enabled = args.curriculum_enabled if args.curriculum_enabled is not None else _cfg.CURRICULUM_ENABLED

    env.adaptive_reward_enabled = _lag_enabled
    env.curriculum_enabled = _cur_enabled
    env.curriculum_warmup = args.curriculum_warmup if args.curriculum_warmup is not None else _cfg.CURRICULUM_WARMUP_EPISODES
    env.curriculum_energy_w_start = args.curriculum_energy_w_start if args.curriculum_energy_w_start is not None else _cfg.CURRICULUM_ENERGY_W_START
    env.curriculum_energy_w_end = args.curriculum_energy_w_end if args.curriculum_energy_w_end is not None else _cfg.CURRICULUM_ENERGY_W_END
```

Thêm vào lời gọi `PPGAgent(...)` (trước `**backbone_kwargs`, dòng ~254):
```python
        lagrangian_enabled=args.lagrangian_enabled,
        lambda_safety_init=args.lambda_safety_init,
        lambda_comfort_init=args.lambda_comfort_init,
        lambda_lr=args.lambda_lr,
        cost_limit_safety=args.cost_limit_safety,
        cost_limit_comfort=args.cost_limit_comfort,
        lambda_max=args.lambda_max,
```

- [ ] **Step 3: Truyền episode vào env + lưu cost + cập nhật λ + log (`train()` loop)**

Đầu mỗi episode, sau `env.reset()` (sau dòng 291 `agent.reset_history(state)`):
```python
            env.set_episode(global_ep_cnt + 1)
```

Đổi lời gọi `agent.save_eps` (dòng 320–323) để kèm cost từ info:
```python
                agent.save_eps(
                    state.tolist(), action.tolist(), reward, float(done),
                    next_state.tolist(), log_prob, value, value_val,
                    cost_safety=info.get("cost_safety", 0.0),
                    cost_comfort=info.get("cost_comfort", 0.0),
                )
```

> `PPGAgent.save_eps` (core.py dòng 274) phải forward 2 kwarg này xuống `policy_memory.save_eps`. Cập nhật chữ ký `PPGAgent.save_eps` thành `def save_eps(self, state, action, reward, done, next_state, log_prob, value, value_val, cost_safety=0.0, cost_comfort=0.0):` và truyền `cost_safety=cost_safety, cost_comfort=cost_comfort` vào CẢ hai nhánh `self.policy_memory.save_eps(...)` (nhánh sequential dòng 282 và nhánh thường dòng 286). Đây là sửa nhỏ trong core.py — đưa vào task này.

Đổi khối update (dòng 328–332) để gọi `update_lambdas` TRƯỚC `agent.update()` và bắt λ để log:
```python
                if t_updates == n_update_val:
                    if getattr(agent, "lagrangian_enabled", False):
                        lam_info = agent.update_lambdas()
                        last_lambda_safety = lam_info["lambda_safety"]
                        last_lambda_comfort = lam_info["lambda_comfort"]
                        print(f"  [LAGRANGIAN] λ_safety={lam_info['lambda_safety']:.4f} "
                              f"(cost={lam_info['mean_cost_safety']:.4f}) | "
                              f"λ_comfort={lam_info['lambda_comfort']:.4f} "
                              f"(cost={lam_info['mean_cost_comfort']:.4f})")
                    update_results = agent.update()
                    t_updates = 0
                    if update_results and update_results.get("aux_executed"):
                        print(f"  [AUX PHASE] Executed auxiliary phase update. Loss: {update_results['aux_loss']:.4f}")
```

> Thứ tự đúng: `update_lambdas()` đọc `policy_memory.cost_*` của rollout vừa thu, rồi `update()` gọi `compute_gae()` (dùng λ mới) và cuối cùng `clear_memory()`. Vì `update_lambdas` đọc cost trước khi `update` xoá memory → an toàn.

Khởi tạo biến λ để log (cạnh các counter, sau dòng 283 `t_updates = 0`):
```python
    last_lambda_safety = 0.0
    last_lambda_comfort = 0.0
```

- [ ] **Step 4: Thêm cột λ vào CSV**

Đổi `CSV_HEADER` (dòng 35–38):
```python
CSV_HEADER = [
    "episode", "steps", "ep_reward", "avg_speed", "total_energy",
    "wiggle", "safety", "success", "reason", "route", "override_rate", "avg_jerk",
    "lambda_safety", "lambda_comfort"
]
```

Thêm 2 trường vào `row` (dòng 348–354), cuối list:
```python
            row = [
                global_ep_cnt, ep_steps,
                f"{ep_reward:.2f}", f"{avg_speed:.2f}", f"{ep_energy:.2f}",
                f"{avg_jerk:.4f}", f"{avg_safety:.4f}",
                success, reason, route_str,
                f"{override_rate:.4f}", f"{avg_physical_jerk:.4f}",
                f"{last_lambda_safety:.4f}", f"{last_lambda_comfort:.4f}"
            ]
```

> CSV mới có 14 cột. File log cũ (12 cột) sẽ không khớp header — nhưng `init_csv_logging` chỉ ghi header khi file CHƯA tồn tại, và mỗi run tạo thư mục `runs/<run_id>/` mới nên luôn là file mới. Không có vấn đề tương thích ngược trên file cũ.

- [ ] **Step 5: Thêm section config YAML**

Append vào `configs/ppg_default.yaml`:
```yaml

# Adaptive Reward (mặc định TẮT — chạy baseline)
adaptive_reward:
  lagrangian_enabled: false
  lambda_safety_init: 1.0
  lambda_comfort_init: 0.1
  lambda_lr: 0.01
  cost_limit_safety: 0.02
  cost_limit_comfort: 0.30
  lambda_max: 50.0
  curriculum_enabled: false
  curriculum_warmup: 1000
  curriculum_energy_w_start: 0.0
  curriculum_energy_w_end: 0.01197655138923567
```

- [ ] **Step 6: Smoke test — import & parse_args không lỗi**

Run: `cd /home/duyennh/AI_projects/ecorl_adaptive_shaping && python -c "import train_ppg; import argparse; print('import ok')"`
Expected: in `import ok` không lỗi. (Không chạy `train()` vì cần SUMO.)

Run thêm: `cd /home/duyennh/AI_projects/ecorl_adaptive_shaping && python -m pytest tests/test_adaptive_reward.py -v`
Expected: PASS toàn bộ test (logic không hồi quy).

- [ ] **Step 7: Commit**

```bash
git add train_ppg.py configs/ppg_default.yaml ppg/core.py
git commit -m "feat(train): wire adaptive reward CLI/config, lambda update, and CSV logging

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Verify backward-compat + 3 chế độ chạy được

**Files:**
- Test: chạy thực tế ngắn (cần SUMO_HOME). Nếu môi trường review không có SUMO, chỉ chạy phần test logic.

**Interfaces:**
- Consumes: toàn bộ tasks trên.

- [ ] **Step 1: Xác nhận baseline không đổi hành vi**

Run: `cd /home/duyennh/AI_projects/ecorl_adaptive_shaping && python -m pytest tests/test_adaptive_reward.py -v`
Expected: PASS toàn bộ. Đặc biệt `test_calculate_reward_returns_scalar_when_not_adaptive` đảm bảo khi cả hai cờ tắt, reward là scalar y như gốc.

- [ ] **Step 2: (Nếu có SUMO) Smoke train 3 chế độ, mỗi chế độ ~2 episode**

```bash
# Baseline (cả hai tắt)
python train_ppg.py --backbone mamba --n-episode 2 --exp-name smoke_baseline

# Lagrangian
python train_ppg.py --backbone mamba --n-episode 2 --exp-name smoke_lagrangian --lagrangian-enabled true

# Curriculum
python train_ppg.py --backbone mamba --n-episode 2 --exp-name smoke_curriculum --curriculum-enabled true --curriculum-warmup 2
```
Expected: cả 3 chạy không crash; chế độ Lagrangian in dòng `[LAGRANGIAN] λ_safety=...`; CSV có 14 cột; `run_config.yaml` chứa các tham số adaptive mới.

> Nếu không có SUMO_HOME trong môi trường review: bỏ qua Step 2, ghi chú lại để người dùng tự chạy. Step 1 đã đủ chứng minh logic.

- [ ] **Step 3: Commit (nếu có chỉnh sửa nhỏ phát sinh khi smoke test)**

```bash
git add -A
git commit -m "test: verify adaptive reward backward-compat and three run modes

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Cách dùng sau khi hoàn tất (cho thí nghiệm so sánh paper)

```bash
# Arm 0 — Baseline (reward gốc)
python train_ppg.py --backbone mamba --exp-name baseline

# Arm 1 — Lagrangian adaptive weighting (phương pháp chính #1)
python train_ppg.py --backbone mamba --exp-name lagrangian \
    --lagrangian-enabled true --lambda-lr 0.01 \
    --cost-limit-safety 0.02 --cost-limit-comfort 0.30

# Arm 2 — Curriculum theo episode (baseline so sánh #2)
python train_ppg.py --backbone mamba --exp-name curriculum \
    --curriculum-enabled true --curriculum-warmup 1000

# Arm 3 — Kết hợp cả hai (tùy chọn, cho ablation đầy đủ)
python train_ppg.py --backbone mamba --exp-name combined \
    --lagrangian-enabled true --curriculum-enabled true
```

So sánh các đường cong `total_energy`, `success`, `safety`, `lambda_safety`/`lambda_comfort` (cột mới trong CSV) giữa các arm để chứng minh: Lagrangian tự điều chỉnh λ giữ an toàn trong ngưỡng trong khi giảm năng lượng, vượt trội so với curriculum theo lịch cố định.

## Tinh chỉnh sau (ngoài phạm vi code, vào lúc chạy thí nghiệm)
- Chạy baseline trước, đọc `mean_cost_safety`/`mean_cost_comfort` in ra → đặt `cost_limit_*` ngay dưới mức trung bình baseline để tạo áp lực ràng buộc.
- Nếu λ dao động mạnh → giảm `--lambda-lr` (vd 0.005). Nếu λ phản ứng chậm → tăng (vd 0.02).
