#!/usr/bin/env python3
"""
visualize_training.py — Vẽ và so sánh training log CSV của các thuật toán.

Đọc các file ``training_log.csv`` sinh ra bởi ``train_ppg.py`` rồi vẽ đường cong
học (learning curves) so sánh giữa các phương pháp, gộp nhiều seed thành
mean ± dải tin cậy (CI 95% hoặc min–max).

CSV được kỳ vọng có các cột (xem train_ppg.CSV_HEADER):
    episode, steps, ep_reward, avg_speed, total_energy, wiggle, safety,
    success, reason, route, override_rate, avg_jerk,
    lambda_safety, lambda_comfort, lambda_redlight   (3 cột λ tuỳ chọn)

────────────────────────────────────────────────────────────────────────────
CÁCH DÙNG

1) Tự động quét (đơn giản nhất) — gộp theo exp_name, gộp seed:
   python visualize_training.py
   python visualize_training.py --root . --out plots/

2) Chỉ định thủ công nhóm = glob (nhãn tuỳ ý), mỗi nhóm gồm ≥1 seed:
   python visualize_training.py \
       --runs "PPO=PPO_PPG/runs/ppo_*/training_log.csv" \
       --runs "PPG-Mamba=Mamba_PPG/runs/ppg_*/training_log.csv" \
       --runs "PPG-Lagrange=Mamba_PPG/runs/lagrangian_*/training_log.csv" \
       --out plots/

Tuỳ chọn:
   --smooth N      cửa sổ trung bình trượt (mặc định 20; 1 = tắt)
   --band ci|range dải quanh mean: CI 95% (mặc định) hoặc min–max giữa seed
   --metrics ...   danh sách cột muốn vẽ (mặc định bộ chuẩn cho journal)
   --max-episode E cắt trục x tại episode E
"""
import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # không cần X server
import matplotlib.pyplot as plt  # noqa: E402


# Bộ metric mặc định (nhãn hiển thị, "hướng tốt" để chú thích)
DEFAULT_METRICS = [
    ("ep_reward",     "Episode Reward",        "higher better"),
    ("total_energy",  "Total Energy (Wh)",     "lower better"),
    ("avg_speed",     "Avg Speed (m/s)",       "higher better"),
    ("safety",        "Safety Penalty",        "lower better"),
    ("avg_jerk",      "Avg Jerk (comfort)",    "lower better"),
    ("override_rate", "RSS Override Rate",     "lower better"),
]
LAMBDA_COLS = ["lambda_safety", "lambda_comfort", "lambda_redlight"]

RUNID_SEED_RE = re.compile(r"^(?P<name>.+)_seed(?P<seed>\d+)_\d{6,}")


# ─────────────────────────────────────────────────────────────────────────────
#  Khám phá & nạp dữ liệu
# ─────────────────────────────────────────────────────────────────────────────
def _parse_run_id(run_dir):
    """Tách (exp_name, seed) từ tên thư mục run `{name}_seed{N}_{timestamp}`."""
    base = os.path.basename(os.path.normpath(run_dir))
    m = RUNID_SEED_RE.match(base)
    if m:
        return m.group("name"), int(m.group("seed"))
    return base, None  # không match → coi cả tên là nhãn, seed unknown


def discover_runs(root):
    """Quét root tìm mọi training_log.csv, gộp theo exp_name.

    Trả về dict {label: [csv_path, ...]}.
    """
    pattern = os.path.join(root, "*_PPG", "runs", "*", "training_log.csv")
    groups = defaultdict(list)
    for csv_path in sorted(glob.glob(pattern)):
        run_dir = os.path.dirname(csv_path)
        name, _seed = _parse_run_id(run_dir)
        groups[name].append(csv_path)
    return dict(groups)


def parse_runs_arg(runs_args):
    """Chuyển danh sách 'LABEL=glob' thành {label: [csv_path,...]}."""
    groups = {}
    for spec in runs_args:
        if "=" not in spec:
            raise ValueError(f"--runs phải dạng LABEL=glob, nhận: {spec!r}")
        label, pattern = spec.split("=", 1)
        paths = sorted(glob.glob(pattern))
        if not paths:
            print(f"  [WARN] không tìm thấy file nào khớp: {pattern}")
        groups[label.strip()] = paths
    return groups


def load_csv(path):
    """Nạp 1 training_log.csv → DataFrame. Ép kiểu số ở các cột số."""
    df = pd.read_csv(path)
    for col in df.columns:
        if col in ("reason", "route"):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "success" in df.columns:
        df["success"] = df["success"].fillna(0).astype(float)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Tổng hợp nhiều seed
# ─────────────────────────────────────────────────────────────────────────────
def _smooth(series, window):
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1, center=False).mean()


def aggregate_metric(dfs, metric, smooth, max_episode):
    """Gộp nhiều seed cho 1 metric.

    Mỗi df là 1 seed; căn theo cột 'episode'. Trả về
    (episodes, mean, lo, hi, n_seeds) — lo/hi do hàm gọi quyết định (CI/range).
    Trả None nếu metric không tồn tại ở bất kỳ seed nào.
    """
    cols = []
    for df in dfs:
        if metric not in df.columns or "episode" not in df.columns:
            continue
        s = df[["episode", metric]].dropna()
        if max_episode is not None:
            s = s[s["episode"] <= max_episode]
        if s.empty:
            continue
        s = s.set_index("episode")[metric]
        s = _smooth(s, smooth)
        cols.append(s)
    if not cols:
        return None

    # Căn các seed theo episode (outer join → NaN ở chỗ seed thiếu)
    mat = pd.concat(cols, axis=1)
    mat = mat.sort_index()
    episodes = mat.index.values
    mean = mat.mean(axis=1).values
    std = mat.std(axis=1, ddof=0).values
    n = mat.notna().sum(axis=1).values
    return episodes, mean, std, n, mat


def band_bounds(mean, std, mat, mode):
    """Tính dải dưới/trên quanh mean."""
    if mode == "range":
        lo = np.nanmin(mat.values, axis=1)
        hi = np.nanmax(mat.values, axis=1)
    else:  # ci 95%
        n = np.clip(mat.notna().sum(axis=1).values, 1, None)
        sem = std / np.sqrt(n)
        lo = mean - 1.96 * sem
        hi = mean + 1.96 * sem
    return lo, hi


# ─────────────────────────────────────────────────────────────────────────────
#  Vẽ
# ─────────────────────────────────────────────────────────────────────────────
def plot_metric_comparison(groups_dfs, metric, label, hint, args, out_path):
    """Vẽ 1 metric, mỗi nhóm (phương pháp) 1 đường + dải."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for gi, (glabel, dfs) in enumerate(groups_dfs.items()):
        agg = aggregate_metric(dfs, metric, args.smooth, args.max_episode)
        if agg is None:
            continue
        episodes, mean, std, n, mat = agg
        lo, hi = band_bounds(mean, std, mat, args.band)
        color = f"C{gi}"
        n_seeds = int(np.nanmax(n)) if len(n) else 0
        ax.plot(episodes, mean, color=color, label=f"{glabel} (n={n_seeds})", lw=1.8)
        ax.fill_between(episodes, lo, hi, color=color, alpha=0.18, linewidth=0)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return False
    band_name = "95% CI" if args.band == "ci" else "min–max"
    ax.set_xlabel("Episode")
    ax.set_ylabel(label)
    ax.set_title(f"{label}  ({hint}; band = {band_name} over seeds)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def plot_success_rate(groups_dfs, args, out_path):
    """Bar chart tỉ lệ success trung bình (toàn run, gộp seed) cho từng nhóm."""
    labels, means, errs = [], [], []
    for glabel, dfs in groups_dfs.items():
        per_seed = []
        for df in dfs:
            if "success" not in df.columns:
                continue
            s = df["success"]
            if args.max_episode is not None and "episode" in df.columns:
                s = df.loc[df["episode"] <= args.max_episode, "success"]
            if len(s):
                per_seed.append(float(np.nanmean(s)))
        if not per_seed:
            continue
        labels.append(glabel)
        means.append(np.mean(per_seed) * 100.0)
        errs.append((np.std(per_seed) * 100.0) if len(per_seed) > 1 else 0.0)
    if not labels:
        return False
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(labels)), 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errs, capsize=5,
           color=[f"C{i}" for i in range(len(labels))], alpha=0.85)
    for i, m in enumerate(means):
        ax.text(i, m, f"{m:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Mean Success Rate (± std over seeds)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def plot_lambda_curves(groups_dfs, args, out_dir):
    """Vẽ tiến hoá λ (chỉ cho nhóm có cột λ khác 0 — tức arm Lagrangian)."""
    saved = []
    for col in LAMBDA_COLS:
        groups_with = {
            g: dfs for g, dfs in groups_dfs.items()
            if any(col in df.columns and df[col].abs().sum() > 0 for df in dfs)
        }
        if not groups_with:
            continue
        label = col.replace("lambda_", "λ_")
        out_path = os.path.join(out_dir, f"curve_{col}.png")
        ok = plot_metric_comparison(groups_with, col, label,
                                    "Lagrangian multiplier", args, out_path)
        if ok:
            saved.append(out_path)
    return saved


# ─────────────────────────────────────────────────────────────────────────────
#  Bảng tóm tắt
# ─────────────────────────────────────────────────────────────────────────────
def write_summary_table(groups_dfs, args, out_path):
    """Ghi CSV tóm tắt: với mỗi nhóm, giá trị cuối (mean ± std giữa seed) của
    các metric chính, lấy trung bình `tail` episode cuối."""
    tail = 50
    rows = []
    metric_keys = [m[0] for m in DEFAULT_METRICS] + ["success"]
    for glabel, dfs in groups_dfs.items():
        row = {"method": glabel, "n_seeds": len(dfs)}
        for mk in metric_keys:
            per_seed = []
            for df in dfs:
                if mk not in df.columns:
                    continue
                s = df[mk].dropna()
                if args.max_episode is not None and "episode" in df.columns:
                    s = df.loc[df["episode"] <= args.max_episode, mk].dropna()
                if len(s):
                    per_seed.append(float(s.tail(tail).mean()))
            if per_seed:
                row[f"{mk}_mean"] = np.mean(per_seed)
                row[f"{mk}_std"] = np.std(per_seed) if len(per_seed) > 1 else 0.0
        rows.append(row)
    if not rows:
        return False
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Vẽ & so sánh training log CSV của các thuật toán RL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--root", default=".",
                    help="Thư mục gốc để tự động quét *_PPG/runs/*/training_log.csv (mặc định '.')")
    ap.add_argument("--runs", action="append", default=[],
                    help="Chỉ định nhóm thủ công: 'LABEL=glob'. Lặp lại cho nhiều nhóm.")
    ap.add_argument("--out", default="plots",
                    help="Thư mục xuất ảnh (mặc định 'plots')")
    ap.add_argument("--smooth", type=int, default=20,
                    help="Cửa sổ trung bình trượt (mặc định 20; 1 = tắt)")
    ap.add_argument("--band", choices=["ci", "range"], default="ci",
                    help="Dải quanh mean: 'ci' 95%% (mặc định) hoặc 'range' min–max")
    ap.add_argument("--metrics", nargs="+", default=None,
                    help="Danh sách cột muốn vẽ (mặc định bộ chuẩn)")
    ap.add_argument("--max-episode", type=int, default=None,
                    help="Cắt trục x tại episode này")
    args = ap.parse_args()

    # 1. Xác định nhóm runs
    if args.runs:
        groups = parse_runs_arg(args.runs)
    else:
        groups = discover_runs(args.root)
        if not groups:
            print(f"[ERROR] Không tìm thấy training_log.csv nào dưới '{args.root}'.")
            print("  Gợi ý: chạy train trước, hoặc dùng --runs 'LABEL=glob'.")
            sys.exit(1)

    # 2. Nạp dataframe cho từng nhóm
    groups_dfs = {}
    for label, paths in groups.items():
        dfs = []
        for p in paths:
            try:
                dfs.append(load_csv(p))
            except Exception as e:  # noqa: BLE001
                print(f"  [WARN] lỗi đọc {p}: {e}")
        if dfs:
            groups_dfs[label] = dfs

    if not groups_dfs:
        print("[ERROR] Không nạp được dữ liệu nào.")
        sys.exit(1)

    print(f"Đã nạp {len(groups_dfs)} nhóm:")
    for label, dfs in groups_dfs.items():
        print(f"  - {label}: {len(dfs)} seed(s)")

    os.makedirs(args.out, exist_ok=True)

    # 3. Vẽ các metric chính
    if args.metrics:
        metric_specs = [(m, m.replace("_", " ").title(), "") for m in args.metrics]
    else:
        metric_specs = DEFAULT_METRICS

    written = []
    for metric, label, hint in metric_specs:
        out_path = os.path.join(args.out, f"curve_{metric}.png")
        if plot_metric_comparison(groups_dfs, metric, label, hint, args, out_path):
            written.append(out_path)

    # 4. Success rate bar
    sr_path = os.path.join(args.out, "success_rate.png")
    if plot_success_rate(groups_dfs, args, sr_path):
        written.append(sr_path)

    # 5. λ curves (chỉ arm Lagrangian)
    written += plot_lambda_curves(groups_dfs, args, args.out)

    # 6. Bảng tóm tắt
    summary_path = os.path.join(args.out, "summary_table.csv")
    if write_summary_table(groups_dfs, args, summary_path):
        written.append(summary_path)

    print(f"\nĐã xuất {len(written)} file vào '{args.out}/':")
    for w in written:
        print(f"  - {os.path.basename(w)}")


if __name__ == "__main__":
    main()
