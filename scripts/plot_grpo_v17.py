#!/usr/bin/env python3
"""GRPO v17 训练结果可视化 (2026-08-12)。

数据: 随训练推进的变化表 (step 区间 1-50, 每区间 160 rollout):
    avg_mixed_reward / resolved_rate / non_empty_patch_rate
产出: docs/plots/grpo_v17_training_progress.png (双轴主图)
      docs/plots/grpo_v17_metrics_panels.png  (1x3 分指标面板)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- 中文字体 ----
FONT_CANDIDATES = ["Hiragino Sans GB", "Arial Unicode MS", "Heiti SC", "PingFang SC"]
for name in font_manager.fontManager.ttflist:
    if name.name in FONT_CANDIDATES:
        plt.rcParams["font.sans-serif"] = [name.name, "Arial Unicode MS"]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11

# ---- 数据 (来自用户实验表) ----
STEPS = [5, 15, 25, 35, 45]          # 区间中点
STEP_LABELS = ["1–10", "11–20", "21–30", "31–40", "41–50"]
AVG_REWARD = [0.0864, 0.0728, 0.0938, 0.0983, 0.1273]
RESOLVED = [1.25, 0.0, 2.25, 4.75, 6.75]          # %
PATCH_RATE = [22.50, 21.25, 31.25, 29.00, 36.25]   # %
ROLLOUT_EACH = 160
TOTAL_STEPS = 50

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 调色板
C_REWARD = "#1f77b4"
C_RESOLVED = "#2ca02c"
C_PATCH = "#ff7f0e"
C_GRID = "#dcdcdc"


def _style_ax(ax):
    ax.grid(True, axis="y", color=C_GRID, linestyle="--", linewidth=0.6, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0)


# ================= 主图: 双轴 三指标 =================
fig, ax1 = plt.subplots(figsize=(10.5, 6.0))
fig.patch.set_facecolor("white")
ax1.set_facecolor("white")

# 左轴: 平均混合奖励 (面积渐变 + 折线)
ax1.plot(STEPS, AVG_REWARD, color=C_REWARD, linewidth=2.6, marker="o",
         markersize=8, markeredgecolor="white", markeredgewidth=1.5, zorder=5)
ax1.fill_between(STEPS, AVG_REWARD, 0.0, color=C_REWARD, alpha=0.10, zorder=1)
ax1.set_xlabel("GRPO 训练步区间", fontsize=12, labelpad=8)
ax1.set_ylabel("平均混合奖励 (F2P partial + P2P 惩罚)", fontsize=12, color=C_REWARD)
ax1.set_xticks(STEPS)
ax1.set_xticklabels(STEP_LABELS)
ax1.set_ylim(0.0, 0.16)
ax1.tick_params(axis="y", colors=C_REWARD)
_style_ax(ax1)

# 右轴: resolved rate + non-empty patch rate
ax2 = ax1.twinx()
ax2.plot(STEPS, RESOLVED, color=C_RESOLVED, linewidth=2.2, marker="s", markersize=6,
         markeredgecolor="white", markeredgewidth=1.2, zorder=4, label="resolved rate")
ax2.plot(STEPS, PATCH_RATE, color=C_PATCH, linewidth=2.2, marker="^", markersize=6,
         markeredgecolor="white", markeredgewidth=1.2, zorder=4, label="non-empty patch rate")
ax2.set_ylabel("比例 (%)", fontsize=12, color="#666666")
ax2.set_ylim(0, 45)
ax2.tick_params(axis="y", colors="#666666")
_style_ax(ax2)

# 数值标注
for x, y in zip(STEPS, AVG_REWARD):
    ax1.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 11),
                 ha="center", fontsize=9, color=C_REWARD, fontweight="bold")
for x, y in zip(STEPS, RESOLVED):
    ax2.annotate(f"{y:.2f}%", (x, y), textcoords="offset points", xytext=(-14, 8),
                 ha="center", fontsize=9, color=C_RESOLVED, fontweight="bold")
for x, y in zip(STEPS, PATCH_RATE):
    ax2.annotate(f"{y:.2f}%", (x, y), textcoords="offset points", xytext=(12, -10),
                 ha="center", fontsize=9, color=C_PATCH, fontweight="bold")

# 起点/终点趋势箭头 (reward)
ax1.annotate("", xy=(45, 0.1273), xytext=(5, 0.0864),
             arrowprops=dict(arrowstyle="->", color=C_REWARD, lw=1.4, alpha=0.5))

# 图例 + 标题
handles = [plt.Line2D([], [], color=C_REWARD, lw=2.6, marker="o", label="平均混合奖励"),
           plt.Line2D([], [], color=C_RESOLVED, lw=2.2, marker="s", label="resolved rate"),
           plt.Line2D([], [], color=C_PATCH, lw=2.2, marker="^", label="non-empty patch rate")]
ax2.legend(handles=handles, loc="center right", frameon=False, fontsize=10)
ax1.set_title("GRPO v17 训练进展 (Step 1–50, 每区间 160 rollout, 共 800)",
              fontsize=15, fontweight="bold", pad=16)

fig.tight_layout()
out1 = OUT_DIR / "grpo_v17_training_progress.png"
fig.savefig(out1, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ================= 分指标面板 (1x3) =================
fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4))
fig.patch.set_facecolor("white")
fig.suptitle("GRPO v17 关键指标逐区间变化 (Step 1–50)", fontsize=15, fontweight="bold", y=1.00)

panels = [
    (axes[0], AVG_REWARD, "平均混合奖励", C_REWARD, "{:.4f}"),
    (axes[1], RESOLVED, "resolved rate (%)", C_RESOLVED, "{:.2f}%"),
    (axes[2], PATCH_RATE, "non-empty patch rate (%)", C_PATCH, "{:.2f}%"),
]
for ax, data, label, color, fmt in panels:
    ax.bar(STEPS, data, color=color, alpha=0.75, width=7.0, edgecolor="white")
    ax.plot(STEPS, data, color=color, lw=1.8, marker="o", markersize=4, zorder=5)
    ax.set_xticks(STEPS)
    ax.set_xticklabels(STEP_LABELS)
    ax.set_title(label, fontsize=11.5, fontweight="bold", pad=8)
    for x, y in zip(STEPS, data):
        ax.annotate(fmt.format(y), (x, y), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=8.5, color=color, fontweight="bold")
    ax.grid(True, axis="y", color=C_GRID, linestyle="--", linewidth=0.6, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
out2 = OUT_DIR / "grpo_v17_metrics_panels.png"
fig.savefig(out2, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"saved: {out1}")
print(f"saved: {out2}")
