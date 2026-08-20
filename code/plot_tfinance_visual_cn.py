import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. 读取真实实验数据
# =========================
csv_path = r".\visual_logs\tfinance_strict_seed42_visual.csv"
df = pd.read_csv(csv_path)

print("读取文件：", os.path.abspath(csv_path))
print("数据行数：", len(df))
print("Epoch 范围：", df["epoch"].min(), "~", df["epoch"].max())

# =========================
# 2. 中文字体设置
# =========================
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 3. 创建 2×2 四联图
# =========================
fig, axes = plt.subplots(
    2,
    2,
    figsize=(16, 10),
)

ax1, ax2, ax3, ax4 = axes.flatten()

epoch = df["epoch"]

# -------------------------------------------------
# (a) 动态风险分组比例
# -------------------------------------------------
ax1.plot(epoch, df["low_ratio"], label="低风险组")
ax1.plot(epoch, df["unclear_ratio"], label="不确定组")
ax1.plot(epoch, df["high_ratio"], label="高风险组")

ax1.set_title("(a) 动态风险分组比例变化")
ax1.set_xlabel("训练轮次（Epoch）")
ax1.set_ylabel("节点比例")
ax1.grid(True, linestyle="--", alpha=0.35)
ax1.legend()

# -------------------------------------------------
# (b) 动态分组状态变化率
# epoch 0 的 group_flip_ratio 是 NaN，直接跳过
# -------------------------------------------------
flip_df = df.dropna(subset=["group_flip_ratio"])

ax2.plot(
    flip_df["epoch"],
    flip_df["group_flip_ratio"],
)

ax2.set_title("(b) 动态分组状态变化率")
ax2.set_xlabel("训练轮次（Epoch）")
ax2.set_ylabel("分组变化率")
ax2.grid(True, linestyle="--", alpha=0.35)

# -------------------------------------------------
# (c) 注意力权重演化
# -------------------------------------------------
ax3.plot(epoch, df["attn_all"], label="整体")
ax3.plot(epoch, df["attn_low"], label="低风险组")
ax3.plot(epoch, df["attn_high"], label="高风险组")

ax3.set_title("(c) 注意力权重演化")
ax3.set_xlabel("训练轮次（Epoch）")
ax3.set_ylabel("注意力权重")
ax3.grid(True, linestyle="--", alpha=0.35)
ax3.legend()

# -------------------------------------------------
# (d) 模型性能演化
# AUC + APS 放在双纵轴
# -------------------------------------------------
ax4.plot(
    epoch,
    df["trn_auc"],
    label="训练集 AUC",
)
ax4.plot(
    epoch,
    df["val_auc"],
    linestyle="--",
    label="验证集 AUC",
)

ax4.set_title("(d) 模型性能演化（AUC 与 APS）")
ax4.set_xlabel("训练轮次（Epoch）")
ax4.set_ylabel("AUC")

ax4_right = ax4.twinx()

ax4_right.plot(
    epoch,
    df["trn_aps"],
    label="训练集 APS",
)
ax4_right.plot(
    epoch,
    df["val_aps"],
    linestyle="--",
    label="验证集 APS",
)

ax4_right.set_ylabel("APS")

# 合并左右坐标轴图例
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_right.get_legend_handles_labels()

ax4.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="lower right",
)

ax4.grid(True, linestyle="--", alpha=0.35)

# =========================
# 4. 总标题
# =========================
fig.suptitle(
    "T-Finance（strict）数据集训练动态与机制演化（随机种子 = 42）",
    fontsize=18,
    y=0.98,
)

fig.tight_layout(rect=[0, 0.04, 1, 0.95])

# =========================
# 5. 保存高清图片
# =========================
output_dir = r".\visual_logs"
os.makedirs(output_dir, exist_ok=True)

png_path = os.path.join(
    output_dir,
    "tfinance_strict_seed42_四联机制图_中文版.png",
)

pdf_path = os.path.join(
    output_dir,
    "tfinance_strict_seed42_四联机制图_中文版.pdf",
)

plt.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
)

plt.savefig(
    pdf_path,
    bbox_inches="tight",
)

print("\n中文版四联图已保存：")
print(os.path.abspath(png_path))
print(os.path.abspath(pdf_path))

plt.show()
