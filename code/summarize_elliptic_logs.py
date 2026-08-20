import re
from pathlib import Path

import pandas as pd


LOG_DIR = Path(".")

EXPERIMENT_PATTERNS = [
    ("strict_franad_dtbe_full", r"^elliptic_dtbe_strict_seed(\d+)_50ep\.log$"),
    ("strict_franad_no_dtbe", r"^elliptic_franad_no_dtbe_strict_seed(\d+)_50ep\.log$"),
    ("strict_franad_no_group", r"^elliptic_franad_no_group_strict_seed(\d+)_50ep\.log$"),
    ("strict_franad_no_risk", r"^elliptic_franad_no_risk_strict_seed(\d+)_50ep\.log$"),
    ("strict_graphsage_dtbe", r"^elliptic_graphsage_dtbe_strict_seed(\d+)_50ep\.log$"),
    ("strict_graphsage", r"^elliptic_graphsage_strict_seed(\d+)_50ep\.log$"),
    ("strict_franad_woe", r"^elliptic_woe_strict_seed(\d+)_50ep\.log$"),
    ("strict_franad_iv", r"^elliptic_iv_strict_seed(\d+)_50ep\.log$"),
    ("old_dtbe", r"^elliptic_dtbe_seed(\d+)_50ep\.log$"),
    ("old_woe", r"^elliptic_woe_seed(\d+)_50ep\.log$"),
    ("old_iv", r"^elliptic_iv_seed(\d+)_50ep\.log$"),
]

METRICS = {
    "AP": "final_tst/aps",
    "AUC": "final_tst/auc",
    "Macro_F1": "final_tst/mf1",
    "G_Mean": "final_tst/gme",
    "Recall": "final_tst/rec",
    "Precision": "final_tst/pre",
}

def read_log_text(path: Path) -> str:
    """
    自动识别 Windows PowerShell Tee-Object 生成的 UTF-16 日志，
    同时兼容 UTF-8 日志。
    """
    data = path.read_bytes()

    # UTF-16 日志通常带 BOM，或包含大量空字节
    if (
        data.startswith(b"\xff\xfe")
        or data.startswith(b"\xfe\xff")
        or data.count(b"\x00") > len(data) * 0.1
    ):
        return data.decode("utf-16", errors="ignore")

    return data.decode("utf-8-sig", errors="ignore")

def identify_experiment(filename: str):
    for experiment, pattern in EXPERIMENT_PATTERNS:
        match = re.match(pattern, filename)
        if match:
            return experiment, int(match.group(1))
    return None, None


def extract_last_metric(text: str, metric_key: str):
    # 同时兼容 WandB summary 和最后输出的字典格式
    pattern = re.compile(
        re.escape(metric_key) + r"[^0-9\-]*([0-9]+(?:\.[0-9]+)?)"
    )
    values = pattern.findall(text)

    if not values:
        return None
    return float(values[-1])


rows = []

for log_file in sorted(LOG_DIR.glob("elliptic*seed*_50ep.log")):
    experiment, seed = identify_experiment(log_file.name)

    if experiment is None:
        print(f"跳过未识别日志：{log_file.name}")
        continue

    text = read_log_text(log_file)

    row = {
        "Experiment": experiment,
        "Seed": seed,
        "LogFile": log_file.name,
    }

    for output_name, metric_key in METRICS.items():
        row[output_name] = extract_last_metric(text, metric_key)

    rows.append(row)


detail_df = pd.DataFrame(rows).sort_values(
    ["Experiment", "Seed"]
)

missing = detail_df[
    detail_df[list(METRICS.keys())].isna().any(axis=1)
]

if not missing.empty:
    print("\n警告：以下日志存在缺失指标：")
    print(missing[["Experiment", "Seed", "LogFile"]].to_string(index=False))
else:
    print("\n全部日志均成功提取六项测试指标。")


summary_df = (
    detail_df
    .groupby("Experiment")
    .agg(
        Runs=("Seed", "count"),
        AP_Mean=("AP", "mean"),
        AP_SD=("AP", "std"),
        AUC_Mean=("AUC", "mean"),
        AUC_SD=("AUC", "std"),
        Macro_F1_Mean=("Macro_F1", "mean"),
        Macro_F1_SD=("Macro_F1", "std"),
        G_Mean_Mean=("G_Mean", "mean"),
        G_Mean_SD=("G_Mean", "std"),
        Recall_Mean=("Recall", "mean"),
        Recall_SD=("Recall", "std"),
        Precision_Mean=("Precision", "mean"),
        Precision_SD=("Precision", "std"),
    )
    .reset_index()
)

detail_df.to_csv(
    "elliptic_all_runs_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)

summary_df.to_csv(
    "elliptic_experiment_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n逐次实验结果：")
print(detail_df.to_string(index=False))

print("\n五次均值与样本标准差：")
print(summary_df.round(5).to_string(index=False))

print("\n已生成：")
print("1. elliptic_all_runs_metrics.csv")
print("2. elliptic_experiment_summary.csv")
