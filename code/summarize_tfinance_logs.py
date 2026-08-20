from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


ROOT = Path(".")
SEEDS = [42, 43, 44, 45, 46]

# 按“更具体的文件名在前”定义，避免误匹配。
EXPERIMENTS = {
    "full_graph_franad_dtbe": "tfinance_strict_dtbe_seed{seed}.log",
    "full_graph_graphsage_dtbe": "tfinance_graphsage_dtbe_seed{seed}.log",
    "full_graph_graphsage_no_dtbe": "tfinance_graphsage_no_dtbe_seed{seed}.log",
    "full_graph_franad_no_dtbe": "tfinance_franad_no_dtbe_seed{seed}.log",
    "full_graph_no_risk": "tfinance_no_risk_seed{seed}.log",
    "full_graph_no_group": "tfinance_no_group_seed{seed}.log",
    "isolated_franad_dtbe": "tfinance_isolated_dtbe_seed{seed}.log",
    "isolated_graphsage_dtbe": "tfinance_isolated_graphsage_dtbe_seed{seed}.log",
    "full_graph_franad_woe": "tfinance_woe_strict_seed{seed}.log",
    "full_graph_franad_iv": "tfinance_iv_strict_seed{seed}.log",
}

METRICS = {
    "aps": "AP",
    "auc": "AUC",
    "mf1": "Macro-F1",
    "gme": "G-Mean",
    "rec": "Recall",
    "pre": "Precision",
}

# A - B，结果为正表示A更高。
COMPARISONS = [
    ("full_graph_franad_dtbe", "full_graph_graphsage_dtbe",
     "Full graph: FraNAD+DTBE vs GraphSAGE+DTBE"),
    ("full_graph_franad_dtbe", "full_graph_franad_no_dtbe",
     "Full graph: FraNAD+DTBE vs FraNAD without DTBE"),
    ("full_graph_graphsage_dtbe", "full_graph_graphsage_no_dtbe",
     "Full graph: GraphSAGE+DTBE vs GraphSAGE without DTBE"),
    ("full_graph_franad_dtbe", "full_graph_no_risk",
     "Full graph: complete model vs no risk layer"),
    ("full_graph_franad_dtbe", "full_graph_no_group",
     "Full graph: complete model vs no dynamic grouping"),
    ("isolated_franad_dtbe", "isolated_graphsage_dtbe",
     "Isolated graph: FraNAD+DTBE vs GraphSAGE+DTBE"),
    ("full_graph_franad_dtbe", "isolated_franad_dtbe",
     "FraNAD+DTBE: full graph vs isolated graph"),
    ("full_graph_graphsage_dtbe", "isolated_graphsage_dtbe",
     "GraphSAGE+DTBE: full graph vs isolated graph"),
    ("full_graph_franad_dtbe", "full_graph_franad_woe",
     "Full graph: DTBE vs WOE"),
    ("full_graph_franad_dtbe", "full_graph_franad_iv",
     "Full graph: DTBE vs IV"),
    ("full_graph_franad_iv", "full_graph_franad_woe",
     "Full graph: IV vs WOE"),
]


def read_log_text(log_path: Path) -> str:
    """Read PowerShell Tee-Object logs across common Windows encodings."""
    raw = log_path.read_bytes()

    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16", errors="ignore")
    elif raw.startswith(b"\xef\xbb\xbf"):
        text = raw.decode("utf-8-sig", errors="ignore")
    elif raw.count(b"\x00") > max(8, len(raw) // 20):
        # Windows PowerShell commonly writes UTF-16LE text.
        text = raw.decode("utf-16-le", errors="ignore")
    else:
        for encoding in ("utf-8", "gb18030", "cp936", "utf-16-le"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = raw.decode("utf-8", errors="ignore")

    # Remove NULs and ANSI terminal escape sequences.
    text = text.replace("\x00", "")
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    return text


def parse_final_test(log_path: Path) -> dict[str, float]:
    text = read_log_text(log_path)
    result: dict[str, float] = {}

    for short_name in METRICS:
        pattern = rf"""['"]final_tst/{re.escape(short_name)}['"]\s*:\s*['"]?([0-9eE+\-.]+)['"]?"""
        matches = re.findall(pattern, text)
        if not matches:
            raise ValueError(
                f"{log_path.name}: cannot find final_tst/{short_name}"
            )
        result[short_name] = float(matches[-1])

    return result


def collect_runs() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    failed: list[str] = []

    for experiment, template in EXPERIMENTS.items():
        for seed in SEEDS:
            path = ROOT / template.format(seed=seed)
            if not path.exists():
                missing.append(path.name)
                continue

            try:
                metrics = parse_final_test(path)
            except Exception as exc:
                failed.append(f"{path.name}: {exc}")
                continue

            row: dict[str, object] = {
                "experiment": experiment,
                "seed": seed,
                "log_file": path.name,
            }
            row.update(metrics)
            rows.append(row)

    if missing:
        print("\n[WARNING] Missing log files:")
        for name in missing:
            print(f"  - {name}")

    if failed:
        print("\n[WARNING] Failed to parse:")
        for item in failed:
            print(f"  - {item}")

    if not rows:
        raise RuntimeError("No valid T-Finance log was parsed.")

    return pd.DataFrame(rows).sort_values(
        ["experiment", "seed"]
    ).reset_index(drop=True)


def build_summary(runs: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []

    for experiment, group in runs.groupby("experiment", sort=False):
        row: dict[str, object] = {
            "experiment": experiment,
            "n_runs": len(group),
        }

        for short_name, display_name in METRICS.items():
            values = group[short_name].astype(float).to_numpy()
            row[f"{display_name}_mean_pct"] = values.mean() * 100
            row[f"{display_name}_std_pct"] = (
                values.std(ddof=1) * 100 if len(values) > 1 else np.nan
            )
            row[f"{display_name}_formatted"] = (
                f"{values.mean() * 100:.2f} ± "
                f"{values.std(ddof=1) * 100:.2f}"
                if len(values) > 1
                else f"{values.mean() * 100:.2f}"
            )

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def paired_tests(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    indexed = runs.set_index(["experiment", "seed"])

    for exp_a, exp_b, label in COMPARISONS:
        for metric, display_name in METRICS.items():
            available_seeds = [
                seed for seed in SEEDS
                if (exp_a, seed) in indexed.index
                and (exp_b, seed) in indexed.index
            ]

            if len(available_seeds) < 2:
                continue

            a = np.array(
                [indexed.loc[(exp_a, seed), metric] for seed in available_seeds],
                dtype=float,
            )
            b = np.array(
                [indexed.loc[(exp_b, seed), metric] for seed in available_seeds],
                dtype=float,
            )
            diff = a - b

            row: dict[str, object] = {
                "comparison": label,
                "experiment_a": exp_a,
                "experiment_b": exp_b,
                "metric": display_name,
                "n_pairs": len(diff),
                "mean_a_pct": a.mean() * 100,
                "mean_b_pct": b.mean() * 100,
                "mean_diff_percentage_points": diff.mean() * 100,
                "std_diff_percentage_points": (
                    diff.std(ddof=1) * 100 if len(diff) > 1 else np.nan
                ),
                "cohen_dz": (
                    diff.mean() / diff.std(ddof=1)
                    if len(diff) > 1 and not np.isclose(diff.std(ddof=1), 0)
                    else np.nan
                ),
            }

            if SCIPY_AVAILABLE:
                try:
                    t_res = stats.ttest_rel(a, b, nan_policy="omit")
                    row["paired_t_p"] = float(t_res.pvalue)
                except Exception:
                    row["paired_t_p"] = np.nan

                try:
                    if np.allclose(diff, 0):
                        row["wilcoxon_p"] = 1.0
                    else:
                        w_res = stats.wilcoxon(
                            a,
                            b,
                            zero_method="wilcox",
                            correction=False,
                            alternative="two-sided",
                            mode="auto",
                        )
                        row["wilcoxon_p"] = float(w_res.pvalue)
                except Exception:
                    row["wilcoxon_p"] = np.nan
            else:
                row["paired_t_p"] = np.nan
                row["wilcoxon_p"] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    runs = collect_runs()
    summary = build_summary(runs)
    tests = paired_tests(runs)

    runs.to_csv(
        "tfinance_all_runs_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        "tfinance_experiment_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    tests.to_csv(
        "tfinance_paired_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 90)
    print("T-Finance experiment summary (mean ± sample standard deviation, %)")
    print("=" * 90)

    columns = ["experiment", "n_runs"] + [
        f"{display}_formatted" for display in METRICS.values()
    ]
    print(summary[columns].to_string(index=False))

    print("\nGenerated files:")
    print("  tfinance_all_runs_metrics.csv")
    print("  tfinance_experiment_summary.csv")
    print("  tfinance_paired_tests.csv")

    if not SCIPY_AVAILABLE:
        print(
            "\n[WARNING] scipy is unavailable. "
            "Paired-test p-values were left blank."
        )

    incomplete = summary.loc[summary["n_runs"] != len(SEEDS), ["experiment", "n_runs"]]
    if not incomplete.empty:
        print("\n[WARNING] Experiments with fewer than 5 runs:")
        print(incomplete.to_string(index=False))


if __name__ == "__main__":
    main()
