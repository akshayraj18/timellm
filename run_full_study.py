#!/usr/bin/env python3
"""
run_full_study.py

1) Sweep experiments via time_series_llm_experiment4.py
2) Collect all metrics into full_study_*.csv
3) Collect all example forecasts into all_examples.csv
4) Produce one combined bar chart of metrics
"""

import yaml
import itertools
import subprocess
import pandas as pd
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# --- point this at your experiment script ---
SCRIPT = Path(__file__).parent / "time_series_llm_experiment4.py"

def run_one(params, idx, total, out_csv, metrics_first, examples_csv, examples_first):
    # display progress
    print(f"[{idx:3d}/{total}] ⏳ {params['model_name']} | "
          f"bins={params['quant_bins']} h={params['horizon']} "
          f"few={params['few_shot']} meta={params['use_metadata']}")
    # build command
    cmd = ["python", str(SCRIPT)]
    for k, v in params.items():
        if v is True:
            cmd.append(f"--{k}")
        elif v is False or v is None:
            continue
        else:
            cmd += [f"--{k}", str(v)]
    # write metrics to a temp CSV
    tmp = "tmp_results.csv"
    cmd += ["--output_csv", tmp]

    try:
        subprocess.run(cmd, check=True)
        metrics_df = pd.read_csv(tmp)
        os.remove(tmp)
    except Exception as e:
        print(f"[{idx:3d}/{total}] ⚠️ Skipped {params['model_name']}: {e}")
        return metrics_first, examples_first

    # tag sweep parameters onto metrics
    for k, v in params.items():
        metrics_df[k] = v

    # append metrics_df to the global metrics CSV
    metrics_df.to_csv(out_csv,
                      mode="w" if metrics_first else "a",
                      header=metrics_first,
                      index=False)
    print(f"[{idx:3d}/{total}] ✅ metrics")

    # now grab that run's example table and append into combined examples CSV
    safe = params["model_name"].replace("/", "_").replace(":", "_")
    ex_file = f"{safe}_example_table.csv"
    if os.path.exists(ex_file):
        ex_df = pd.read_csv(ex_file)
        # tag params onto example rows
        for k, v in params.items():
            ex_df[k] = v
        ex_df.to_csv(examples_csv,
                     mode="w" if examples_first else "a",
                     header=examples_first,
                     index=False)
        os.remove(ex_file)
        print(f"[{idx:3d}/{total}] ✅ examples")
        examples_first = False

    return False, examples_first  # metrics_first flips after first write

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_file", required=True,
                   help="path to your study_config YAML")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config_file))
    grid = list(itertools.product(
        cfg["quant_bins"],
        cfg["horizons"],
        cfg["few_shots"],
        cfg["use_metadata"],
        cfg["model_names"]
    ))
    total = len(grid)
    out_csv = cfg.get("output_csv", "full_study_results.csv")
    examples_csv = "all_examples.csv"

    # clear old outputs
    if os.path.exists(out_csv):
        os.remove(out_csv)
    if os.path.exists(examples_csv):
        os.remove(examples_csv)

    metrics_first = True
    examples_first = True

    for idx, (q, h, f, m, model) in enumerate(grid, start=1):
        params = {
            "dataset_csv":       cfg["dataset_csv"],
            "time_col":          cfg["time_col"],
            "value_col":         cfg["value_col"],
            "window":            cfg["window"],
            "quant_bins":        q,
            "horizon":           h,
            "few_shot":          f,
            "use_metadata":      m,
            "model_name":        model,
            "max_rows":          cfg.get("max_rows"),
            "fallback_strategy": cfg.get("fallback_strategy","mean"),
            "max_new_tokens":    cfg.get("max_new_tokens",None),
        }
        metrics_first, examples_first = run_one(
            params, idx, total, out_csv,
            metrics_first, examples_csv, examples_first
        )

    print(f"\n✅ Study complete — metrics in `{out_csv}`, examples in `{examples_csv}`")

    # --- Combined bar chart of metrics ---
    df = pd.read_csv(out_csv)
    ax = df.plot(
        x="model_name",              # or "Model" if you prefer
        y=["MSE","MAE","MASE"],
        kind="bar",
        figsize=(10,5)
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    bar_path = "combined_metrics_bar.png"
    plt.savefig(bar_path)
    print(f"Saved combined bar chart → {bar_path}")

if __name__ == "__main__":
    main()
