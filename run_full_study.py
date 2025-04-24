#!/usr/bin/env python3
import yaml, itertools, subprocess, pandas as pd, os, csv
import argparse
from pathlib import Path

# — point to your patched experiment script —
SCRIPT = Path(__file__).parent / "time_series_llm_experiment3.py"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_file", required=True,
                   help="path to your study_config.yml")
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
    # remove any old file
    if os.path.exists(out_csv):
        os.remove(out_csv)

    # write one header row
    param_cols = [
        "dataset_csv","time_col","value_col","window",
        "quant_bins","horizon","few_shot","use_metadata",
        "model_name","max_rows","fallback_strategy","max_new_tokens"
    ]
    header = ["Model","MSE","MAE","MASE"] + param_cols + ["status"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    for idx, (q, h, f, m, model) in enumerate(grid, start=1):
        params = dict(
            dataset_csv=cfg["dataset_csv"],
            time_col=cfg["time_col"],
            value_col=cfg["value_col"],
            window=cfg["window"],
            quant_bins=q,
            horizon=h,
            few_shot=f,
            use_metadata=m,
            model_name=model,
            max_rows=cfg.get("max_rows"),
            fallback_strategy=cfg.get("fallback_strategy","last"),
            max_new_tokens=cfg.get("max_new_tokens", None),
        )

        print(f"[{idx:3d}/{total}] ⏳ {model} | bins={q} h={h} few={f} meta={m}")

        # build and run the experiment command
        cmd = ["python", str(SCRIPT)]
        for k, v in params.items():
            if v is True:
                cmd.append(f"--{k}")
            elif v is False or v is None:
                continue
            else:
                cmd += [f"--{k}", str(v)]
        tmp = "tmp_results.csv"
        cmd += ["--output_csv", tmp]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            df = pd.read_csv(tmp)
            os.remove(tmp)
            status = "success"
            print(f"[{idx:3d}/{total}] ✅ {model}")
        except Exception:
            print(f"[{idx:3d}/{total}] ⚠️ Skipped {model}")
            # create three “skipped” rows to match your usual metric output
            df = pd.DataFrame({
                "Model": ["GPT2-Baseline", "GPT2-PaP", "ARIMA"],
                "MSE":   [float("nan")] * 3,
                "MAE":   [float("nan")] * 3,
                "MASE":  [float("nan")] * 3,
            })
            status = "skipped"

        # tag on all sweep params + status
        for k, v in params.items():
            df[k] = v
        df["status"] = status

        # append them to the CSV
        df.to_csv(out_csv, mode="a", header=False, index=False)

    print(f"\n✅ Study complete — results in `{out_csv}`")

if __name__ == "__main__":
    main()
