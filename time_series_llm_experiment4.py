#!/usr/bin/env python3
"""
Time-Series Forecasting Comparison: Baseline vs. Prompt-as-Prefix (PaP) on HF LLMs *and* OpenAI API

Usage:
    python time_series_llm_experiment4.py \
      --dataset_csv ./dataset/ETTh1.csv \
      --time_col date \
      --value_col OT \
      --window 12 \
      --quant_bins 50 \
      --max_rows 500 \
      --fallback_strategy mean \
      --horizon 1 \
      --max_new_tokens 2 \
      --few_shot 2 \
      --use_metadata \
      --model_name sshleifer/tiny-gpt2 \
      --output_csv tmp_results.csv

    # or for OpenAI:
    python time_series_llm_experiment4.py ... --model_name openai:gpt-3.5-turbo
"""

import argparse
import os
import re
import pandas as pd
import numpy as np
import torch

from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from transformers import AutoTokenizer, AutoModelForCausalLM

import openai
from openai import OpenAI  # new 1.x client

DEVICE = torch.device("cpu")
openai_client = OpenAI()  # instantiate once


def load_series(path, time_col, value_col):
    df = pd.read_csv(path, parse_dates=[time_col])
    df = df.sort_values(time_col)
    return df[time_col].values, df[value_col].values


def baseline_prompt(window_values, meta=""):
    vals = ", ".join(f"{v:.2f}" for v in window_values)
    if meta:
        return f"Time series ({meta}): {vals}. Predict the next value."
    return f"Time series: {vals}. Predict the next value."


def pap_prompt(window_values, freq_desc="hourly data", bins=None):
    vals = window_values.copy()
    if bins and bins > 0:
        mn, mx = vals.min(), vals.max()
        q = np.floor((vals - mn) / (mx - mn + 1e-8) * bins).astype(int)
        vals = q
    vals_str = ", ".join(str(int(v)) if bins else f"{v:.2f}" for v in vals)
    header = f"Given the historical {freq_desc}, forecast the next value:\nData:"
    return f"{header} {vals_str}. Next:"


def gpt2_predict(prompts, model, tokenizer, max_new_tokens, fallback_strategy):
    preds = []
    for prompt in prompts:
        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, padding=True, max_length=128
        ).to(DEVICE)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        text = tokenizer.decode(
            out[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()
        m = re.search(r"[-+]?[0-9]*\.?[0-9]+", text)
        if m:
            val = float(m.group())
        else:
            nums = [float(x) for x in re.findall(r"[-+]?[0-9]*\.?[0-9]+", prompt)]
            if fallback_strategy == "mean":
                val = float(np.mean(nums))
            elif fallback_strategy == "zero":
                val = 0.0
            else:
                val = nums[-1]
        preds.append(val)
    return np.array(preds, dtype=float)


def openai_predict(prompts, openai_model, max_new_tokens, fallback_strategy):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set OPENAI_API_KEY in your environment")
    preds = []
    for prompt in prompts:
        resp = openai_client.chat.completions.create(
            model=openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=0.0
        )
        text = resp.choices[0].message.content.strip()
        m = re.search(r"[-+]?[0-9]*\.?[0-9]+", text)
        if m:
            preds.append(float(m.group()))
        else:
            nums = [float(x) for x in re.findall(r"[-+]?[0-9]*\.?[0-9]+", prompt)]
            if fallback_strategy == "mean":
                preds.append(float(np.mean(nums)))
            elif fallback_strategy == "zero":
                preds.append(0.0)
            else:
                preds.append(nums[-1])
    return np.array(preds, dtype=float)


def arima_predict(train, test, order=(1,1,1)):
    res = ARIMA(train, order=order).fit()
    return res.forecast(steps=len(test))


def mase(y_true, y_pred, y_train):
    return mean_absolute_error(y_true, y_pred) / np.abs(np.diff(y_train)).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_csv",   required=True)
    p.add_argument("--time_col",      default="date")
    p.add_argument("--value_col",     default="value")
    p.add_argument("--window",        type=int, default=12)
    p.add_argument("--quant_bins",    type=int, default=10)
    p.add_argument("--max_rows",      type=int, default=None)
    p.add_argument("--fallback_strategy",
                   choices=["last","mean","zero"], default="last")
    p.add_argument("--horizon",       type=int, default=1)
    p.add_argument("--max_new_tokens",type=int, default=None)
    p.add_argument("--few_shot",      type=int, default=0)
    p.add_argument("--use_metadata",  action="store_true")
    p.add_argument("--model_name",    default="sshleifer/tiny-gpt2")
    p.add_argument("--output_csv",    default=None)
    args = p.parse_args()

    # load & truncate
    times, values = load_series(args.dataset_csv, args.time_col, args.value_col)
    if args.max_rows:
        times, values = times[:args.max_rows], values[:args.max_rows]

    # split
    split = int(len(values)*0.8)
    train_vals = values[:split]
    test_vals  = values[split-args.window:]
    test_times = times[split:]

    # few-shot context
    few_text = ""
    for i in range(args.few_shot):
        if i + args.window >= len(train_vals): break
        ex = train_vals[i:i+args.window]
        nxt = train_vals[i+args.window]
        few_text += ("Time series: " + ", ".join(f"{v:.2f}" for v in ex)
                     + f" → Next: {nxt:.2f}\n\n")

    # load HF model if needed
    if not args.model_name.startswith("openai:"):
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        ).to(DEVICE)

    # build prompts + collect y_true
    base_prompts, pap_prompts, y_true = [], [], []
    for i in range(args.window, len(test_vals)):
        w = test_vals[i-args.window:i]

        # metadata string
        meta = ""
        if args.use_metadata:
            dt = pd.to_datetime(test_times[i-args.window])
            meta = dt.strftime('%A %H:%M')

        p0 = baseline_prompt(w, meta=meta)
        freq = "hourly readings" + (f" ({meta})" if meta else "")
        p1 = pap_prompt(w, freq_desc=freq, bins=args.quant_bins)

        base_prompts.append(few_text + p0)
        pap_prompts .append(few_text + p1)
        y_true.append(test_vals[i])
    y_true = np.array(y_true, dtype=float)
    gen_tokens = args.max_new_tokens or args.horizon

    print(f"\n▶️ Forecasting with `{args.model_name}` (tokens={gen_tokens})…")
    if args.model_name.startswith("openai:"):
        om = args.model_name.split("openai:")[1]
        y_base = openai_predict(base_prompts, om, gen_tokens, args.fallback_strategy)
        y_pap  = openai_predict(pap_prompts,  om, gen_tokens, args.fallback_strategy)
    else:
        print("  • GPT-2 baseline…")
        y_base = gpt2_predict(base_prompts, model, tokenizer,
                              gen_tokens, args.fallback_strategy)
        print("  • GPT-2 PaP…")
        y_pap  = gpt2_predict(pap_prompts, model, tokenizer,
                              gen_tokens, args.fallback_strategy)

    print("  • ARIMA…")
    y_ari = arima_predict(train_vals, values[split:])

    # --- build & relabel results ---
    base_label = f"{args.model_name}-Baseline"
    pap_label  = f"{args.model_name}-PaP"
    ari_label  = "ARIMA"

    rows = []
    for label, preds in [(base_label, y_base), (pap_label, y_pap), (ari_label, y_ari)]:
        rows.append((
            label,
            mean_squared_error(y_true, preds),
            mean_absolute_error(y_true, preds),
            mase(y_true, preds, train_vals)
        ))

    df = pd.DataFrame(rows, columns=["Model","MSE","MAE","MASE"])
    df["run_model"] = args.model_name

    # write out metrics CSV
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"Saved metrics CSV → {args.output_csv}")

    # --- save visualizations & example table ---
    import matplotlib.pyplot as plt

    safe_name = args.model_name.replace("/", "_").replace(":", "_")

    # bar chart of metrics
    ax = df.plot(x="Model", y=["MSE","MAE","MASE"], kind="bar", figsize=(8,4))
    plt.tight_layout()
    bar_path = f"{safe_name}_metrics_bar.png"
    plt.savefig(bar_path)
    print(f"Saved bar chart → {bar_path}")
    plt.clf()

    # line plot: first 50 forecasts vs actual
    N = min(50, len(y_true))
    plt.figure(figsize=(9,3))
    plt.plot(test_times[:N], y_true[:N], label="Actual")
    plt.plot(test_times[:N], y_base[:N], label="Baseline")
    plt.plot(test_times[:N], y_pap[:N],  label="PaP")
    plt.plot(test_times[:N], y_ari[:N],  label="ARIMA")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    line_path = f"{safe_name}_forecasts_line.png"
    plt.savefig(line_path)
    print(f"Saved line plot  → {line_path}")
    plt.clf()

    # example table
    examples = pd.DataFrame({
        "time":     test_times[:N],
        "actual":   y_true[:N],
        "baseline": y_base[:N],
        "pap":      y_pap[:N],
    })
    table_path = f"{safe_name}_example_table.csv"
    examples.to_csv(table_path, index=False)
    print(f"Saved example table → {table_path}")


if __name__ == "__main__":
    main()
