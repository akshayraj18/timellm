#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# force CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_series(path, time_col, value_col):
    df = pd.read_csv(path, parse_dates=[time_col])
    df = df.sort_values(time_col)
    return df[time_col].values, df[value_col].values


def baseline_prompt(window_values):
    vals = ", ".join(f"{v:.2f}" for v in window_values)
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


def gpt2_predict(prompts, model, tokenizer, gen_tokens, fallback_strategy="last"):
    preds = []
    for prompt in prompts:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        out = model.generate(
            **inputs,
            max_new_tokens=gen_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        # decode only the new tokens
        new_text = tokenizer.decode(
            out[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()
        m = re.search(r"[-+]?[0-9]*\.?[0-9]+", new_text)
        if m:
            preds.append(float(m.group()))
        else:
            # fallback
            nums = [float(x) for x in re.findall(r"[-+]?[0-9]*\.?[0-9]+", prompt)]
            if fallback_strategy == "mean":
                preds.append(float(np.mean(nums)))
            elif fallback_strategy == "zero":
                preds.append(0.0)
            else:
                preds.append(nums[-1])
    return np.array(preds, dtype=float)


def arima_predict(train, test, order=(1, 1, 1)):
    res = ARIMA(train, order=order).fit()
    return res.forecast(steps=len(test))


def mase(y_true, y_pred, y_train):
    return mean_absolute_error(y_true, y_pred) / np.abs(np.diff(y_train)).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_csv",   required=True)
    p.add_argument("--time_col",      default="date")
    p.add_argument("--value_col",     default="value")
    p.add_argument("--window",        type=int,   default=12)
    p.add_argument("--quant_bins",    type=int,   default=10)
    p.add_argument("--max_rows",      type=int,   default=None)
    p.add_argument("--fallback_strategy",
                   choices=["last", "mean", "zero"],
                   default="last")
    p.add_argument("--horizon",       type=int,   default=1,
                   help="forecast horizon (used if --max_new_tokens not set)")
    p.add_argument("--max_new_tokens", type=int,  default=None,
                   help="override horizon for # of tokens to generate")
    p.add_argument("--few_shot",      type=int,   default=0,
                   help="(unused) placeholder for few-shot")
    p.add_argument("--use_metadata",  action="store_true",
                   help="(unused) placeholder for metadata flag")
    p.add_argument("--model_name",    default="sshleifer/tiny-gpt2")
    p.add_argument("--output_csv",    default=None,
                   help="if set, write metrics here and exit")
    args = p.parse_args()

    # load & truncate
    times, values = load_series(
        args.dataset_csv, args.time_col, args.value_col
    )
    if args.max_rows:
        times, values = times[:args.max_rows], values[:args.max_rows]

    # train/test split
    split = int(len(values) * 0.8)
    train_vals = values[:split]
    test_vals  = values[split - args.window :]
    test_times = times[split :]

    # load LLM
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    ).to(DEVICE)

    # build prompts
    base_prompts, pap_prompts, y_true = [], [], []
    for i in range(args.window, len(test_vals)):
        w = test_vals[i - args.window : i]
        base_prompts.append(baseline_prompt(w))
        pap_prompts .append(pap_prompt(w, bins=args.quant_bins))
        y_true.append(test_vals[i])
    y_true = np.array(y_true, dtype=float)

    # decide how many tokens to gen
    gen_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else args.horizon
    )

    # forecasts
    print(f"GPT-2 baseline (tokens={gen_tokens})…")
    y_base = gpt2_predict(base_prompts, model, tokenizer,
                          gen_tokens, args.fallback_strategy)
    print(f"GPT-2 PaP      (tokens={gen_tokens})…")
    y_pap  = gpt2_predict(pap_prompts,  model, tokenizer,
                          gen_tokens, args.fallback_strategy)
    print("ARIMA…")
    y_ari  = arima_predict(train_vals, values[split:])

    # metrics
    rows = []
    for name, y_pred in [
        ("GPT2-Baseline", y_base),
        ("GPT2-PaP",      y_pap),
        ("ARIMA",         y_ari),
    ]:
        rows.append((
            name,
            mean_squared_error(y_true, y_pred),
            mean_absolute_error(y_true, y_pred),
            mase(y_true, y_pred, train_vals),
        ))
    df = pd.DataFrame(rows, columns=["Model","MSE","MAE","MASE"])
    print(df)

    # if driver requested a CSV-only run, write & exit
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        return

    # otherwise show plots
    df.plot(x="Model", y=["MSE","MAE","MASE"], kind="bar", figsize=(8,4))
    plt.tight_layout(); plt.show()

    pts = min(50, len(y_true))
    plt.figure(figsize=(9,3))
    plt.plot(test_times[:pts], y_true[:pts],  label="Actual")
    plt.plot(test_times[:pts], y_base[:pts],  label="Baseline")
    plt.plot(test_times[:pts], y_pap[:pts],   label="PaP")
    plt.plot(test_times[:pts], y_ari[:pts],   label="ARIMA")
    plt.legend(); plt.xticks(rotation=45); plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
