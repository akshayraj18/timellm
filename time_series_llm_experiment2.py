"""
Time-Series Forecasting Comparison: Baseline vs. Prompt-as-Prefix (PaP) on GPT-2 and ARIMA

Usage:
    python time_series_llm_experiment.py \
        --dataset_csv ./dataset/ETTh1.csv \
        --time_col date \
        --value_col OT \
        --window 12 \
        --quant_bins 10 \
        --max_rows 500 \
        --fallback_strategy mean \
        --max_new_tokens 2 \
        --model_name sshleifer/tiny-gpt2

Requirements:
    pip install transformers torch pandas numpy scikit-learn statsmodels matplotlib safetensors

Notes:
- Default `sshleifer/tiny-gpt2` is only ~4M params and runs on Mac CPU without crashing.
- To test larger variants, pass e.g. `--model_name distilgpt2` or `gpt2-medium`.
- Installing `safetensors` can avoid Bus errors when reading safetensors weights.
"""
import argparse
import pandas as pd
import numpy as np
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Force CPU usage
DEVICE = torch.device("cpu")


def load_series(path, time_col, value_col):
    df = pd.read_csv(path, parse_dates=[time_col])
    df = df.sort_values(time_col)
    return df[time_col].values, df[value_col].values

# Prompt generators

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

# GPT-2 predictions

def gpt2_predict(prompts, model, tokenizer, max_new_tokens=1, fallback_strategy="last"):
    preds = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        text = tokenizer.decode(out[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
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

# ARIMA forecasting

def arima_predict(train, test, order=(1,1,1)):
    res = ARIMA(train, order=order).fit()
    return res.forecast(steps=len(test))

# MASE metric

def mase(y_true, y_pred, y_train):
    return mean_absolute_error(y_true, y_pred) / np.abs(np.diff(y_train)).mean()

# Main entry

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_csv", required=True)
    parser.add_argument("--time_col", default="date")
    parser.add_argument("--value_col", default="value")
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--quant_bins", type=int, default=10)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--fallback_strategy", choices=["last","mean","zero"], default="last")
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--model_name", default="sshleifer/tiny-gpt2",
                        help="HF model name: tiny-gpt2, distilgpt2, gpt2 etc.")
    args = parser.parse_args()

    # Load and optionally truncate series
    times, values = load_series(args.dataset_csv, args.time_col, args.value_col)
    if args.max_rows:
        times, values = times[:args.max_rows], values[:args.max_rows]

    # Train/test split
    split = int(len(values) * 0.8)
    train_vals = values[:split]
    test_vals  = values[split-args.window:]
    test_times = times[split:]

    # Load model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Ensure pad_token is defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    ).to(DEVICE)

    # Build prompts and collect ground truth
    base_prompts, pap_prompts, y_true = [], [], []
    for i in range(args.window, len(test_vals)):
        w = test_vals[i-args.window:i]
        base_prompts.append(baseline_prompt(w))
        pap_prompts.append(pap_prompt(w, bins=args.quant_bins))
        y_true.append(test_vals[i])
    y_true = np.array(y_true)

    # Run predictions
    print("Running baseline GPT-2...")
    y_base = gpt2_predict(base_prompts, model, tokenizer,
                          max_new_tokens=args.max_new_tokens,
                          fallback_strategy=args.fallback_strategy)
    print("Running PaP GPT-2...")
    y_pap  = gpt2_predict(pap_prompts, model, tokenizer,
                          max_new_tokens=args.max_new_tokens,
                          fallback_strategy=args.fallback_strategy)
    print("Running ARIMA...")
    y_ari  = arima_predict(train_vals, values[split:])

    # Compute metrics
    rows = [
        ("GPT2-Baseline", mean_squared_error(y_true, y_base), mean_absolute_error(y_true, y_base), mase(y_true, y_base, train_vals)),
        ("GPT2-PaP",      mean_squared_error(y_true, y_pap), mean_absolute_error(y_true, y_pap),   mase(y_true, y_pap, train_vals)),
        ("ARIMA",         mean_squared_error(y_true, y_ari), mean_absolute_error(y_true, y_ari),   mase(y_true, y_ari, train_vals))
    ]
    df = pd.DataFrame(rows, columns=["Model","MSE","MAE","MASE"])
    print(df)

    # Plot metrics
    df.plot(x="Model", y=["MSE","MAE","MASE"], kind="bar", figsize=(8,4))
    plt.tight_layout(); plt.show()

    # Plot first 50 forecasts
    pts = min(50, len(y_true))
    plt.figure(figsize=(9,3))
    plt.plot(test_times[:pts], y_true[:pts], label="Actual")
    plt.plot(test_times[:pts], y_base[:pts], label="Baseline")
    plt.plot(test_times[:pts], y_pap[:pts],  label="PaP")
    plt.plot(test_times[:pts], y_ari[:pts],  label="ARIMA")
    plt.legend(); plt.xticks(rotation=45); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
