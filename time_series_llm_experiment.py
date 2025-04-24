"""
Time-Series Forecasting Comparison: Baseline vs. Prompt-as-Prefix (PaP) on GPT-2 and ARIMA

Usage:
    python time_series_llm_experiment.py \
        --dataset_csv ./dataset/ETTh1.csv \
        --time_col date \
        --value_col OT \
        --window 12 \
        --horizon 1 \
        --quant_bins 10 \
        [--max_rows 500]

Requirements:
    pip install transformers torch pandas numpy scikit-learn statsmodels matplotlib
"""
import argparse
import pandas as pd
import numpy as np
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


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
    if bins is not None:
        mn, mx = vals.min(), vals.max()
        q = np.floor((vals - mn) / (mx - mn + 1e-8) * bins).astype(int)
        vals = q
    vals_str = ", ".join(str(int(v)) if bins else f"{v:.2f}" for v in vals)
    header = f"Given the historical {freq_desc}, forecast the next value:\nData:"  
    return f"{header} {vals_str}. Next:"

# Generate next-step predictions using GPT-2 with fallback for non-numeric output

def gpt2_predict(prompts, model, tokenizer, device, fallback_strategy="last"):
    preds = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        text = tokenizer.decode(
            out[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        ).strip()
        # extract first numeric substring
        m = re.search(r"[-+]?[0-9]*\.?[0-9]+", text)
        if m:
            val = float(m.group())
        else:
            nums = [float(x) for x in re.findall(r"[-+]?[0-9]*\.?[0-9]+", prompt)]
            if fallback_strategy == "last":
                val = nums[-1]
            elif fallback_strategy == "mean":
                val = float(np.mean(nums))
            else:
                val = 0.0
        preds.append(val)
    return np.array(preds, dtype=float)

# ARIMA forecasting

def arima_predict(train, test, order=(1,1,1)):
    model = ARIMA(train, order=order)
    res = model.fit()
    return res.forecast(steps=len(test))

# Metrics

def mase(y_true, y_pred, y_train):
    naiv_err = np.abs(np.diff(y_train)).mean()
    return mean_absolute_error(y_true, y_pred) / naiv_err

# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_csv", type=str, required=True)
    parser.add_argument("--time_col", type=str, default="date")
    parser.add_argument("--value_col", type=str, default="value")
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--quant_bins", type=int, default=10)
    parser.add_argument("--max_rows", type=int, default=None,
                        help="(Optional) limit the series to the first N rows before split")
    args = parser.parse_args()

    # Load series
    times, values = load_series(args.dataset_csv, args.time_col, args.value_col)

    # Truncate for quick debug/test
    if args.max_rows is not None:
        times  = times[:args.max_rows]
        values = values[:args.max_rows]

    # Train/test split (80/20)
    n = len(values)
    split = int(n * 0.8)
    train_vals = values[:split]
    test_vals  = values[split - args.window:]
    test_times = times[split:]

    # GPT-2 Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model     = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    # Build prompts and true values
    baseline_prompts, pap_prompts, y_true = [], [], []
    for i in range(args.window, len(test_vals)):
        window = test_vals[i - args.window:i]
        baseline_prompts.append(baseline_prompt(window))
        pap_prompts.append(pap_prompt(window, freq_desc="hourly readings", bins=args.quant_bins))
        y_true.append(test_vals[i])
    y_true = np.array(y_true, dtype=float)

    # GPT-2 predictions
    print("Running GPT-2 baseline prompts...")
    y_pred_base = gpt2_predict(baseline_prompts, model, tokenizer, device, fallback_strategy="last")
    print("Running GPT-2 PaP prompts...")
    y_pred_pap  = gpt2_predict(pap_prompts,    model, tokenizer, device, fallback_strategy="last")

    # ARIMA
    print("Running ARIMA baseline...")
    arima_train = train_vals
    arima_test  = values[split:]
    y_pred_arima = arima_predict(arima_train, arima_test)

    # Compute metrics
    results = []
    for name, y_pred in [("GPT2-Baseline", y_pred_base), ("GPT2-PaP", y_pred_pap), ("ARIMA", y_pred_arima)]:
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        ms  = mase(y_true, y_pred, train_vals)
        results.append((name, mse, mae, ms))

    # Display metrics
    df_res = pd.DataFrame(results, columns=["Model","MSE","MAE","MASE"])
    print(df_res)

    # Plot metrics
    df_res.plot(x="Model", y=["MSE","MAE","MASE"], kind="bar", figsize=(8,5))
    plt.title("Forecasting Metrics Comparison")
    plt.tight_layout()
    plt.show()

    # Sample forecasts vs actuals
    plt.figure(figsize=(10,4))
    pts = min(50, len(y_true))
    plt.plot(test_times[:pts], y_true[:pts], label="Actual")
    plt.plot(test_times[:pts], y_pred_base[:pts], label="GPT2-Baseline")
    plt.plot(test_times[:pts], y_pred_pap[:pts], label="GPT2-PaP")
    plt.plot(test_times[:pts], y_pred_arima[:pts],label="ARIMA")
    plt.legend(); plt.xticks(rotation=45)
    plt.title("First 50 Forecasts vs Actual")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
