#!/usr/bin/env python3
"""
Gop cac file CSV thu duoc tu data_collector.py, cat theo cua so truot
(sliding window), trich xuat dac trung thong ke va xuat ra 1 file
dataset.csv dung de train model.

Chay:
    python3 preprocess.py --data_dir ./data --out dataset.csv --window 20 --stride 10

window=20, stride=10 voi tan so lay mau ~10Hz tuong duong cua so ~2s,
truot 1s moi buoc. Dieu chinh theo toc do dong hoc cua UAV/kich ban.

Yeu cau: pip install pandas numpy
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "roll", "pitch", "yaw",
    "pos_n", "pos_e", "pos_d",
    "vel_n", "vel_e", "vel_d",
    "servo1", "servo2", "servo3", "servo4",
    "rc1", "rc2", "rc3", "rc4",
]


def load_all(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"Khong tim thay file CSV nao trong {data_dir}")
    return [pd.read_csv(f) for f in files]


def make_windows(df, window, stride):
    rows = []
    n = len(df)
    for start in range(0, max(n - window + 1, 0), stride):
        chunk = df.iloc[start:start + window]
        feat = {}
        for col in FEATURE_COLS:
            feat[f"{col}_mean"] = chunk[col].mean()
            feat[f"{col}_std"] = chunk[col].std()
            feat[f"{col}_min"] = chunk[col].min()
            feat[f"{col}_max"] = chunk[col].max()
        # toc do bien thien (dao ham xap xi) cho vai dac trung dong hoc quan trong
        for col in ["roll", "pitch", "pos_d"]:
            dt = max(chunk["time"].iloc[-1] - chunk["time"].iloc[0], 1e-6)
            feat[f"{col}_rate"] = (chunk[col].iloc[-1] - chunk[col].iloc[0]) / dt
        feat["label"] = chunk["label"].mode()[0]
        rows.append(feat)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--out", default="dataset.csv")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()

    dfs = load_all(args.data_dir)
    windows = [make_windows(df, args.window, args.stride) for df in dfs]
    windows = [w for w in windows if len(w) > 0]
    dataset = pd.concat(windows, ignore_index=True).dropna()
    dataset.to_csv(args.out, index=False)

    print(f"Da tao {len(dataset)} mau dac trung, {dataset['label'].nunique()} lop:")
    print(dataset["label"].value_counts())
    print(f"Luu tai: {args.out}")


if __name__ == "__main__":
    main()
