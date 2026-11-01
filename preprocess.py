#!/usr/bin/env python3
"""
Gop cac file CSV thu duoc tu data_collector.py, cat theo cua so truot
(sliding window) TRONG TUNG DOAN BAY (segment) rieng biet, trich xuat dac trung
thong ke va xuat ra 1 file dataset.csv dung de train model.

Chay:
    python3 preprocess.py --data_dir ./data --out dataset.csv --window 20 --stride 10

window=20, stride=10 voi tan so lay mau ~10Hz tuong duong cua so ~2s,
truot 1s moi buoc. Dieu chinh theo toc do dong hoc cua UAV/kich ban.

Nhan cuoi cung (cot 'label') duoc suy ra tu 2 cot maneuver/fault do
data_collector.py ghi:
    - fault == "normal"  -> label = maneuver   (vd: hover, forward, ...)
    - fault != "normal"  -> label = fault       (vd: motor_fault, wind, gps_glitch)
Neu ban muon quy tac khac (vd giu maneuver rieng, train 2 model), sua ham
`derive_label`.

Cac dong co maneuver == "ignore" (doan chuyen tiep / recenter) bi bo qua.
Cua so truot chi chay BEN TRONG tung (pass_id, seg_id) - khong vat qua ranh
gioi 2 doan bay khac nhau nhu ban truoc.

Yeu cau: pip install pandas numpy
"""
import argparse
import glob
import os

import pandas as pd

FEATURE_COLS = [
    "roll", "pitch", "yaw",
    "pos_n", "pos_e", "pos_d",
    "vel_n", "vel_e", "vel_d",
    "servo1", "servo2", "servo3", "servo4",
    "rc1", "rc2", "rc3", "rc4",
]
RATE_COLS = ["roll", "pitch", "pos_d"]
LABEL_COLS = ["maneuver", "fault", "pass_id", "seg_id"]


def load_all(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"Khong tim thay file CSV nao trong {data_dir}")
    return files, [pd.read_csv(f) for f in files]


def derive_label(maneuver, fault):
    """Quy tac gan nhan cuoi cung tu (maneuver, fault). Sua o day neu muon
    quy tac khac."""
    return maneuver if fault == "normal" else fault


def make_windows(chunk_df, window, stride, label):
    rows = []
    n = len(chunk_df)
    for start in range(0, max(n - window + 1, 0), stride):
        chunk = chunk_df.iloc[start:start + window]
        feat = {}
        for col in FEATURE_COLS:
            feat[f"{col}_mean"] = chunk[col].mean()
            feat[f"{col}_std"] = chunk[col].std()
            feat[f"{col}_min"] = chunk[col].min()
            feat[f"{col}_max"] = chunk[col].max()
        for col in RATE_COLS:
            dt = max(chunk["time"].iloc[-1] - chunk["time"].iloc[0], 1e-6)
            feat[f"{col}_rate"] = (chunk[col].iloc[-1] - chunk[col].iloc[0]) / dt
        feat["label"] = label
        rows.append(feat)
    return rows


def process_file(fname, df, window, stride):
    missing = [c for c in FEATURE_COLS + LABEL_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{fname}: thieu cot {missing} - file nay co dung dinh dang cua "
            "data_collector.py hien tai khong (co the la CSV thu bang ban cu)?"
        )

    df = df[df["maneuver"] != "ignore"].reset_index(drop=True)
    if df.empty:
        print(f"  [bo qua] {fname}: chi co doan 'ignore', khong co du lieu dung")
        return []

    all_rows = []
    for (pass_id, seg_id), group in df.groupby(["pass_id", "seg_id"], sort=False):
        group = group.reset_index(drop=True)
        maneuvers = group["maneuver"].unique()
        faults = group["fault"].unique()
        if len(maneuvers) > 1 or len(faults) > 1:
            print(f"  [canh bao] {fname} seg {pass_id}/{seg_id}: lan nhieu nhan "
                  f"{list(maneuvers)}/{list(faults)} trong cung 1 doan - "
                  "dung nhan pho bien nhat, kiem tra lai data_collector/scenario_runner")
        maneuver = group["maneuver"].mode()[0]
        fault = group["fault"].mode()[0]
        label = derive_label(maneuver, fault)

        if len(group) < window:
            print(f"  [canh bao] {fname} seg {pass_id}/{seg_id} ({label}): "
                  f"chi co {len(group)} dong, it hon window={window} -> bo qua doan nay")
            continue
        all_rows.extend(make_windows(group, window, stride, label))
    return all_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--out", default="dataset.csv")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--stride", type=int, default=10)
    args = ap.parse_args()

    files, dfs = load_all(args.data_dir)
    all_rows = []
    print(f"Dang xu ly {len(files)} file CSV...")
    for fname, df in zip(files, dfs):
        all_rows.extend(process_file(os.path.basename(fname), df, args.window, args.stride))

    if not all_rows:
        raise RuntimeError(
            "Khong tao duoc mau dac trung nao. Kiem tra: co dong nao khac 'ignore' "
            "khong, cac doan co du >= window dong khong."
        )

    dataset = pd.DataFrame(all_rows).dropna()
    dataset.to_csv(args.out, index=False)

    print(f"\nDa tao {len(dataset)} mau dac trung, {dataset['label'].nunique()} lop:")
    counts = dataset["label"].value_counts()
    print(counts)
    imbalance = counts.max() / max(counts.min(), 1)
    if imbalance > 5:
        print(f"[canh bao] lop nhieu nhat gap {imbalance:.1f}x lop it nhat - "
              "can nhac thu them du lieu cho lop it (train_model.py da dung "
              "class_weight='balanced' san, nhung mat can bang qua lon van anh huong).")
    print(f"Luu tai: {args.out}")


if __name__ == "__main__":
    main()