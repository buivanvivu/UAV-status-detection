#!/usr/bin/env python3
"""
Train model phan loai trang thai / hanh dong UAV tu dataset.csv
(tao boi preprocess.py).

Chay:
    python3 train_model.py --data dataset.csv --out_dir ./model

Yeu cau: pip install pandas numpy scikit-learn joblib matplotlib
"""
import argparse
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset.csv")
    ap.add_argument("--out_dir", default="./model")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.data).dropna()
    X = df.drop(columns=["label"]).values
    y_raw = df["label"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42
    )
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(le.classes_)))
    ax.set_xticklabels(le.classes_, rotation=45, ha="right")
    ax.set_yticks(range(len(le.classes_)))
    ax.set_yticklabels(le.classes_)
    ax.set_xlabel("Du doan")
    ax.set_ylabel("Thuc te")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    plt.colorbar(im)
    plt.tight_layout()
    cm_path = os.path.join(args.out_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Luu confusion matrix: {cm_path}")

    feat_names = df.drop(columns=["label"]).columns
    importances = pd.Series(clf.feature_importances_, index=feat_names).sort_values(ascending=False)
    print("\nTop 10 dac trung quan trong nhat:")
    print(importances.head(10))

    joblib.dump(clf, os.path.join(args.out_dir, "uav_state_model.joblib"))
    joblib.dump(scaler, os.path.join(args.out_dir, "scaler.joblib"))
    joblib.dump(le, os.path.join(args.out_dir, "label_encoder.joblib"))
    print(f"\nDa luu model, scaler, label encoder vao: {args.out_dir}")


if __name__ == "__main__":
    main()
