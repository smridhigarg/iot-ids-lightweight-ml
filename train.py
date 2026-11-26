"""Lightweight ML-based intrusion detection for IoT traffic.

Usage examples:
  python train.py --data data/ --label-col label --binary --sample 200000
  python train.py --data data/part-00001.csv --label-col label

Works with CIC-IoT2023 / ToN-IoT style flow CSVs (numeric features + a label column).
"""
import argparse, glob, io, json, os, time
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             balanced_accuracy_score)

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

BENIGN = {"benign", "benigntraffic", "normal", "0", "0.0"}
SEED = 42


def load(path, label_col, sample):
    files = sorted(glob.glob(os.path.join(path, "*.csv"))) if os.path.isdir(path) else [path]
    if not files:
        raise SystemExit("No CSV files found in " + path)
    per_file = max(sample // len(files), 1) if sample else None
    parts = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df.columns = df.columns.str.strip()
        if per_file and len(df) > per_file:
            df = df.sample(per_file, random_state=SEED)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def prepare(df, label_col, binary):
    if label_col not in df.columns:
        raise SystemExit(f"Label column '{label_col}' not found. Columns: {list(df.columns)[:20]}...")
    y_raw = df[label_col].astype(str)
    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)
    keep = X.notna().all(axis=1)
    X, y_raw = X[keep], y_raw[keep]
    X = X.loc[:, X.nunique() > 1]  # drop constant columns
    if binary:
        y = (~y_raw.str.lower().str.replace(" ", "").isin(BENIGN)).astype(int).values
        names = ["Benign", "Attack"]
    else:
        codes, uniques = pd.factorize(y_raw)
        y, names = codes, list(uniques)
    return X, y, names


def size_kb(model):
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return round(buf.getbuffer().nbytes / 1024, 1)


def evaluate(name, model, Xtr, ytr, Xte, yte):
    t0 = time.perf_counter(); model.fit(Xtr, ytr); train_s = time.perf_counter() - t0
    t0 = time.perf_counter(); pred = model.predict(Xte); inf_s = time.perf_counter() - t0
    row = {
        "model": name,
        "n_features": Xtr.shape[1],
        "accuracy": accuracy_score(yte, pred),
        "precision": precision_score(yte, pred, average="weighted", zero_division=0),
        "recall": recall_score(yte, pred, average="weighted", zero_division=0),
        "f1": f1_score(yte, pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(yte, pred),
        "train_time_s": round(train_s, 2),
        "inference_ms_per_1000": round(inf_s / len(Xte) * 1000 * 1000, 3),
        "model_size_kb": size_kb(model),
    }
    if len(np.unique(yte)) == 2:  # binary: per-class recall (class 0 = Benign)
        row["benign_recall"] = recall_score(yte, pred, pos_label=0, zero_division=0)
        row["attack_recall"] = recall_score(yte, pred, pos_label=1, zero_division=0)
    return row, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--binary", action="store_true", help="Benign vs Attack")
    ap.add_argument("--sample", type=int, default=200000, help="max rows total (0 = all)")
    ap.add_argument("--topk", type=int, default=15, help="features kept by selection")
    ap.add_argument("--balanced", action="store_true",
                    help="class-weighted training to handle imbalance")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = load(a.data, a.label_col, a.sample)
    X, y, names = prepare(df, a.label_col, a.binary)
    print(f"Rows: {len(X)}, features: {X.shape[1]}, classes: {len(names)}")
    print(pd.Series(y).value_counts().rename(lambda i: names[i]).to_string())

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)

    # Feature selection: RF importance on the training set only
    sel = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=SEED).fit(Xtr, ytr)
    imp = pd.Series(sel.feature_importances_, index=X.columns).sort_values(ascending=False)
    top = list(imp.index[:a.topk])
    imp.head(20)[::-1].plot.barh(figsize=(7, 6), title="Top feature importances")
    plt.tight_layout(); plt.savefig(f"{a.out}/feature_importance.png", dpi=200); plt.close()

    cw = "balanced" if a.balanced else None
    spw = 1.0
    if a.balanced and a.binary:
        spw = float((ytr == 0).sum()) / float((ytr == 1).sum())  # negatives / positives
    rf_kw = dict(n_estimators=100, n_jobs=-1, random_state=SEED, class_weight=cw)
    experiments = [
        ("Logistic Regression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight=cw)), list(X.columns)),
        ("Decision Tree", DecisionTreeClassifier(random_state=SEED, class_weight=cw), list(X.columns)),
        ("Random Forest (all features)", RandomForestClassifier(**rf_kw), list(X.columns)),
        (f"Random Forest (top-{a.topk})", RandomForestClassifier(**rf_kw), top),
        (f"Random Forest small (top-{a.topk}, 30 trees, depth 12)",
         RandomForestClassifier(n_estimators=30, max_depth=12, n_jobs=-1, random_state=SEED,
                                class_weight=cw), top),
    ]
    if XGBClassifier:
        experiments.append((f"XGBoost (top-{a.topk})",
                            XGBClassifier(n_estimators=100, max_depth=6, n_jobs=-1, random_state=SEED,
                                          eval_metric="logloss", scale_pos_weight=spw), top))

    rows, preds = [], {}
    for name, model, cols in experiments:
        print("Training:", name)
        r, p = evaluate(name, model, Xtr[cols], ytr, Xte[cols], yte)
        rows.append(r); preds[name] = p
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    res = pd.DataFrame(rows)
    res.to_csv(f"{a.out}/results.csv", index=False)
    print("\n", res.to_string(index=False))

    best = res.sort_values("f1", ascending=False).iloc[0]["model"]
    cm = confusion_matrix(yte, preds[best])
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=names if len(names) <= 12 else None).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(f"Confusion matrix: {best}")
    plt.tight_layout(); plt.savefig(f"{a.out}/confusion_matrix.png", dpi=200); plt.close()

    json.dump({"rows": int(len(X)), "features_total": int(X.shape[1]), "top_features": top,
               "classes": names, "best_model": best, "binary": a.binary, "balanced": a.balanced},
              open(f"{a.out}/summary.json", "w"), indent=2)
    print("\nSaved to", a.out)


if __name__ == "__main__":
    main()
