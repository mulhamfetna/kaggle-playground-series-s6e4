import argparse
import os
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split


def resolve_device(device: str) -> str:
    if device == "auto":
        if os.getenv("KAGGLE_KERNEL_RUN_TYPE") or os.getenv("KAGGLE_URL_BASE"):
            return "gpu"
        return "cpu"
    return device


def prepare_catboost_frame(X: pd.DataFrame, categorical_features):
    X_cb = X.copy()
    for col in categorical_features:
        X_cb[col] = X_cb[col].fillna("__MISSING__").astype(str)
    return X_cb


def build_model(task_type: str, gpu_devices: str):
    model_kwargs = {
        "iterations": 800,
        "depth": 8,
        "learning_rate": 0.05,
        "loss_function": "MultiClass",
        "auto_class_weights": "Balanced",
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
        "task_type": "GPU" if task_type == "gpu" else "CPU",
    }
    if task_type == "gpu":
        model_kwargs["devices"] = gpu_devices
    return CatBoostClassifier(**model_kwargs)


def main(data_dir: Path, output_file: Path, val_size: float, device: str, gpu_devices: str):
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = "Irrigation_Need"
    id_col = "id"

    y = train_df[target_col]
    X = train_df.drop(columns=[target_col])

    feature_cols = [c for c in X.columns if c != id_col]
    X = X[feature_cols]
    X_test = test_df[feature_cols]

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    cat_features_idx = [X.columns.get_loc(c) for c in categorical_features]
    resolved_device = resolve_device(device)
    print(f"Training device: {resolved_device.upper()}")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=val_size,
        random_state=42,
        stratify=y,
    )

    X_train_cb = prepare_catboost_frame(X_train, categorical_features)
    X_val_cb = prepare_catboost_frame(X_val, categorical_features)
    X_full_cb = prepare_catboost_frame(X, categorical_features)
    X_test_cb = prepare_catboost_frame(X_test, categorical_features)

    model = build_model(resolved_device, gpu_devices)
    model.fit(X_train_cb, y_train, cat_features=cat_features_idx)

    val_pred = model.predict(X_val_cb).ravel()
    val_bal_acc = balanced_accuracy_score(y_val, val_pred)
    print(f"Validation balanced accuracy: {val_bal_acc:.6f}")

    model = build_model(resolved_device, gpu_devices)
    model.fit(X_full_cb, y, cat_features=cat_features_idx)
    test_pred = model.predict(X_test_cb).ravel()

    submission = pd.DataFrame(
        {
            id_col: test_df[id_col],
            target_col: test_pred,
        }
    )
    submission.to_csv(output_file, index=False)
    print(f"Saved submission to: {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train irrigation classifier and make submission")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Directory containing train.csv and test.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission_baseline.csv"),
        help="Output submission CSV path",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Validation split fraction",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Compute device for CatBoost (auto prefers GPU on Kaggle)",
    )
    parser.add_argument(
        "--gpu-devices",
        type=str,
        default="0",
        help='GPU devices string for CatBoost, e.g. "0" or "0:1"',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.data_dir, args.output, args.val_size, args.device, args.gpu_devices)
