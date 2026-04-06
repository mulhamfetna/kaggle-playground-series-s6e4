import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def resolve_device(device: str) -> str:
    if device == "auto":
        if os.getenv("KAGGLE_KERNEL_RUN_TYPE") or os.getenv("KAGGLE_URL_BASE"):
            return "gpu"
        return "cpu"
    return device


def build_catboost_model(task_type: str, gpu_devices: str, iterations: int, random_seed: int):
    model_kwargs = {
        "iterations": iterations,
        "depth": 8,
        "learning_rate": 0.05,
        "loss_function": "MultiClass",
        "auto_class_weights": "Balanced",
        "random_seed": random_seed,
        "verbose": False,
        "allow_writing_files": False,
        "task_type": "GPU" if task_type == "gpu" else "CPU",
    }
    if task_type == "gpu":
        model_kwargs["devices"] = gpu_devices
    return CatBoostClassifier(**model_kwargs)


def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def build_hgb_pipeline(numeric_features, categorical_features, model_params=None):
    if model_params is None:
        model_params = {}

    model = HistGradientBoostingClassifier(
        random_state=42,
        class_weight="balanced",
        learning_rate=0.05,
        max_depth=10,
        max_leaf_nodes=63,
        min_samples_leaf=40,
        max_bins=255,
        **model_params,
    )

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("model", model),
        ]
    )


def get_cv_subset(X, y, max_rows):
    if max_rows <= 0 or len(X) <= max_rows:
        return X, y

    X_sub, _, y_sub, _ = train_test_split(
        X,
        y,
        train_size=max_rows,
        random_state=42,
        stratify=y,
    )
    return X_sub, y_sub


def to_catboost_features(X, categorical_features):
    X_cb = X.copy()
    for col in categorical_features:
        X_cb[col] = X_cb[col].fillna("__MISSING__").astype(str)
    return X_cb


def catboost_cv_scores(
    X,
    y,
    cv,
    cat_features_idx,
    task_type: str,
    gpu_devices: str,
    iterations: int,
    random_seed: int,
):
    scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        model = build_catboost_model(task_type, gpu_devices, iterations, random_seed)
        model.fit(X_train, y_train, cat_features=cat_features_idx)
        val_pred = model.predict(X_val).ravel()
        fold_score = balanced_accuracy_score(y_val, val_pred)
        print(f"CatBoost fold {fold_idx} balanced accuracy: {fold_score:.6f}")
        scores.append(fold_score)

    return np.array(scores)


def aligned_proba_df(model, X, all_classes):
    proba = model.predict_proba(X)
    class_to_idx = {c: i for i, c in enumerate(model.classes_)}
    arr = np.zeros((len(X), len(all_classes)), dtype=float)
    for j, cls in enumerate(all_classes):
        arr[:, j] = proba[:, class_to_idx[cls]]
    return arr


def parse_seed_list(seed_list: str):
    values = [s.strip() for s in seed_list.split(",") if s.strip()]
    if not values:
        raise ValueError("At least one CatBoost seed is required.")
    return [int(s) for s in values]


def main(
    data_dir: Path,
    output_file: Path,
    report_file: Path,
    cv_sample_size: int,
    cv_folds: int,
    search_iters: int,
    cat_iterations: int,
    cat_seeds: str,
    device: str,
    gpu_devices: str,
):
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
    resolved_device = resolve_device(device)
    print(f"Training device: {resolved_device.upper()}")

    if cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    seed_values = parse_seed_list(cat_seeds)

    X_cv, y_cv = get_cv_subset(X, y, cv_sample_size)
    print(f"CV rows used: {len(X_cv):,}")

    baseline_scores = None
    tuned_scores = None
    tuned_hgb = None
    search = None

    if resolved_device == "cpu":
        baseline_hgb = build_hgb_pipeline(numeric_features, categorical_features)
        baseline_scores = cross_val_score(
            baseline_hgb,
            X_cv,
            y_cv,
            cv=cv,
            scoring="balanced_accuracy",
            n_jobs=-1,
        )
        print(f"HGB baseline CV balanced accuracy: {baseline_scores.mean():.6f} +/- {baseline_scores.std():.6f}")

        search_pipeline = build_hgb_pipeline(numeric_features, categorical_features)
        param_distributions = {
            "model__learning_rate": np.logspace(-2.0, -0.9, num=40),
            "model__max_depth": [6, 8, 10, 12, None],
            "model__max_leaf_nodes": [31, 63, 127],
            "model__min_samples_leaf": [10, 20, 40, 80, 120],
            "model__l2_regularization": np.logspace(-4, 1, num=30),
        }

        if search_iters > 0:
            search = RandomizedSearchCV(
                estimator=search_pipeline,
                param_distributions=param_distributions,
                n_iter=search_iters,
                scoring="balanced_accuracy",
                cv=min(3, cv_folds),
                random_state=42,
                n_jobs=-1,
                verbose=1,
            )
            search.fit(X_cv, y_cv)
            tuned_hgb = search.best_estimator_
            print(f"Best tuned HGB score (CV3 on subset): {search.best_score_:.6f}")
            print(f"Best tuned HGB params: {search.best_params_}")

            tuned_scores = cross_val_score(
                tuned_hgb,
                X_cv,
                y_cv,
                cv=cv,
                scoring="balanced_accuracy",
                n_jobs=-1,
            )
            print(f"HGB tuned CV balanced accuracy: {tuned_scores.mean():.6f} +/- {tuned_scores.std():.6f}")
        else:
            print("Skipping HGB randomized search (--search-iters=0).")
    else:
        print("GPU mode enabled: skipping CPU-heavy HGB CV/search and using CatBoost path.")

    cat_features_idx = [X.columns.get_loc(c) for c in categorical_features]
    X_cv_cb = to_catboost_features(X_cv, categorical_features)
    cat_scores = catboost_cv_scores(
        X_cv_cb,
        y_cv,
        cv,
        cat_features_idx,
        resolved_device,
        gpu_devices,
        cat_iterations,
        seed_values[0],
    )
    print(f"CatBoost CV balanced accuracy: {cat_scores.mean():.6f} +/- {cat_scores.std():.6f}")

    X_cb_full = to_catboost_features(X, categorical_features)
    X_test_cb = to_catboost_features(X_test, categorical_features)
    classes = sorted(y.unique())
    cat_seed_proba = []
    for seed in seed_values:
        print(f"Fitting CatBoost seed {seed}...")
        cat_model = build_catboost_model(resolved_device, gpu_devices, cat_iterations, seed)
        cat_model.fit(X_cb_full, y, cat_features=cat_features_idx)
        cat_seed_proba.append(aligned_proba_df(cat_model, X_test_cb, classes))
    cat_proba = np.mean(cat_seed_proba, axis=0)
    if tuned_hgb is not None and tuned_scores is not None:
        tuned_hgb.fit(X, y)
        hgb_proba = aligned_proba_df(tuned_hgb, X_test, classes)
        hgb_w = max(float(tuned_scores.mean()), 1e-6)
    else:
        hgb_proba = None
        hgb_w = 0.0
    cat_w = max(float(cat_scores.mean()), 1e-6)
    if hgb_proba is not None:
        blend_proba = (hgb_w * hgb_proba + cat_w * cat_proba) / (hgb_w + cat_w)
    else:
        blend_proba = cat_proba
    blend_pred = np.array(classes)[blend_proba.argmax(axis=1)]

    submission = pd.DataFrame({id_col: test_df[id_col], target_col: blend_pred})
    submission.to_csv(output_file, index=False)
    print(f"Saved submission to: {output_file}")

    report = {
        "device": resolved_device,
        "gpu_devices": gpu_devices if resolved_device == "gpu" else None,
        "cv_rows_used": int(len(X_cv)),
        "cv_folds": int(cv_folds),
        "cat_iterations": int(cat_iterations),
        "cat_seeds": seed_values,
        "hgb_baseline_cv_mean": float(baseline_scores.mean()) if baseline_scores is not None else None,
        "hgb_baseline_cv_std": float(baseline_scores.std()) if baseline_scores is not None else None,
        "hgb_tuned_cv_mean": float(tuned_scores.mean()) if tuned_scores is not None else None,
        "hgb_tuned_cv_std": float(tuned_scores.std()) if tuned_scores is not None else None,
        "hgb_search_best_cv3_score": float(search.best_score_) if search is not None else None,
        "hgb_search_best_params": (
            {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
            if search is not None
            else None
        ),
        "catboost_cv_mean": float(cat_scores.mean()),
        "catboost_cv_std": float(cat_scores.std()),
        "blend_weights": {"hgb_tuned": hgb_w, "catboost": cat_w},
        "classes": classes,
        "output_file": str(output_file),
    }
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report to: {report_file}")



def parse_args():
    parser = argparse.ArgumentParser(description="Advanced training and submission builder")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Directory with train.csv and test.csv")
    parser.add_argument("--output", type=Path, default=Path("submission_ready.csv"), help="Submission CSV output path")
    parser.add_argument("--report", type=Path, default=Path("model_report.json"), help="Training report JSON path")
    parser.add_argument(
        "--cv-sample-size",
        type=int,
        default=250000,
        help="Maximum rows used for CV/search for speed (0 = full train)",
    )
    parser.add_argument(
        "--search-iters",
        type=int,
        default=14,
        help="Randomized search iterations for HGB",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Cross-validation folds (use 3 on Kaggle for speed)",
    )
    parser.add_argument(
        "--cat-iterations",
        type=int,
        default=600,
        help="CatBoost iterations (lower values are faster on Kaggle)",
    )
    parser.add_argument(
        "--cat-seeds",
        type=str,
        default="42,314,2024",
        help="Comma-separated random seeds for CatBoost test-time ensembling",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Compute device (auto prefers GPU on Kaggle)",
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
    main(
        data_dir=args.data_dir,
        output_file=args.output,
        report_file=args.report,
        cv_sample_size=args.cv_sample_size,
        cv_folds=args.cv_folds,
        search_iters=args.search_iters,
        cat_iterations=args.cat_iterations,
        cat_seeds=args.cat_seeds,
        device=args.device,
        gpu_devices=args.gpu_devices,
    )
