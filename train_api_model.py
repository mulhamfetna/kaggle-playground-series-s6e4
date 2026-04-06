import argparse
import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier


def to_catboost_features(df: pd.DataFrame, categorical_features: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in categorical_features:
        out[col] = out[col].fillna("__MISSING__").astype(str)
    return out


def main(data_dir: Path, output_dir: Path, iterations: int, seed: int) -> None:
    train_df = pd.read_csv(data_dir / "train.csv")
    target_col = "Irrigation_Need"
    id_col = "id"

    y = train_df[target_col]
    X = train_df.drop(columns=[target_col])
    feature_cols = [c for c in X.columns if c != id_col]
    X = X[feature_cols]

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in feature_cols if c not in numeric_features]
    cat_features_idx = [feature_cols.index(c) for c in categorical_features]
    X_cb = to_catboost_features(X, categorical_features)

    model = CatBoostClassifier(
        iterations=iterations,
        depth=8,
        learning_rate=0.05,
        loss_function="MultiClass",
        auto_class_weights="Balanced",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        task_type="CPU",
    )
    model.fit(X_cb, y, cat_features=cat_features_idx)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "catboost_model.cbm"
    metadata_path = output_dir / "model_metadata.json"

    model.save_model(str(model_path))
    metadata = {
        "target_col": target_col,
        "id_col": id_col,
        "feature_cols": feature_cols,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "classes": sorted(y.unique().tolist()),
        "model_path": str(model_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CatBoost model and export artifacts for FastAPI inference")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Directory containing train.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("model_artifacts"), help="Artifact output directory")
    parser.add_argument("--iterations", type=int, default=700, help="CatBoost iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.data_dir, args.output_dir, args.iterations, args.seed)
