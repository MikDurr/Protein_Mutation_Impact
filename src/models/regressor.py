"""Train a Random Forest regressor to predict continuous DMS fitness scores.

The primary model: maps delta ESM2 embeddings to the raw `DMS_score` and is
scored by Spearman correlation, matching how ProteinGym evaluates DMS models.
The zero-shot score is reported as a baseline only -- adding it as an input
was tested and made no difference (Spearman 0.6398 with vs 0.6405 without).
"""
from __future__ import annotations

import json

import joblib
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

from src.config import PATHS, RANDOM_STATE
from src.models.split import load_split, train_subsample_indices

MODEL_PATH = PATHS.models_dir / "rf_regressor.joblib"
METRICS_PATH = PATHS.models_dir / "regressor_metrics.json"


def main() -> None:
    idx = train_subsample_indices(len(load_split("y_continuous", "train")))
    X_train, y_train = load_split("X_delta", "train")[idx], load_split("y_continuous", "train")[idx]
    X_test, y_test = load_split("X_delta", "test"), load_split("y_continuous", "test")

    print(f"Training Random Forest regressor on {len(y_train)} variants...")
    model = RandomForestRegressor(
        n_estimators=150, max_depth=15, min_samples_leaf=3, n_jobs=-1, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    n_mut_test = load_split("n_mutations", "test")

    metrics = {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        # More mutations -> more likely broken, so mutation count alone is a strong baseline.
        "mutation_count_baseline_spearman": float(spearmanr(-n_mut_test, y_test).statistic),
        "zero_shot_spearman": float(spearmanr(load_split("zero_shot_score", "test"), y_test).statistic),
        "regressor_spearman": float(spearmanr(y_pred, y_test).statistic),
        # Within a fixed mutation count, counting can't help: this is the signal beyond it.
        "regressor_spearman_by_n_mutations": {
            str(k): float(spearmanr(y_pred[n_mut_test == k], y_test[n_mut_test == k]).statistic)
            for k in range(1, 7)
        },
    }
    for name in ("mutation_count_baseline_spearman", "zero_shot_spearman", "regressor_spearman"):
        print(f"{name}: {metrics[name]:.4f}")

    PATHS.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
