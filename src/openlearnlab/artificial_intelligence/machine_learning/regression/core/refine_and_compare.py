from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

from ..reporting.templates import RefineAndCompareTemplate

N_FOLDS = 5
STRATEGIES = (
    "Original",
    "Outlier Only",
    "High Error Only",
    "High Error + Outlier",
    "Top 5% Errors",
    "Top 10% Errors",
)


def _metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if len(actual) == 0:
        return {"R²": np.nan, "RMSE": np.nan, "MAE": np.nan}
    if len(actual) != len(predicted) or not np.isfinite(predicted).all():
        raise ValueError("Predictions must be finite and match the target length.")
    return {
        "R²": float(r2_score(actual, predicted)) if len(actual) >= 2 else np.nan,
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE": float(mean_absolute_error(actual, predicted)),
    }


def _detect(training, features, target, model, random_state):
    """Fit-free OOF error labels and target bounds, learned from training only."""
    if len(training) < N_FOLDS:
        raise ValueError(f"At least {N_FOLDS} training records are required.")
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)
    actual = training[target].to_numpy(dtype=float)
    predicted = np.asarray(
        cross_val_predict(clone(model), training[features], training[target], cv=cv),
        dtype=float,
    ).reshape(-1)
    if len(predicted) != len(actual) or not np.isfinite(predicted).all():
        raise ValueError("OOF predictions must be finite and one-dimensional.")
    errors = np.abs(actual - predicted)
    mae = float(errors.mean())
    q1, q3 = np.quantile(actual, [0.25, 0.75])
    iqr = float(q3 - q1)
    return {
        "errors": errors,
        "high_error_threshold": 1.5 * mae,
        "lower": float(q1 - 1.5 * iqr),
        "upper": float(q3 + 1.5 * iqr),
    }


def _train_masks(actual, detection):
    errors = detection["errors"]
    count = len(errors)
    high = errors > detection["high_error_threshold"]
    outlier = (actual < detection["lower"]) | (actual > detection["upper"])
    masks = {
        "Original": np.zeros(count, dtype=bool),
        "Outlier Only": outlier,
        "High Error Only": high,
        "High Error + Outlier": high | outlier,
    }
    order = np.argsort(-errors, kind="stable")
    for pct in (5, 10):
        mask = np.zeros(count, dtype=bool)
        mask[order[: int(np.floor(count * pct / 100))]] = True
        masks[f"Top {pct}% Errors"] = mask
    return masks


def _test_masks(actual, baseline_predictions, detection):
    """Post-hoc test subsets. Uses test targets ONLY for reporting, never fitting."""
    errors = np.abs(actual - baseline_predictions)
    high = errors > detection["high_error_threshold"]
    outlier = (actual < detection["lower"]) | (actual > detection["upper"])
    masks = {
        "Original": np.zeros(len(actual), dtype=bool),
        "Outlier Only": outlier,
        "High Error Only": high,
        "High Error + Outlier": high | outlier,
    }
    # Error cutoffs are learned exclusively from the outer training fold.
    train_errors = detection["errors"]
    for pct in (5, 10):
        n = int(np.floor(len(train_errors) * pct / 100))
        if n == 0:
            masks[f"Top {pct}% Errors"] = np.zeros(len(actual), dtype=bool)
        else:
            cutoff = float(np.sort(train_errors)[-n])
            masks[f"Top {pct}% Errors"] = errors >= cutoff
    return masks


def refine_and_compare(
    df: pd.DataFrame,
    features,
    target: str,
    model,
    dataset_name: str,
    output: Path,
    random_state: int = 42,
) -> Path:
    """Compare fold-local refinement on original and post-hoc refined test subsets.

    Original-test scores support model selection. Refined-test scores are conditional
    descriptive results: the test subset uses observed test targets/residuals.
    """
    features = list(features)
    if not features or len(features) != len(set(features)):
        raise ValueError("features must contain unique feature names.")
    if target in features:
        raise ValueError("The target cannot be included in features.")
    missing = [name for name in [*features, target] if name not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if len(df) < 7:
        raise ValueError(
            "At least 7 records are required for nested 5-fold evaluation."
        )
    if df.index.has_duplicates:
        raise ValueError("DataFrame index must be unique.")
    if not np.isfinite(df[target].to_numpy(dtype=float)).all():
        raise ValueError("Target values must be finite.")

    outer = KFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(outer.split(df), start=1):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        detection = _detect(train, features, target, model, random_state)
        train_actual = train[target].to_numpy(dtype=float)
        test_actual = test[target].to_numpy(dtype=float)
        train_masks = _train_masks(train_actual, detection)

        # The baseline test prediction is independent of every test target.
        baseline = clone(model).fit(train[features], train[target])
        baseline_test = np.asarray(
            baseline.predict(test[features]), dtype=float
        ).reshape(-1)
        if len(baseline_test) != len(test) or not np.isfinite(baseline_test).all():
            raise ValueError("Baseline test predictions must be finite.")
        test_masks = _test_masks(test_actual, baseline_test, detection)

        for strategy in STRATEGIES:
            removed_train = train_masks[strategy]
            selected = train.iloc[np.flatnonzero(~removed_train)]
            if len(selected) < 2:
                raise ValueError(
                    f"Too few training records for {strategy} in fold {fold}."
                )
            fitted = (
                baseline
                if strategy == "Original"
                else clone(model).fit(selected[features], selected[target])
            )
            train_pred = np.asarray(
                fitted.predict(selected[features]), dtype=float
            ).reshape(-1)
            test_pred = np.asarray(fitted.predict(test[features]), dtype=float).reshape(
                -1
            )
            train_scores = _metrics(selected[target], train_pred)
            evaluations = [
                (
                    (
                        "Original → Original"
                        if strategy == "Original"
                        else "Refined → Original"
                    ),
                    np.ones(len(test), dtype=bool),
                )
            ]
            if strategy != "Original":
                evaluations.append(("Refined → Refined", ~test_masks[strategy]))
            for evaluation, keep in evaluations:
                scores = _metrics(test_actual[keep], test_pred[keep])
                rows.append(
                    {
                        "Strategy": strategy,
                        "Evaluation": evaluation,
                        "Fold": fold,
                        "Train Records": len(selected),
                        "Test Records": int(keep.sum()),
                        "Removed Train Records": int(removed_train.sum()),
                        "Removed Train (%)": 100.0 * float(removed_train.mean()),
                        "Excluded Test Records": int((~keep).sum()),
                        **{
                            f"Train {key}": value for key, value in train_scores.items()
                        },
                        **{f"Test {key}": value for key, value in scores.items()},
                    }
                )

    # Full-data OOF detection is ONLY for exported datasets and summary, not CV scores.
    full = _detect(df, features, target, model, random_state)
    full_masks = _train_masks(df[target].to_numpy(dtype=float), full)
    combined = full_masks["High Error + Outlier"]
    high = full_masks["High Error Only"]
    outlier = full_masks["Outlier Only"]

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for strategy in STRATEGIES[1:]:
        filename = (
            strategy.lower()
            .replace(" + ", "_and_")
            .replace(" ", "_")
            .replace("%", "pct")
        )
        df.iloc[np.flatnonzero(~full_masks[strategy])].to_csv(
            output / f"{filename}_dataset.csv", index=True
        )
    df.iloc[np.flatnonzero(~combined)].to_csv(
        output / "refined_dataset.csv", index=True
    )
    df.iloc[np.flatnonzero(combined)].to_csv(output / "removed_records.csv", index=True)
    pd.DataFrame(
        {
            "Row Index": df.index,
            "Absolute OOF Error": full["errors"],
            "Is High Error": high,
            "Is Target Outlier": outlier,
            **{f"Remove: {name}": full_masks[name] for name in STRATEGIES[1:]},
        }
    ).to_csv(output / "refinement_decisions.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "fold_results.csv", index=False)

    template = RefineAndCompareTemplate(
        dataset_name=dataset_name,
        folds=pd.DataFrame(rows),
        n_samples=len(df),
        n_high_error=int(high.sum()),
        n_outlier=int(outlier.sum()),
        n_overlap=int((high & outlier).sum()),
        threshold=float(full["high_error_threshold"]),
        strategy_counts={name: int(mask.sum()) for name, mask in full_masks.items()},
        n_folds=N_FOLDS,
    )
    return template.save(output=output)
