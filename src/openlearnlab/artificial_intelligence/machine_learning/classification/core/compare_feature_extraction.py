import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..reporting.templates import FeatureExtractionComparisonTemplate


def compare_feature_extraction(
    df: pd.DataFrame,
    features,
    target: str,
    models: dict,
    dataset_name: str,
    output: Path,
    random_state: int,
    variance_threshold: float = 0.95,
):
    if not 0 < variance_threshold < 1:
        raise ValueError("variance_threshold must be between 0 and 1.")

    if not models:
        raise ValueError("At least one model is required.")

    if features is None:
        features = (
            df.drop(columns=[target]).select_dtypes(include="number").columns.tolist()
        )

    features = list(features)

    if not features:
        raise ValueError("At least one feature is required.")

    X = df[features]
    y = df[target]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    splits = list(cv.split(X, y))

    results = []

    # -----------------------------------
    # Model Comparison
    # -----------------------------------

    for model_name, model in models.items():

        versions = {
            "Original": clone(model),
            "PCA": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "pca",
                        PCA(
                            n_components=variance_threshold,
                            svd_solver="full",
                        ),
                    ),
                    ("model", clone(model)),
                ]
            ),
        }

        model_results = {}

        for version, estimator in versions.items():

            # -----------------------------------
            # Cross Validation
            # -----------------------------------

            scores = cross_validate(
                estimator,
                X,
                y,
                cv=splits,
                scoring={
                    "accuracy": "accuracy",
                    "f1": "f1_weighted",
                },
                return_train_score=True,
                return_estimator=True,
            )

            train_accuracy = scores["train_accuracy"]
            test_accuracy = scores["test_accuracy"]

            train_f1 = scores["train_f1"]
            test_f1 = scores["test_f1"]

            train_accuracy_mean = float(train_accuracy.mean())
            test_accuracy_mean = float(test_accuracy.mean())

            train_f1_mean = float(train_f1.mean())
            test_f1_mean = float(test_f1.mean())

            fit_times = scores["fit_time"]

            fit_time_mean = float(fit_times.mean())
            fit_time_std = float(fit_times.std())

            # -----------------------------------
            # Prediction Time
            # Reuse fitted CV estimators
            # -----------------------------------

            predict_times = []

            for fold_model, (_, test_index) in zip(
                scores["estimator"],
                splits,
            ):
                X_test = X.iloc[test_index]

                # Warm-up
                fold_model.predict(X_test.iloc[:1])

                start = time.perf_counter()

                fold_model.predict(X_test)

                elapsed = time.perf_counter() - start

                predict_times.append(elapsed)

            predict_time_mean = float(np.mean(predict_times))
            predict_time_std = float(np.std(predict_times))

            # -----------------------------------
            # Number of Features
            # -----------------------------------

            if version == "PCA":

                component_counts = [
                    fold_model.named_steps["pca"].n_components_
                    for fold_model in scores["estimator"]
                ]

                if len(set(component_counts)) == 1:
                    feature_count = str(component_counts[0])
                else:
                    feature_count = (
                        f"{min(component_counts)}–" f"{max(component_counts)}"
                    )

            else:
                feature_count = str(len(features))

            # -----------------------------------
            # Gaps
            # -----------------------------------

            accuracy_gap = train_accuracy_mean - test_accuracy_mean
            f1_gap = train_f1_mean - test_f1_mean

            model_results[version] = {
                "Model": model_name,
                "Dataset Version": version,
                "Features": feature_count,
                "Accuracy": (
                    f"Train: {train_accuracy_mean:.3f} ± {train_accuracy.std():.3f}\n"
                    f"Test: {test_accuracy_mean:.3f} ± {test_accuracy.std():.3f}\n"
                    f"Gap: {accuracy_gap:.3f}"
                ),
                "F1 Weighted": (
                    f"Train: {train_f1_mean:.3f} ± {train_f1.std():.3f}\n"
                    f"Test: {test_f1_mean:.3f} ± {test_f1.std():.3f}\n"
                    f"Gap: {f1_gap:.3f}"
                ),
                "Fit / Predict Time (s)": (
                    f"Fit: {fit_time_mean:.4f} ± {fit_time_std:.4f}\n"
                    f"Predict: {predict_time_mean:.4f} ± "
                    f"{predict_time_std:.4f}"
                ),
                # Internal values
                "_Test Accuracy": test_accuracy_mean,
                "_Test F1": test_f1_mean,
                "_Fit Time": fit_time_mean,
                "_Predict Time": predict_time_mean,
            }

        # -----------------------------------
        # Original Baseline
        # -----------------------------------

        original = model_results["Original"]

        original_accuracy = original["_Test Accuracy"]
        original_f1 = original["_Test F1"]

        original_fit_time = original["_Fit Time"]
        original_predict_time = original["_Predict Time"]

        # -----------------------------------
        # Relative Loss and Speed
        # -----------------------------------

        for version in ("Original", "PCA"):

            row = model_results[version]

            # -----------------------------------
            # Test Accuracy Relative Loss
            # -----------------------------------

            accuracy_loss = original_accuracy - row["_Test Accuracy"]

            accuracy_base = abs(original_accuracy)

            accuracy_loss_percent = (
                100 * accuracy_loss / accuracy_base
                if accuracy_base > 0
                else 0.0 if accuracy_loss == 0 else np.nan
            )

            row["Test Accuracy Loss"] = (
                f"{accuracy_loss_percent:+.2f}% ({accuracy_loss:+.3f})"
            )

            # -----------------------------------
            # Test F1 Relative Loss
            # -----------------------------------

            f1_loss = original_f1 - row["_Test F1"]

            f1_loss_percent = (
                100 * f1_loss / original_f1
                if original_f1 > 0
                else 0.0 if f1_loss == 0 else np.nan
            )

            row["Test F1 Loss"] = f"{f1_loss_percent:+.2f}% ({f1_loss:+.3f})"

            # -----------------------------------
            # Fit Speed vs Original
            # -----------------------------------

            current_fit_time = row["_Fit Time"]

            if original_fit_time > 0 and current_fit_time > 0:

                if current_fit_time <= original_fit_time:
                    fit_speed = original_fit_time / current_fit_time
                else:
                    fit_speed = -(current_fit_time / original_fit_time)

            else:
                fit_speed = np.nan

            # -----------------------------------
            # Predict Speed vs Original
            # -----------------------------------

            current_predict_time = row["_Predict Time"]

            if original_predict_time > 0 and current_predict_time > 0:

                if current_predict_time <= original_predict_time:
                    predict_speed = original_predict_time / current_predict_time
                else:
                    predict_speed = -(current_predict_time / original_predict_time)

            else:
                predict_speed = np.nan

            row["Speed vs Original"] = (
                f"Predict: {predict_speed:.2f}x\n" f"Fit: {fit_speed:.2f}x"
            )

            results.append(row)

    # -----------------------------------
    # Results DataFrame
    # -----------------------------------

    result = pd.DataFrame(results)

    # -----------------------------------
    # Final Column Order
    # -----------------------------------

    result = result[
        [
            "Model",
            "Dataset Version",
            "Features",
            "Accuracy",
            "F1 Weighted",
            "Test Accuracy Loss",
            "Test F1 Loss",
            "Fit / Predict Time (s)",
            "Speed vs Original",
        ]
    ]

    # -----------------------------------
    # Create Output Directory
    # -----------------------------------

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    # -----------------------------------
    # Scaling for Exported Dataset
    # -----------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # -----------------------------------
    # PCA for Exported Dataset
    # -----------------------------------

    pca = PCA(
        n_components=variance_threshold,
        svd_solver="full",
    )

    X_pca = pca.fit_transform(X_scaled)

    # -----------------------------------
    # Extracted Features
    # -----------------------------------

    component_names = [f"PC{i + 1}" for i in range(pca.n_components_)]

    extracted_df = pd.DataFrame(
        X_pca,
        columns=component_names,
        index=df.index,
    )

    extracted_df[target] = df[target]

    # -----------------------------------
    # Save Extracted Dataset
    # -----------------------------------

    extracted_df.to_csv(
        output / "extracted_dataset.csv",
        index=False,
    )

    # -----------------------------------
    # HTML Report
    # -----------------------------------

    template = FeatureExtractionComparisonTemplate(
        dataset_name=dataset_name,
        results=result,
        n_samples=len(df),
        n_features=len(features),
        variance_threshold=variance_threshold,
        n_folds=cv.get_n_splits(),
    )

    return template.save(output=output)
