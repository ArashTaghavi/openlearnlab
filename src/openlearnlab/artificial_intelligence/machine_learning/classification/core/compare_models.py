import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from ..reporting.templates import CompareModelsTemplate


def compare_models(
    df: pd.DataFrame,
    features,
    target: str,
    models: dict,
    dataset_name: str,
    output: Path,
    random_state: int,
):
    features = list(features)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    X = df[features]
    y = df[target]

    results = {}

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    for name, model in models.items():

        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
        )

        # Accuracy
        train_accuracy_mean = scores["train_accuracy"].mean()
        test_accuracy_mean = scores["test_accuracy"].mean()

        train_accuracy_std = scores["train_accuracy"].std()
        test_accuracy_std = scores["test_accuracy"].std()

        accuracy_gap = train_accuracy_mean - test_accuracy_mean

        # Precision
        train_precision_mean = scores["train_precision"].mean()
        test_precision_mean = scores["test_precision"].mean()

        train_precision_std = scores["train_precision"].std()
        test_precision_std = scores["test_precision"].std()

        precision_gap = train_precision_mean - test_precision_mean

        # Recall
        train_recall_mean = scores["train_recall"].mean()
        test_recall_mean = scores["test_recall"].mean()

        train_recall_std = scores["train_recall"].std()
        test_recall_std = scores["test_recall"].std()

        recall_gap = train_recall_mean - test_recall_mean

        # F1
        train_f1_mean = scores["train_f1"].mean()
        test_f1_mean = scores["test_f1"].mean()

        train_f1_std = scores["train_f1"].std()
        test_f1_std = scores["test_f1"].std()

        f1_gap = train_f1_mean - test_f1_mean

        # ROC AUC
        train_roc_auc_mean = scores["train_roc_auc"].mean()
        test_roc_auc_mean = scores["test_roc_auc"].mean()

        train_roc_auc_std = scores["train_roc_auc"].std()
        test_roc_auc_std = scores["test_roc_auc"].std()

        roc_auc_gap = train_roc_auc_mean - test_roc_auc_mean

        # Fit time
        fit_time_mean = scores["fit_time"].mean()
        fit_time_std = scores["fit_time"].std()

        # Prediction time
        predict_times = []

        for train_index, test_index in cv.split(X, y):
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            X_test = X.iloc[test_index]

            fold_model = clone(model)
            fold_model.fit(X_train, y_train)

            # Warm-up
            fold_model.predict(X_test.iloc[:1])

            start = time.perf_counter()
            fold_model.predict(X_test)
            elapsed = time.perf_counter() - start

            predict_times.append(elapsed)

        predict_time_mean = float(np.mean(predict_times))
        predict_time_std = float(np.std(predict_times))

        results[name] = {
            "Accuracy": (
                f"Train: {train_accuracy_mean:.3f} ± {train_accuracy_std:.3f}\n"
                f"Test: {test_accuracy_mean:.3f} ± {test_accuracy_std:.3f}\n"
                f"Gap: {accuracy_gap:.3f}"
            ),
            "Precision": (
                f"Train: {train_precision_mean:.3f} ± {train_precision_std:.3f}\n"
                f"Test: {test_precision_mean:.3f} ± {test_precision_std:.3f}\n"
                f"Gap: {precision_gap:.3f}"
            ),
            "Recall": (
                f"Train: {train_recall_mean:.3f} ± {train_recall_std:.3f}\n"
                f"Test: {test_recall_mean:.3f} ± {test_recall_std:.3f}\n"
                f"Gap: {recall_gap:.3f}"
            ),
            "F1": (
                f"Train: {train_f1_mean:.3f} ± {train_f1_std:.3f}\n"
                f"Test: {test_f1_mean:.3f} ± {test_f1_std:.3f}\n"
                f"Gap: {f1_gap:.3f}"
            ),
            "ROC AUC": (
                f"Train: {train_roc_auc_mean:.3f} ± {train_roc_auc_std:.3f}\n"
                f"Test: {test_roc_auc_mean:.3f} ± {test_roc_auc_std:.3f}\n"
                f"Gap: {roc_auc_gap:.3f}"
            ),
            "_Test Accuracy Mean": test_accuracy_mean,
            "_Test Precision Mean": test_precision_mean,
            "_Test Recall Mean": test_recall_mean,
            "_Test F1 Mean": test_f1_mean,
            "_Test ROC AUC Mean": test_roc_auc_mean,
            "_Fit Time Mean": fit_time_mean,
            "_Fit Time Std": fit_time_std,
            "_Predict Time Mean": predict_time_mean,
            "_Predict Time Std": predict_time_std,
        }

    result = pd.DataFrame(results).T
    result.index.name = "Model"

    result = result.sort_values(
        by=[
            "_Test Accuracy Mean",
            "_Test ROC AUC Mean",
            "_Predict Time Mean",
            "_Fit Time Mean",
        ],
        ascending=[False, False, True, True],
    )

    # Best model = first row
    best_model = str(result.index[0])

    best_test_accuracy = float(result.iloc[0]["_Test Accuracy Mean"])
    best_test_precision = float(result.iloc[0]["_Test Precision Mean"])
    best_test_recall = float(result.iloc[0]["_Test Recall Mean"])
    best_test_f1 = float(result.iloc[0]["_Test F1 Mean"])
    best_test_roc_auc = float(result.iloc[0]["_Test ROC AUC Mean"])

    best_fit_time = float(result.iloc[0]["_Fit Time Mean"])
    best_predict_time = float(result.iloc[0]["_Predict Time Mean"])

    # Accuracy loss
    accuracy_values = result["_Test Accuracy Mean"].astype(float)
    accuracy_loss = best_test_accuracy - accuracy_values

    accuracy_loss_percent = (
        accuracy_loss / best_test_accuracy * 100
        if best_test_accuracy != 0
        else pd.Series(0.0, index=result.index)
    )

    accuracy_loss_values = [
        f"{percent:.2f}% ({loss:.3f})"
        for percent, loss in zip(accuracy_loss_percent, accuracy_loss)
    ]

    # Precision loss
    precision_values = result["_Test Precision Mean"].astype(float)
    precision_loss = best_test_precision - precision_values

    precision_loss_percent = (
        precision_loss / best_test_precision * 100
        if best_test_precision != 0
        else pd.Series(0.0, index=result.index)
    )

    precision_loss_values = [
        f"{percent:.2f}% ({loss:.3f})"
        for percent, loss in zip(precision_loss_percent, precision_loss)
    ]

    # Recall loss
    recall_values = result["_Test Recall Mean"].astype(float)
    recall_loss = best_test_recall - recall_values

    recall_loss_percent = (
        recall_loss / best_test_recall * 100
        if best_test_recall != 0
        else pd.Series(0.0, index=result.index)
    )

    recall_loss_values = [
        f"{percent:.2f}% ({loss:.3f})"
        for percent, loss in zip(recall_loss_percent, recall_loss)
    ]

    # F1 loss
    f1_values = result["_Test F1 Mean"].astype(float)
    f1_loss = best_test_f1 - f1_values

    f1_loss_percent = (
        f1_loss / best_test_f1 * 100
        if best_test_f1 != 0
        else pd.Series(0.0, index=result.index)
    )

    f1_loss_values = [
        f"{percent:.2f}% ({loss:.3f})"
        for percent, loss in zip(f1_loss_percent, f1_loss)
    ]

    # Combined Metrics Loss
    result["Test Metrics Loss"] = [
        (
            f"Accuracy: {accuracy}\n"
            f"Precision: {precision}\n"
            f"Recall: {recall}\n"
            f"F1: {f1}"
        )
        for accuracy, precision, recall, f1 in zip(
            accuracy_loss_values,
            precision_loss_values,
            recall_loss_values,
            f1_loss_values,
        )
    ]

    # ROC AUC loss
    roc_auc_values = result["_Test ROC AUC Mean"].astype(float)
    roc_auc_loss = best_test_roc_auc - roc_auc_values

    roc_auc_loss_percent = (
        roc_auc_loss / best_test_roc_auc * 100
        if best_test_roc_auc != 0
        else pd.Series(0.0, index=result.index)
    )

    result["Test ROC AUC Loss"] = [
        f"{percent:.2f}% ({loss:.3f})"
        for percent, loss in zip(roc_auc_loss_percent, roc_auc_loss)
    ]

    # Fit / Predict time
    time_values = []

    for index in result.index:
        fit_mean = float(result.loc[index, "_Fit Time Mean"])
        fit_std = float(result.loc[index, "_Fit Time Std"])

        predict_mean = float(result.loc[index, "_Predict Time Mean"])
        predict_std = float(result.loc[index, "_Predict Time Std"])

        time_values.append(
            f"Fit: {fit_mean:.3f} ± {fit_std:.3f}\n"
            f"Predict: {predict_mean:.3f} ± {predict_std:.3f}"
        )

    result["Fit / Predict Time (s)"] = time_values

    # Fit / Predict speed vs Best
    speed_values = []

    for index in result.index:
        fit_time = float(result.loc[index, "_Fit Time Mean"])
        predict_time = float(result.loc[index, "_Predict Time Mean"])

        if fit_time == 0 or best_fit_time == 0:
            fit_speed = 1.0 if fit_time == best_fit_time else np.nan
        elif fit_time <= best_fit_time:
            fit_speed = best_fit_time / fit_time
        else:
            fit_speed = -(fit_time / best_fit_time)

        if predict_time == 0 or best_predict_time == 0:
            predict_speed = 1.0 if predict_time == best_predict_time else np.nan
        elif predict_time <= best_predict_time:
            predict_speed = best_predict_time / predict_time
        else:
            predict_speed = -(predict_time / best_predict_time)

        speed_values.append(f"Predict: {predict_speed:.2f}x\n" f"Fit: {fit_speed:.2f}x")

    result["Speed vs Best"] = speed_values

    # Final column order
    result = result[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC AUC",
            "Test Metrics Loss",
            "Test ROC AUC Loss",
            "Fit / Predict Time (s)",
            "Speed vs Best",
            "_Test Accuracy Mean",
            "_Test Precision Mean",
            "_Test Recall Mean",
            "_Test F1 Mean",
            "_Test ROC AUC Mean",
            "_Fit Time Mean",
            "_Fit Time Std",
            "_Predict Time Mean",
            "_Predict Time Std",
        ]
    ]

    # Remove internal columns
    result = result.drop(
        columns=[
            "_Test Accuracy Mean",
            "_Test Precision Mean",
            "_Test Recall Mean",
            "_Test F1 Mean",
            "_Test ROC AUC Mean",
            "_Fit Time Mean",
            "_Fit Time Std",
            "_Predict Time Mean",
            "_Predict Time Std",
        ]
    )

    template = CompareModelsTemplate(
        dataset_name=dataset_name,
        results=result,
        best_model=best_model,
        n_samples=len(df),
        n_features=len(features),
        n_folds=cv.get_n_splits(),
    )
    return template.save(output=output)
