from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"


def ensure_output_directories() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    numeric_features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [column for column in feature_frame.columns if column not in numeric_features]

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def plot_churn_distribution(df: pd.DataFrame) -> None:
    counts = df["Churn"].value_counts().sort_index()
    ax = counts.plot(kind="bar", color=["#4C72B0", "#DD8452"], figsize=(6, 4))
    ax.set_title("Customer Churn Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Customer Count")
    ax.bar_label(ax.containers[0], padding=3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "churn_distribution.png", dpi=200)
    plt.close()


def plot_contract_churn(df: pd.DataFrame) -> None:
    contract_churn = (
        df.groupby(["Contract", "Churn"]).size().unstack(fill_value=0)[["No", "Yes"]]
    )
    contract_rate = contract_churn.div(contract_churn.sum(axis=1), axis=0) * 100

    ax = contract_rate.plot(kind="bar", stacked=True, figsize=(8, 5), color=["#55A868", "#C44E52"])
    ax.set_title("Churn Rate by Contract Type")
    ax.set_xlabel("Contract")
    ax.set_ylabel("Percent of Customers")
    ax.legend(title="Churn", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "contract_churn_rate.png", dpi=200)
    plt.close()


def build_models(preprocessor: ColumnTransformer, random_state: int = 42) -> tuple[GridSearchCV, Pipeline]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    knn_pipeline = Pipeline([("preprocess", preprocessor), ("model", KNeighborsClassifier())])
    knn_grid = GridSearchCV(
        estimator=knn_pipeline,
        param_grid={
            "model__n_neighbors": list(range(3, 26, 2)),
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )

    logistic_pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    C=10,
                    max_iter=2000,
                    random_state=random_state,
                    solver="liblinear",
                ),
            ),
        ]
    )
    return knn_grid, logistic_pipeline


def evaluate_model(model: Pipeline, features: pd.DataFrame, target: pd.Series) -> dict[str, object]:
    predicted_labels = model.predict(features)
    predicted_probabilities = model.predict_proba(features)[:, 1]
    metrics = {
        "accuracy": accuracy_score(target, predicted_labels),
        "precision": precision_score(target, predicted_labels),
        "recall": recall_score(target, predicted_labels),
        "f1": f1_score(target, predicted_labels),
        "roc_auc": roc_auc_score(target, predicted_probabilities),
        "confusion_matrix": confusion_matrix(target, predicted_labels).tolist(),
        "classification_report": classification_report(target, predicted_labels, output_dict=True),
    }
    return metrics


def plot_knn_search(knn_search: GridSearchCV) -> None:
    cv_results = pd.DataFrame(knn_search.cv_results_)
    summary = (
        cv_results.groupby("param_model__n_neighbors", observed=False)["mean_test_score"]
        .max()
        .reset_index()
        .rename(
            columns={
                "param_model__n_neighbors": "k",
                "mean_test_score": "mean_cv_roc_auc",
            }
        )
    )
    summary["k"] = summary["k"].astype(int)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(summary["k"], summary["mean_cv_roc_auc"], marker="o", color="#4C72B0")
    ax.axvline(knn_search.best_params_["model__n_neighbors"], color="#C44E52", linestyle="--", label="Selected k")
    ax.set_title("kNN Cross-Validated ROC AUC by k")
    ax.set_xlabel("Number of Neighbors (k)")
    ax.set_ylabel("Mean CV ROC AUC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "knn_cv_search.png", dpi=200)
    plt.close(fig)
    summary.to_csv(OUTPUT_DIR / "knn_cv_summary.csv", index=False)


def plot_confusion_matrices(
    y_test: pd.Series,
    knn_predictions: np.ndarray,
    logistic_predictions: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis, predictions, title in [
        (axes[0], knn_predictions, "kNN Confusion Matrix"),
        (axes[1], logistic_predictions, "Logistic Regression Confusion Matrix"),
    ]:
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            display_labels=["No churn", "Churn"],
            cmap="Blues",
            ax=axis,
            colorbar=False,
        )
        axis.set_title(title)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=200)
    plt.close()


def plot_roc_curves(
    y_test: pd.Series,
    knn_probabilities: np.ndarray,
    logistic_probabilities: np.ndarray,
) -> None:
    knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probabilities)
    logistic_fpr, logistic_tpr, _ = roc_curve(y_test, logistic_probabilities)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(knn_fpr, knn_tpr, label="kNN", color="#DD8452")
    ax.plot(logistic_fpr, logistic_tpr, label="Logistic regression", color="#4C72B0")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC Curve Comparison")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_output_directories()
    dataframe = load_data()
    plot_churn_distribution(dataframe)
    plot_contract_churn(dataframe)

    features = dataframe.drop(columns=["customerID", "Churn"])
    target = dataframe["Churn"].map({"No": 0, "Yes": 1})
    preprocessor = build_preprocessor(features)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    knn_search, logistic_pipeline = build_models(preprocessor)
    knn_search.fit(x_train, y_train)
    logistic_pipeline.fit(x_train, y_train)

    knn_metrics = evaluate_model(knn_search.best_estimator_, x_test, y_test)
    logistic_metrics = evaluate_model(logistic_pipeline, x_test, y_test)

    knn_predictions = knn_search.best_estimator_.predict(x_test)
    logistic_predictions = logistic_pipeline.predict(x_test)
    knn_probabilities = knn_search.best_estimator_.predict_proba(x_test)[:, 1]
    logistic_probabilities = logistic_pipeline.predict_proba(x_test)[:, 1]

    plot_knn_search(knn_search)
    plot_confusion_matrices(y_test, knn_predictions, logistic_predictions)
    plot_roc_curves(y_test, knn_probabilities, logistic_probabilities)

    comparison_table = pd.DataFrame(
        {
            "metric": ["accuracy", "precision", "recall", "f1", "roc_auc"],
            "knn": [knn_metrics["accuracy"], knn_metrics["precision"], knn_metrics["recall"], knn_metrics["f1"], knn_metrics["roc_auc"]],
            "logistic_regression": [
                logistic_metrics["accuracy"],
                logistic_metrics["precision"],
                logistic_metrics["recall"],
                logistic_metrics["f1"],
                logistic_metrics["roc_auc"],
            ],
        }
    )
    comparison_table.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    results = {
        "dataset": {
            "rows": int(dataframe.shape[0]),
            "columns": int(dataframe.shape[1]),
            "missing_total_charges": int(dataframe["TotalCharges"].isna().sum()),
            "churn_rate": float(target.mean()),
        },
        "train_test_split": {
            "train_rows": int(x_train.shape[0]),
            "test_rows": int(x_test.shape[0]),
            "test_size": 0.2,
            "random_state": 42,
        },
        "evaluation_framework": {
            "cross_validation": "5-fold stratified cross-validation on the training set",
            "selection_metric": "ROC AUC",
            "holdout_metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
        },
        "knn": {
            "best_params": knn_search.best_params_,
            "best_cv_roc_auc": knn_search.best_score_,
            "test_metrics": knn_metrics,
        },
        "logistic_regression": {
            "configuration": {"solver": "liblinear", "C": 10, "max_iter": 2000},
            "test_metrics": logistic_metrics,
        },
    }

    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
