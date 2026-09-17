from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PUBLIC_DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/telco-customer-churn-by-IBM.csv"
SEED = 42
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate kNN and logistic regression analysis artifacts for the Telco churn data."
    )
    parser.add_argument(
        "--input",
        default=PUBLIC_DATA_URL,
        help="Path or URL to the Excel/CSV source file. Defaults to the public IBM Telco churn CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/figures",
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--summary-path",
        default="reports/model_metrics.json",
        help="Path for the generated JSON summary.",
    )
    return parser.parse_args()


def load_dataset(source: str) -> pd.DataFrame:
    if source.lower().endswith((".xlsx", ".xls")):
        data = pd.read_excel(source)
    else:
        data = pd.read_csv(source)

    data = data.copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})
    return data


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [column for column in features.columns if column not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def evaluate_model(name: str, model: Pipeline, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series, cv: StratifiedKFold) -> dict:
    cv_results = cross_validate(model, x_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    return {
        "name": name,
        "cv_mean": {metric: float(np.mean(cv_results[f"test_{metric}"])) for metric in SCORING},
        "test": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions)),
            "recall": float(recall_score(y_test, predictions)),
            "f1": float(f1_score(y_test, predictions)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        },
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "roc_curve": {
            "fpr": roc_curve(y_test, probabilities)[0].tolist(),
            "tpr": roc_curve(y_test, probabilities)[1].tolist(),
        },
    }


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_class_balance(data: pd.DataFrame, figure_dir: Path) -> None:
    counts = data["Churn"].map({0: "No churn", 1: "Churn"}).value_counts().rename_axis("Class").reset_index(name="Count")
    counts["Percent"] = counts["Count"] / counts["Count"].sum() * 100

    plt.figure(figsize=(7, 4))
    ax = sns.barplot(data=counts, x="Class", y="Count", hue="Class", palette="Blues", legend=False)
    for patch, percent in zip(ax.patches, counts["Percent"]):
        ax.annotate(f"{percent:.1f}%", (patch.get_x() + patch.get_width() / 2, patch.get_height()), ha="center", va="bottom")
    ax.set_title("Class Balance in the Telco Churn Data")
    ax.set_ylabel("Customers")
    plt.tight_layout()
    plt.savefig(figure_dir / "class_balance.png", dpi=200)
    plt.close()


def plot_contract_churn(data: pd.DataFrame, figure_dir: Path) -> None:
    contract_rates = (
        data.assign(ChurnRate=data["Churn"])
        .groupby("Contract", as_index=False)["ChurnRate"]
        .mean()
        .sort_values("ChurnRate", ascending=False)
    )
    contract_rates["ChurnRate"] *= 100

    plt.figure(figsize=(8, 4.5))
    ax = sns.barplot(data=contract_rates, x="Contract", y="ChurnRate", hue="Contract", palette="viridis", legend=False)
    for patch, value in zip(ax.patches, contract_rates["ChurnRate"]):
        ax.annotate(f"{value:.1f}%", (patch.get_x() + patch.get_width() / 2, value), ha="center", va="bottom")
    ax.set_title("Observed Churn Rate by Contract Type")
    ax.set_ylabel("Churn rate (%)")
    ax.set_xlabel("Contract")
    plt.tight_layout()
    plt.savefig(figure_dir / "contract_churn_rate.png", dpi=200)
    plt.close()


def plot_knn_tuning(grid: GridSearchCV, figure_dir: Path) -> None:
    results = pd.DataFrame(grid.cv_results_)
    tuning_curve = results.groupby("param_model__n_neighbors", as_index=False)["mean_test_score"].max()
    tuning_curve["param_model__n_neighbors"] = tuning_curve["param_model__n_neighbors"].astype(int)

    plt.figure(figsize=(8, 4.5))
    sns.lineplot(data=tuning_curve, x="param_model__n_neighbors", y="mean_test_score", marker="o")
    plt.title("kNN Hyperparameter Search: Best Cross-Validated F1 by k")
    plt.xlabel("Number of neighbors (k)")
    plt.ylabel("Mean cross-validated F1")
    plt.tight_layout()
    plt.savefig(figure_dir / "knn_tuning_curve.png", dpi=200)
    plt.close()


def plot_confusion_matrices(results: Iterable[dict], figure_dir: Path) -> None:
    results = list(results)
    fig, axes = plt.subplots(1, len(results), figsize=(10, 4))
    if len(results) == 1:
        axes = [axes]

    for axis, result in zip(axes, results):
        matrix = np.array(result["confusion_matrix"])
        display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["No churn", "Churn"])
        display.plot(ax=axis, colorbar=False)
        axis.set_title(f"{result['name']} confusion matrix")
    plt.tight_layout()
    plt.savefig(figure_dir / "confusion_matrices.png", dpi=200)
    plt.close()


def plot_metric_comparison(results: Iterable[dict], figure_dir: Path) -> None:
    results = list(results)
    metric_table = pd.DataFrame(
        [
            {"Model": result["name"], "Metric": metric, "Score": result["test"][metric]}
            for result in results
            for metric in SCORING
        ]
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(data=metric_table, x="Metric", y="Score", hue="Model", palette="Set2")
    plt.title("Holdout-Test Metric Comparison")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(figure_dir / "metric_comparison.png", dpi=200)
    plt.close()


def plot_roc_curves(results: Iterable[dict], figure_dir: Path) -> None:
    plt.figure(figsize=(7, 5))
    for result in results:
        plt.plot(result["roc_curve"]["fpr"], result["roc_curve"]["tpr"], label=f"{result['name']} (AUC={result['test']['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.title("ROC Curve Comparison")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "roc_curves.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    figure_dir = Path(args.output_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_path)
    ensure_parent(summary_path)

    sns.set_theme(style="whitegrid")

    data = load_dataset(args.input)
    features = data.drop(columns=["Churn", "customerID"])
    target = data["Churn"]

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    preprocessor = build_preprocessor(features)

    knn_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", KNeighborsClassifier())]
    )
    knn_grid = GridSearchCV(
        estimator=knn_pipeline,
        param_grid={
            "model__n_neighbors": list(range(3, 32, 2)),
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )
    knn_grid.fit(x_train, y_train)

    logistic_regression = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=2000, solver="liblinear", random_state=SEED)),
        ]
    )

    knn_results = evaluate_model("kNN", knn_grid.best_estimator_, x_train, y_train, x_test, y_test, cv)
    logistic_results = evaluate_model("Logistic regression", logistic_regression, x_train, y_train, x_test, y_test, cv)

    summary = {
        "data_source": args.input,
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "missing_total_charges": int(data["TotalCharges"].isna().sum()),
        "class_distribution": {
            "no_churn": int((target == 0).sum()),
            "churn": int((target == 1).sum()),
        },
        "best_knn_parameters": knn_grid.best_params_,
        "models": [knn_results, logistic_results],
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_class_balance(data, figure_dir)
    plot_contract_churn(data, figure_dir)
    plot_knn_tuning(knn_grid, figure_dir)
    plot_confusion_matrices([knn_results, logistic_results], figure_dir)
    plot_metric_comparison([knn_results, logistic_results], figure_dir)
    plot_roc_curves([knn_results, logistic_results], figure_dir)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
