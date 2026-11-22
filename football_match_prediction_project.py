"""
Football Match Outcome Prediction Pipeline

Usage:
    python football_match_prediction_project.py --csv path/to/dataset.csv --output outputs
"""

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
CLASS_ORDER = ["away", "draw", "home"]


def make_output_dirs(output_dir: str) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    plots_path = output_path / "plots"
    output_path.mkdir(parents=True, exist_ok=True)
    plots_path.mkdir(parents=True, exist_ok=True)
    return output_path, plots_path


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Make column names robust without changing their meaning."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def standardize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": np.nan, "nan": np.nan, "None": np.nan, "NONE": np.nan})
    return df


def standardize_binary_series(s: pd.Series) -> pd.Series:
    """Convert mixed boolean / numeric / string flags into 0/1 values."""
    mapped = (
        s.astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "true": "1", "t": "1", "yes": "1", "y": "1", "1.0": "1", "1": "1",
            "false": "0", "f": "0", "no": "0", "n": "0", "0.0": "0", "0": "0",
            "nan": np.nan, "none": np.nan, "": np.nan,
        })
    )
    return pd.to_numeric(mapped, errors="coerce")


def weighted_mean(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Weighted mean where suffix _1 is most recent and gets the highest weight."""
    if not cols:
        return pd.Series(np.nan, index=df.index)
    ordered = sort_recent_columns(cols)
    values = df[ordered].apply(pd.to_numeric, errors="coerce")
    n = len(ordered)
    weights = np.arange(n, 0, -1, dtype=float)  # _1 gets largest weight
    valid = values.notna().astype(float)
    numerator = values.fillna(0).mul(weights, axis=1).sum(axis=1)
    denominator = valid.mul(weights, axis=1).sum(axis=1).replace(0, np.nan)
    return numerator / denominator


def sort_recent_columns(cols: list[str]) -> list[str]:
    def suffix_num(col: str) -> int:
        m = re.search(r"_(\d+)$", col)
        return int(m.group(1)) if m else 999
    return sorted(cols, key=suffix_num)


def recent_cols(df: pd.DataFrame, side: str, feature: str) -> list[str]:
    prefix = f"{side}_club_recent_{feature}_"
    return sort_recent_columns([col for col in df.columns if col.startswith(prefix)])


def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "fixture_date" in df.columns:
        fixture_dt = pd.to_datetime(df["fixture_date"], errors="coerce")
        df["fixture_year"] = fixture_dt.dt.year
        df["fixture_month"] = fixture_dt.dt.month
        df["fixture_dayofweek"] = fixture_dt.dt.dayofweek
        df["fixture_is_weekend"] = fixture_dt.dt.dayofweek.isin([5, 6]).astype(float)
        # Football seasons usually cross calendar years. This is a compact season feature.
        df["fixture_season_start_year"] = np.where(
            fixture_dt.dt.month >= 7,
            fixture_dt.dt.year,
            fixture_dt.dt.year - 1,
        )
    else:
        fixture_dt = pd.Series(pd.NaT, index=df.index)

    for side in ["home", "away"]:
        date_cols = recent_cols(df, side, "fixture_date")
        if date_cols and "fixture_date" in df.columns:
            for col in date_cols:
                recent_dt = pd.to_datetime(df[col], errors="coerce")
                new_col = col.replace("fixture_date", "days_since")
                df[new_col] = (fixture_dt - recent_dt).dt.days

            first_recent = pd.to_datetime(df[date_cols[0]], errors="coerce")
            df[f"{side}_days_since_last_match"] = (fixture_dt - first_recent).dt.days

    # Drop raw date columns after extracting numeric date features.
    raw_date_cols = [col for col in df.columns if "fixture_date" in col]
    df = df.drop(columns=raw_date_cols, errors="ignore")
    return df


def create_recent_form_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for side in ["home", "away"]:
        goals_for_cols = recent_cols(df, side, "goals_scored")
        goals_against_cols = recent_cols(df, side, "goals_conceded")
        team_rating_cols = recent_cols(df, side, "team_rating")
        opp_rating_cols = recent_cols(df, side, "opponent_rating")
        played_home_cols = recent_cols(df, side, "played_at_home")
        cup_cols = recent_cols(df, side, "cup_game")

        # Recent goals and defensive form.
        if goals_for_cols:
            goals_for = df[goals_for_cols].apply(pd.to_numeric, errors="coerce")
            df[f"{side}_avg_goals_scored_recent"] = goals_for.mean(axis=1)
            df[f"{side}_weighted_goals_scored_recent"] = weighted_mean(df, goals_for_cols)
        if goals_against_cols:
            goals_against = df[goals_against_cols].apply(pd.to_numeric, errors="coerce")
            df[f"{side}_avg_goals_conceded_recent"] = goals_against.mean(axis=1)
            df[f"{side}_weighted_goals_conceded_recent"] = weighted_mean(df, goals_against_cols)

        if goals_for_cols and goals_against_cols:
            goals_for = df[goals_for_cols].apply(pd.to_numeric, errors="coerce")
            goals_against = df[goals_against_cols].apply(pd.to_numeric, errors="coerce")
            goal_diff = goals_for.to_numpy() - goals_against.to_numpy()
            goal_diff_df = pd.DataFrame(goal_diff, index=df.index, columns=[f"tmp_{i}" for i in range(goal_diff.shape[1])])
            df[f"{side}_avg_goal_diff_recent"] = goal_diff_df.mean(axis=1)

            # Recent points: win=3, draw=1, loss=0.
            points = np.where(goal_diff > 0, 3, np.where(goal_diff == 0, 1, 0)).astype(float)
            points_df = pd.DataFrame(points, index=df.index)
            both_missing = goals_for.isna().to_numpy() | goals_against.isna().to_numpy()
            points_df = points_df.mask(both_missing)
            df[f"{side}_avg_points_recent"] = points_df.mean(axis=1)
            df[f"{side}_win_rate_recent"] = (points_df == 3).mean(axis=1)
            df[f"{side}_draw_rate_recent"] = (points_df == 1).mean(axis=1)
            df[f"{side}_loss_rate_recent"] = (points_df == 0).mean(axis=1)

            weights = np.arange(points_df.shape[1], 0, -1, dtype=float)
            valid = points_df.notna().astype(float)
            df[f"{side}_weighted_points_recent"] = (
                points_df.fillna(0).mul(weights, axis=1).sum(axis=1)
                / valid.mul(weights, axis=1).sum(axis=1).replace(0, np.nan)
            )

        # Ratings and opponent strength.
        if team_rating_cols:
            team_rating = df[team_rating_cols].apply(pd.to_numeric, errors="coerce")
            df[f"{side}_avg_team_rating_recent"] = team_rating.mean(axis=1)
            df[f"{side}_weighted_team_rating_recent"] = weighted_mean(df, team_rating_cols)
        if opp_rating_cols:
            opp_rating = df[opp_rating_cols].apply(pd.to_numeric, errors="coerce")
            df[f"{side}_avg_opponent_rating_recent"] = opp_rating.mean(axis=1)
            df[f"{side}_weighted_opponent_rating_recent"] = weighted_mean(df, opp_rating_cols)
        if team_rating_cols and opp_rating_cols:
            team_rating = df[team_rating_cols].apply(pd.to_numeric, errors="coerce")
            opp_rating = df[opp_rating_cols].apply(pd.to_numeric, errors="coerce")
            rating_diff = team_rating.to_numpy() - opp_rating.to_numpy()
            rating_diff_df = pd.DataFrame(rating_diff, index=df.index)
            df[f"{side}_avg_rating_diff_recent"] = rating_diff_df.mean(axis=1)

        if played_home_cols:
            df[f"{side}_recent_home_rate"] = df[played_home_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        if cup_cols:
            df[f"{side}_recent_cup_rate"] = df[cup_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    # Home-vs-away difference features. These are often useful for match prediction.
    comparable_features = [
        "avg_goals_scored_recent",
        "avg_goals_conceded_recent",
        "weighted_goals_scored_recent",
        "weighted_goals_conceded_recent",
        "avg_goal_diff_recent",
        "avg_points_recent",
        "weighted_points_recent",
        "win_rate_recent",
        "draw_rate_recent",
        "loss_rate_recent",
        "avg_team_rating_recent",
        "weighted_team_rating_recent",
        "avg_opponent_rating_recent",
        "weighted_opponent_rating_recent",
        "avg_rating_diff_recent",
        "days_since_last_match",
    ]
    for feature in comparable_features:
        h = f"home_{feature}"
        a = f"away_{feature}"
        if h in df.columns and a in df.columns:
            df[f"home_minus_away_{feature}"] = df[h] - df[a]

    return df


def prepare_data(df: pd.DataFrame, output_path: Path) -> tuple[pd.DataFrame, pd.Series, dict]:
    summary = {}
    summary["original_rows"] = len(df)
    summary["original_columns"] = len(df.columns)

    df = normalize_column_names(df)
    df = standardize_string_columns(df)

    if "match_result" not in df.columns:
        raise ValueError("Target column 'match_result' was not found in the CSV file.")

    # Normalize target labels.
    df["match_result"] = df["match_result"].astype(str).str.strip().str.lower()
    valid_targets = set(CLASS_ORDER)
    invalid_target_mask = ~df["match_result"].isin(valid_targets)
    summary["removed_invalid_target"] = int(invalid_target_mask.sum())
    df = df.loc[~invalid_target_mask].copy()

    # Remove duplicate matches if match_id exists.
    if "match_id" in df.columns:
        duplicate_count = int(df.duplicated(subset=["match_id"]).sum())
        summary["removed_duplicate_match_id"] = duplicate_count
        df = df.drop_duplicates(subset=["match_id"], keep="first")
    else:
        summary["removed_duplicate_match_id"] = 0

    # Standardize current and recent binary flags.
    binary_patterns = ["cup_game", "played_at_home"]
    for col in df.columns:
        if any(pattern in col for pattern in binary_patterns):
            converted = standardize_binary_series(df[col])
            # Use converted if at least some values could be interpreted.
            if converted.notna().sum() > 0:
                df[col] = converted

    # Date features before dropping raw dates.
    df = create_date_features(df)

    # Recent-history aggregate features.
    df = create_recent_form_features(df)

    # Treat IDs and code columns as categorical identifiers, not continuous values.
    for col in df.columns:
        lowered = col.lower()
        if (
            col != "match_result"
            and ("manager_id" in lowered or lowered.endswith("competition_code") or lowered == "competition_code")
        ):
            df[col] = df[col].astype("object")

    # Drop non-predictive identifier.
    df = df.drop(columns=["match_id"], errors="ignore")

    # Replace infinite values created by feature engineering.
    df = df.replace([np.inf, -np.inf], np.nan)

    y = df["match_result"].copy()
    X = df.drop(columns=["match_result"])

    summary["final_rows"] = len(df)
    summary["final_features_before_encoding"] = X.shape[1]
    summary["target_distribution"] = y.value_counts().to_dict()
    summary["missing_values_after_feature_engineering"] = int(X.isna().sum().sum())

    # Save a small preview for checking.
    preview_path = output_path / "cleaned_data_preview.csv"
    pd.concat([X.head(20), y.head(20)], axis=1).to_csv(preview_path, index=False)

    return X, y, summary


def save_basic_report(raw_df: pd.DataFrame, output_path: Path) -> None:
    report_path = output_path / "01_basic_data_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Football Match Prediction - Basic Data Report\n")
        f.write("================================================\n\n")
        f.write(f"Number of rows: {len(raw_df):,}\n")
        f.write(f"Number of columns: {raw_df.shape[1]:,}\n\n")
        f.write("Columns:\n")
        for col in raw_df.columns:
            f.write(f"- {col}\n")
        f.write("\nMissing values per column:\n")
        f.write(str(raw_df.isna().sum()))
        f.write("\n\n")
        if "match_id" in raw_df.columns:
            f.write(f"Duplicate match_id count: {raw_df.duplicated(subset=['match_id']).sum():,}\n\n")
        if "match_result" in raw_df.columns:
            f.write("Target distribution - match_result:\n")
            f.write(str(raw_df["match_result"].value_counts(dropna=False)))
            f.write("\n\n")
        f.write("Data types:\n")
        f.write(str(raw_df.dtypes))
        f.write("\n")


def save_cleaning_summary(summary: dict, output_path: Path) -> None:
    path = output_path / "02_cleaning_summary.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("Football Match Prediction - Cleaning and Feature Engineering Summary\n")
        f.write("==================================================================\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")


def make_one_hot_encoder() -> OneHotEncoder:
    """Create OneHotEncoder while supporting older and newer scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=True)
    except TypeError:
        try:
            return OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse=True)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_one_hot_encoder()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return preprocessor, numerical_cols, categorical_cols


def get_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "Baseline most frequent": Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]),
        "Logistic Regression balanced": Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                solver="saga",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
        "Decision Tree balanced": Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", DecisionTreeClassifier(
                max_depth=14,
                min_samples_leaf=30,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "Random Forest balanced": Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=160,
                max_depth=16,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }


def evaluate_model(name: str, pipeline: Pipeline, X, y_true) -> dict:
    y_pred = pipeline.predict(X)
    result = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if hasattr(pipeline, "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(X)
            labels = list(pipeline.named_steps["model"].classes_)
            result["log_loss"] = log_loss(y_true, y_proba, labels=labels)
        except Exception:
            result["log_loss"] = np.nan
    else:
        result["log_loss"] = np.nan
    return result


def plot_target_distribution(y: pd.Series, plots_path: Path) -> None:
    counts = y.value_counts().reindex(CLASS_ORDER).fillna(0)
    plt.figure(figsize=(7, 5))
    plt.bar(counts.index, counts.values)
    plt.title("Distribution of Match Results")
    plt.xlabel("Match result")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(plots_path / "match_result_distribution.png", dpi=300)
    plt.close()


def plot_model_comparison(results_df: pd.DataFrame, plots_path: Path) -> None:
    ordered = results_df.sort_values("validation_macro_f1", ascending=True)
    plt.figure(figsize=(9, 5))
    plt.barh(ordered["model"], ordered["validation_macro_f1"])
    plt.title("Model Comparison by Validation Macro F1")
    plt.xlabel("Validation macro F1")
    plt.tight_layout()
    plt.savefig(plots_path / "model_comparison_macro_f1.png", dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, plots_path: Path) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_df = pd.DataFrame(cm, index=[f"Actual {c}" for c in class_names], columns=[f"Predicted {c}" for c in class_names])

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix - Best Model")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(plots_path / "confusion_matrix.png", dpi=300)
    plt.close()
    return cm_df


def plot_predicted_distribution(y_pred, plots_path: Path) -> None:
    counts = pd.Series(y_pred).value_counts().reindex(CLASS_ORDER).fillna(0)
    plt.figure(figsize=(7, 5))
    plt.bar(counts.index, counts.values)
    plt.title("Predicted Match Result Distribution - Best Model")
    plt.xlabel("Predicted match result")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(plots_path / "predicted_result_distribution.png", dpi=300)
    plt.close()


def clean_feature_name(name: str) -> str:
    name = name.replace("num__", "").replace("cat__", "")
    # Shorten one-hot names if needed.
    if len(name) > 90:
        name = name[:87] + "..."
    return name


def save_feature_importance(best_pipeline: Pipeline, output_path: Path, plots_path: Path) -> None:
    model = best_pipeline.named_steps["model"]
    preprocessor = best_pipeline.named_steps["preprocess"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"feature_{i}" for i in range(len(getattr(model, "feature_importances_", [])))])

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.mean(np.abs(model.coef_), axis=0)
    else:
        return

    n = min(len(feature_names), len(importances))
    fi = pd.DataFrame({
        "feature": [clean_feature_name(x) for x in feature_names[:n]],
        "importance": importances[:n],
    }).sort_values("importance", ascending=False)

    fi.to_csv(output_path / "06_feature_importance.csv", index=False)

    top = fi.head(15).sort_values("importance", ascending=True)
    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top["importance"])
    plt.title("Top Feature Importances - Best Model")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(plots_path / "feature_importance.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Multiclass football match outcome prediction pipeline.")
    parser.add_argument("--csv", required=True, help="Path to the input CSV dataset")
    parser.add_argument("--output", default="outputs", help="Output directory")
    args = parser.parse_args()

    output_path, plots_path = make_output_dirs(args.output)

    print(f"Loading data from: {args.csv}")
    raw_df = pd.read_csv(args.csv, low_memory=False)
    print(f"Loaded rows: {len(raw_df):,}")

    save_basic_report(raw_df, output_path)

    X, y, cleaning_summary = prepare_data(raw_df, output_path)
    save_cleaning_summary(cleaning_summary, output_path)
    plot_target_distribution(y, plots_path)

    # Train / validation / test split: 70% / 10% / 20%.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.125, random_state=RANDOM_STATE, stratify=y_train_val
    )

    split_df = pd.DataFrame({
        "split": ["train", "validation", "test"],
        "rows": [len(X_train), len(X_val), len(X_test)],
    })
    split_df.to_csv(output_path / "07_split_info.csv", index=False)

    preprocessor, numerical_cols, categorical_cols = build_preprocessor(X_train)
    with open(output_path / "03_feature_set_summary.txt", "w", encoding="utf-8") as f:
        f.write("Feature Set Summary\n")
        f.write("===================\n\n")
        f.write(f"Number of numerical columns before encoding: {len(numerical_cols)}\n")
        f.write(f"Number of categorical columns before encoding: {len(categorical_cols)}\n\n")
        f.write("Numerical columns:\n")
        for col in numerical_cols:
            f.write(f"- {col}\n")
        f.write("\nCategorical columns:\n")
        for col in categorical_cols:
            f.write(f"- {col}\n")

    models = get_models(preprocessor)
    rows = []
    fitted_models = {}

    for name, pipeline in models.items():
        print(f"Training model: {name}")
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline

        val_result = evaluate_model(name, pipeline, X_val, y_val)
        test_result = evaluate_model(name, pipeline, X_test, y_test)

        row = {"model": name}
        for key, value in val_result.items():
            if key != "model":
                row[f"validation_{key}"] = value
        for key, value in test_result.items():
            if key != "model":
                row[f"test_{key}"] = value
        rows.append(row)

    results_df = pd.DataFrame(rows).sort_values("validation_macro_f1", ascending=False)
    results_df.to_csv(output_path / "04_model_results.csv", index=False)
    plot_model_comparison(results_df, plots_path)

    print("\nModel results:")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    best_pipeline = fitted_models[best_name]
    y_pred = best_pipeline.predict(X_test)

    class_names = [c for c in CLASS_ORDER if c in sorted(y.unique())]
    report = classification_report(y_test, y_pred, labels=class_names, zero_division=0)
    with open(output_path / "05_classification_report_best_model.txt", "w", encoding="utf-8") as f:
        f.write(f"Best model selected by validation macro F1: {best_name}\n")
        f.write("\nClassification report on test set:\n")
        f.write(report)

    cm_df = plot_confusion_matrix(y_test, y_pred, class_names, plots_path)
    cm_df.to_csv(output_path / "08_confusion_matrix_best_model.csv")
    plot_predicted_distribution(y_pred, plots_path)
    save_feature_importance(best_pipeline, output_path, plots_path)

    print(f"\nBest model based on validation macro F1: {best_name}")
    print(f"All outputs are saved in: {output_path.resolve()}")


if __name__ == "__main__":
    main()
