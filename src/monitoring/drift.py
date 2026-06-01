import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CMAPSS_COLUMNS = ["unit", "cycle", "setting1", "setting2", "setting3"] + [
    f"sensor{i}" for i in range(1, 22)
]
FEATURE_COLUMNS = ["setting1", "setting2", "setting3"] + [
    f"sensor{i}" for i in range(1, 22)
]


def load_cmapss_dataset(data_path="data/raw", dataset_id="FD001"):
    data_dir = Path(data_path)
    train = _read_cmapss_file(data_dir / f"train_{dataset_id}.txt")
    test = _read_cmapss_file(data_dir / f"test_{dataset_id}.txt")
    rul = pd.read_csv(
        data_dir / f"RUL_{dataset_id}.txt",
        sep=r"\s+",
        header=None,
        names=["final_rul"],
    )

    return add_train_rul(train), add_test_rul(test, rul)


def add_train_rul(df):
    df = df.copy()
    max_cycles = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = max_cycles - df["cycle"]
    return df


def add_test_rul(test_df, rul_df):
    df = test_df.copy()
    max_cycles = df.groupby("unit")["cycle"].transform("max")
    final_rul_by_unit = {
        unit: float(rul)
        for unit, rul in zip(sorted(df["unit"].unique()), rul_df["final_rul"])
    }
    df["RUL"] = (
        max_cycles - df["cycle"] + df["unit"].map(final_rul_by_unit).astype(float)
    )
    return df


def calculate_data_drift(reference_df, current_df, columns=None, threshold=0.3):
    columns = columns or FEATURE_COLUMNS
    features = {}
    drifted_features = []
    scores = []

    for column in columns:
        feature_result = _calculate_column_shift(
            reference_df[column],
            current_df[column],
            threshold=threshold,
        )
        features[column] = feature_result
        if feature_result["status"] == "calculated":
            scores.append(feature_result["score"])
            if feature_result["drifted"]:
                drifted_features.append(column)

    overall_score = max(scores) if scores else 0.0
    return {
        "drift_detected": bool(drifted_features),
        "score": overall_score,
        "threshold": threshold,
        "drifted_features_count": len(drifted_features),
        "drifted_features": drifted_features,
        "features": features,
    }


def calculate_target_drift(
    reference_df, current_df, target_column="RUL", threshold=0.3
):
    result = _calculate_column_shift(
        reference_df[target_column],
        current_df[target_column],
        threshold=threshold,
    )
    return {
        "drift_detected": result["drifted"],
        "score": result["score"],
        "threshold": threshold,
        "details": result,
    }


def calculate_concept_drift(
    reference_predictions,
    reference_actual,
    current_predictions,
    current_actual,
    threshold=0.25,
):
    reference_mae = _mean_absolute_error(reference_predictions, reference_actual)
    current_mae = _mean_absolute_error(current_predictions, current_actual)

    if reference_mae == 0:
        score = 0.0 if current_mae == 0 else 1_000_000.0
    else:
        score = max(0.0, (current_mae - reference_mae) / reference_mae)

    return {
        "drift_detected": bool(score > threshold),
        "score": score,
        "threshold": threshold,
        "reference_mae": reference_mae,
        "current_mae": current_mae,
    }


def run_cmapss_drift_report(
    data_path="data/raw",
    reports_dir="reports/drift",
    dataset_id="FD001",
    model=None,
    scaler=None,
    feature_names=None,
    data_drift_threshold=0.3,
    target_drift_threshold=0.3,
    concept_drift_threshold=0.25,
):
    reference_df, current_df = load_cmapss_dataset(data_path, dataset_id)
    feature_names = feature_names or FEATURE_COLUMNS
    data_drift = calculate_data_drift(
        reference_df,
        current_df,
        columns=feature_names,
        threshold=data_drift_threshold,
    )
    target_drift = calculate_target_drift(
        reference_df,
        current_df,
        threshold=target_drift_threshold,
    )
    concept_drift = _calculate_model_concept_drift(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        reference_df=reference_df,
        current_df=current_df,
        threshold=concept_drift_threshold,
    )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_dataset": f"train_{dataset_id}",
        "current_dataset": f"test_{dataset_id}",
        "data_path": str(data_path),
        "data_drift": data_drift,
        "target_drift": target_drift,
        "concept_drift": concept_drift,
    }
    return save_drift_report(report, reports_dir)


def save_drift_report(report, reports_dir="reports/drift"):
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    timestamp = report["created_at"].replace(":", "-").replace("+", "Z")
    report_path = reports_path / f"drift_report_{timestamp}.json"
    latest_path = reports_path / "latest.json"

    _write_json(report_path, report)
    _write_json(latest_path, report)
    return report


def load_latest_drift_report(reports_dir="reports/drift"):
    latest_path = Path(reports_dir) / "latest.json"
    if not latest_path.exists():
        return None
    with latest_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _read_cmapss_file(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)


def _calculate_column_shift(reference_series, current_series, threshold):
    reference = pd.to_numeric(reference_series, errors="coerce").dropna()
    current = pd.to_numeric(current_series, errors="coerce").dropna()
    reference_std = float(reference.std())
    reference_mean = float(reference.mean())
    current_mean = float(current.mean())

    if reference_std == 0:
        if reference_mean == current_mean:
            score = 0.0
            status = "skipped"
        else:
            score = 1_000_000.0
            status = "calculated"
    else:
        score = abs(current_mean - reference_mean) / reference_std
        status = "calculated"

    return {
        "status": status,
        "score": float(score),
        "threshold": threshold,
        "drifted": bool(status == "calculated" and score > threshold),
        "reference_mean": reference_mean,
        "current_mean": current_mean,
        "reference_std": reference_std,
    }


def _calculate_model_concept_drift(
    model,
    scaler,
    feature_names,
    reference_df,
    current_df,
    threshold,
):
    if model is None or scaler is None:
        return {
            "drift_detected": False,
            "score": 0.0,
            "threshold": threshold,
            "status": "skipped",
            "reason": "model or scaler is not available",
        }

    reference_features = reference_df[feature_names]
    current_features = current_df[feature_names]
    reference_predictions = model.predict(scaler.transform(reference_features))
    current_predictions = model.predict(scaler.transform(current_features))
    result = calculate_concept_drift(
        reference_predictions=reference_predictions,
        reference_actual=reference_df["RUL"],
        current_predictions=current_predictions,
        current_actual=current_df["RUL"],
        threshold=threshold,
    )
    result["status"] = "calculated"
    return result


def _mean_absolute_error(predictions, actual):
    errors = pd.Series(predictions).reset_index(drop=True) - pd.Series(
        actual
    ).reset_index(drop=True)
    return float(errors.abs().mean())


def _write_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
