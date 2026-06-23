import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from data.data_loader import RAW_FEATURES, get_official_test_last_rows
from monitoring.drift_simulation import (
    MAX_SIMULATION_INTENSITY,
    apply_simulation_scenario,
    focus_result_on_scenario,
)

CMAPSS_COLUMNS = ["unit", "cycle", "setting1", "setting2", "setting3"] + [
    f"sensor{i}" for i in range(1, 22)
]
FEATURE_COLUMNS = RAW_FEATURES
MAX_DISPLAY_DRIFT_SCORE = 10.0
FLOAT_TOLERANCE = 1e-6
DATA_DRIFT_REFERENCE_DATASET = "FD001"
DATA_DRIFT_CURRENT_DATASET = "FD002"
TARGET_DRIFT_REFERENCE_DATASET = "FD001"
TARGET_DRIFT_CURRENT_DATASET = "FD004"
CONCEPT_DRIFT_REFERENCE_DATASET = "FD001"
CONCEPT_DRIFT_CURRENT_DATASET = "FD003"
WINDOW_REFERENCE_DATASET = "FD001"
DEFAULT_MIN_WINDOW_SIZE = 5
DEFAULT_DATA_THRESHOLD_QUANTILE = 0.95
DEFAULT_TARGET_THRESHOLD_QUANTILE = 0.85
DEFAULT_CONCEPT_THRESHOLD_QUANTILE = 0.95
DEFAULT_THRESHOLD_WINDOW_SIZE = 10
COLLEAGUE_DATA_DRIFT_THRESHOLD = 0.3
COLLEAGUE_TARGET_DRIFT_THRESHOLD = 0.3
COLLEAGUE_CONCEPT_DRIFT_THRESHOLD = 0.3
_THRESHOLD_CACHE = {}


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
        "status": "calculated",
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
        "status": "calculated",
    }


def calculate_concept_drift(
    reference_predictions,
    reference_actual,
    current_predictions,
    current_actual,
    threshold=0.3,
):
    reference_mae = _mean_absolute_error(reference_predictions, reference_actual)
    current_mae = _mean_absolute_error(current_predictions, current_actual)

    if reference_mae == 0:
        score = 0.0 if current_mae == 0 else MAX_DISPLAY_DRIFT_SCORE
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
    concept_drift_threshold=0.3,
):
    # The monitoring datasets are intentionally split by drift type:
    # FD002 has different operating conditions, making it a clearer feature/data
    # drift scenario than comparing FD001 train with truncated FD001 test rows.
    data_reference_df, data_current_df = load_train_pair_for_data_drift(data_path)
    # FD004 is the most heterogeneous subset, so its train RUL distribution is a
    # useful target-drift stress case while keeping the existing target formula.
    target_reference_df, target_current_df = load_train_pair_for_target_drift(data_path)
    # Concept drift compares the FD001-trained model on official FD001 test
    # engines versus official FD003 test engines using the same MAE-ratio score.
    concept_reference_df, concept_current_df = (
        load_official_test_pair_for_concept_drift(data_path)
    )
    feature_names = feature_names or FEATURE_COLUMNS
    data_drift = calculate_data_drift(
        data_reference_df,
        data_current_df,
        columns=feature_names,
        threshold=data_drift_threshold,
    )
    target_drift = calculate_target_drift(
        target_reference_df,
        target_current_df,
        threshold=target_drift_threshold,
    )
    concept_drift = _calculate_model_concept_drift(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        reference_df=concept_reference_df,
        current_df=concept_current_df,
        threshold=concept_drift_threshold,
    )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_dataset": (
            f"data=train_{DATA_DRIFT_REFERENCE_DATASET};"
            f"target=train_{TARGET_DRIFT_REFERENCE_DATASET};"
            f"concept=test_{CONCEPT_DRIFT_REFERENCE_DATASET}"
        ),
        "current_dataset": (
            f"data=train_{DATA_DRIFT_CURRENT_DATASET};"
            f"target=train_{TARGET_DRIFT_CURRENT_DATASET};"
            f"concept=test_{CONCEPT_DRIFT_CURRENT_DATASET}"
        ),
        "data_path": str(data_path),
        "data_drift": data_drift,
        "target_drift": target_drift,
        "concept_drift": concept_drift,
    }
    return save_drift_report(report, reports_dir)


def run_simulated_test_drift_report(
    data_path="data/raw",
    reports_dir="reports/drift",
    dataset_id="FD001",
    model=None,
    feature_names=None,
    scenario="all",
    intensity=1.0,
):
    if model is None:
        raise ValueError("Model is required for simulated drift report")

    feature_names = feature_names or FEATURE_COLUMNS
    clean_current_df, _ = load_cmapss_dataset(data_path, dataset_id)
    reference_df = clean_current_df.copy()
    current_df = apply_simulation_scenario(
        clean_current_df,
        scenario=scenario,
        intensity=intensity,
        random_state=_simulation_random_state(scenario, intensity),
    )
    thresholds = get_colleague_thresholds()

    reference_predictions = model.predict(reference_df[feature_names])
    current_predictions = model.predict(current_df[feature_names])
    data_drift = calculate_data_drift(
        reference_df,
        current_df,
        columns=feature_names,
        threshold=thresholds["data_drift"],
    )
    target_drift = calculate_target_drift(
        reference_df,
        current_df,
        threshold=thresholds["target_drift"],
    )
    concept_drift = calculate_concept_drift(
        reference_predictions=reference_predictions,
        reference_actual=reference_df["RUL"],
        current_predictions=current_predictions,
        current_actual=current_df["RUL"],
        threshold=thresholds["concept_drift"],
    )
    concept_drift["status"] = "calculated"

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario,
        "intensity": float(max(0.0, min(float(intensity), MAX_SIMULATION_INTENSITY))),
        "reference_dataset": f"train_{dataset_id}_baseline",
        "current_dataset": f"simulated_{dataset_id}",
        "data_path": str(data_path),
        "thresholds": {
            "data_drift": thresholds["data_drift"],
            "target_drift": thresholds["target_drift"],
            "concept_drift": thresholds["concept_drift"],
        },
        "threshold_source": thresholds["threshold_source"],
        "data_drift": data_drift,
        "target_drift": target_drift,
        "concept_drift": concept_drift,
        "prediction_summary": build_dataframe_prediction_summary(
            current_df=current_df,
            current_predictions=current_predictions,
        ),
    }
    report = focus_result_on_scenario(report, scenario, intensity)
    return save_drift_report(report, reports_dir)


def calculate_calibrated_thresholds(
    reference_df,
    baseline_current_df,
    model,
    feature_names,
    data_quantile=DEFAULT_DATA_THRESHOLD_QUANTILE,
    target_quantile=DEFAULT_TARGET_THRESHOLD_QUANTILE,
    concept_quantile=DEFAULT_CONCEPT_THRESHOLD_QUANTILE,
    window_size=DEFAULT_THRESHOLD_WINDOW_SIZE,
):
    feature_names = feature_names or FEATURE_COLUMNS
    windows = list(_iter_unit_windows(reference_df, window_size=window_size))
    if not windows:
        windows = [baseline_current_df]

    reference_feature_stats = _reference_stats(reference_df, feature_names)
    reference_target_stats = _reference_stats(reference_df, ["RUL"])["RUL"]
    data_scores = []
    target_scores = []
    for window in windows:
        feature_scores = [
            _score_from_stats(
                current_mean=float(window[feature].mean()),
                reference_mean=stats["mean"],
                reference_std=stats["std"],
            )
            for feature, stats in reference_feature_stats.items()
        ]
        data_scores.append(max(feature_scores) if feature_scores else 0.0)
        target_scores.append(
            _score_from_stats(
                current_mean=float(window["RUL"].mean()),
                reference_mean=reference_target_stats["mean"],
                reference_std=reference_target_stats["std"],
            )
        )

    reference_predictions = model.predict(reference_df[feature_names])
    reference_mae = _mean_absolute_error(reference_predictions, reference_df["RUL"])
    window_predictions = _predict_windows(model, windows, feature_names)
    concept_scores = []
    for window, predictions in zip(windows, window_predictions):
        window_mae = _mean_absolute_error(predictions, window["RUL"])
        if reference_mae == 0:
            score = 0.0 if window_mae == 0 else MAX_DISPLAY_DRIFT_SCORE
        else:
            score = max(0.0, (window_mae - reference_mae) / reference_mae)
        concept_scores.append(score)

    return {
        "data_drift": _quantile_threshold(data_scores, data_quantile),
        "target_drift": _quantile_threshold(target_scores, target_quantile),
        "concept_drift": _quantile_threshold(concept_scores, concept_quantile),
        "threshold_source": (
            f"calibrated_data_p{int(data_quantile * 100)}_"
            f"target_p{int(target_quantile * 100)}_"
            f"concept_p{int(concept_quantile * 100)}_from_reference_"
            f"unit_windows_size_{window_size}"
        ),
    }


def get_colleague_thresholds():
    return {
        "data_drift": COLLEAGUE_DATA_DRIFT_THRESHOLD,
        "target_drift": COLLEAGUE_TARGET_DRIFT_THRESHOLD,
        "concept_drift": COLLEAGUE_CONCEPT_DRIFT_THRESHOLD,
        "threshold_source": "colleague_default_thresholds",
    }


def _predict_windows(model, windows, feature_names):
    if not windows:
        return []

    lengths = [len(window) for window in windows]
    combined_features = pd.concat(
        [window[feature_names] for window in windows],
        ignore_index=True,
    )
    combined_predictions = pd.Series(model.predict(combined_features))
    predictions = []
    start = 0
    for length in lengths:
        end = start + length
        predictions.append(combined_predictions.iloc[start:end].reset_index(drop=True))
        start = end
    return predictions


def _reference_stats(df, columns):
    stats = {}
    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        stats[column] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
        }
    return stats


def _score_from_stats(current_mean, reference_mean, reference_std):
    mean_shift = abs(current_mean - reference_mean)
    if abs(reference_std) <= FLOAT_TOLERANCE:
        return 0.0 if mean_shift <= FLOAT_TOLERANCE else MAX_DISPLAY_DRIFT_SCORE
    return mean_shift / reference_std


def get_calibrated_thresholds(
    data_path,
    dataset_id,
    reference_df,
    baseline_current_df,
    model,
    feature_names,
    data_quantile=DEFAULT_DATA_THRESHOLD_QUANTILE,
    target_quantile=DEFAULT_TARGET_THRESHOLD_QUANTILE,
    concept_quantile=DEFAULT_CONCEPT_THRESHOLD_QUANTILE,
    window_size=DEFAULT_THRESHOLD_WINDOW_SIZE,
):
    cache_key = (
        str(Path(data_path)),
        dataset_id,
        tuple(feature_names or FEATURE_COLUMNS),
        float(data_quantile),
        float(target_quantile),
        float(concept_quantile),
        int(window_size),
        id(model),
    )
    if cache_key not in _THRESHOLD_CACHE:
        _THRESHOLD_CACHE[cache_key] = calculate_calibrated_thresholds(
            reference_df=reference_df,
            baseline_current_df=baseline_current_df,
            model=model,
            feature_names=feature_names,
            data_quantile=data_quantile,
            target_quantile=target_quantile,
            concept_quantile=concept_quantile,
            window_size=window_size,
        )
    return dict(_THRESHOLD_CACHE[cache_key])


def build_dataframe_prediction_summary(current_df, current_predictions):
    predicted = pd.Series(current_predictions).reset_index(drop=True)
    actual = current_df["RUL"].reset_index(drop=True)
    errors = (predicted - actual).abs()
    return {
        "window_rows": int(len(current_df)),
        "labeled_rows": int(len(current_df)),
        "predicted_rul_mean": float(predicted.mean()),
        "actual_rul_mean": float(actual.mean()),
        "absolute_error_mae": float(errors.mean()),
        "absolute_error_p95": float(errors.quantile(0.95)),
    }


def _iter_unit_windows(reference_df, window_size):
    for _, unit_df in reference_df.groupby("unit"):
        unit_df = unit_df.sort_values("cycle")
        if len(unit_df) < window_size:
            yield unit_df
            continue
        for start in range(0, len(unit_df) - window_size + 1, window_size):
            yield unit_df.iloc[start : start + window_size]


def _quantile_threshold(scores, quantile):
    if not scores:
        return 0.0
    return float(pd.Series(scores).quantile(float(quantile)))


def _simulation_random_state(scenario, intensity):
    return 42 + sum(ord(char) for char in scenario) + int(float(intensity) * 1000)


def run_prediction_window_drift_report(
    predictions,
    data_path="data/raw",
    reports_dir="reports/drift",
    dataset_id="FD001",
    feature_names=None,
    reference_mae=None,
    model=None,
    window_size=100,
    min_window_size=DEFAULT_MIN_WINDOW_SIZE,
    data_drift_threshold=2.0,
    target_drift_threshold=1.5,
    concept_drift_threshold=2.0,
    threshold_source="manual_prediction_window_thresholds",
):
    feature_names = feature_names or FEATURE_COLUMNS
    train_reference_df = load_train_dataset(data_path, dataset_id)
    window_df = build_prediction_window_dataframe(predictions, feature_names)
    reference_df = build_clean_reference_window(
        train_reference_df,
        window_df,
        feature_names,
    )
    reference_source = reference_df.attrs.get("source", f"train_{dataset_id}")
    labeled_window_df = window_df.dropna(subset=["RUL"])
    if len(reference_df) == len(window_df):
        labeled_positions = window_df.index.get_indexer(labeled_window_df.index)
        labeled_reference_df = reference_df.iloc[labeled_positions].reset_index(
            drop=True
        )
        labeled_window_df = labeled_window_df.reset_index(drop=True)
    else:
        labeled_reference_df = reference_df.dropna(subset=["RUL"])
    prediction_summary = build_prediction_window_summary(
        window_df=window_df,
        labeled_window_df=labeled_window_df,
    )

    if len(window_df) < min_window_size:
        data_drift = _skipped_data_drift(
            threshold=data_drift_threshold,
            reason="not enough prediction rows",
            current_rows=len(window_df),
            min_window_size=min_window_size,
        )
    else:
        data_drift = calculate_data_drift(
            reference_df,
            window_df,
            columns=feature_names,
            threshold=data_drift_threshold,
        )

    if (
        len(labeled_window_df) < min_window_size
        or len(labeled_reference_df) < min_window_size
    ):
        target_drift = _skipped_target_drift(
            threshold=target_drift_threshold,
            reason="not enough labeled prediction rows",
            current_rows=len(labeled_window_df),
            min_window_size=min_window_size,
        )
        concept_drift = _skipped_concept_drift(
            threshold=concept_drift_threshold,
            reason="not enough labeled prediction rows",
            current_rows=len(labeled_window_df),
            min_window_size=min_window_size,
        )
    else:
        target_drift = calculate_target_drift(
            labeled_reference_df,
            labeled_window_df,
            threshold=target_drift_threshold,
        )
        if model is not None:
            reference_predictions = model.predict(labeled_reference_df[feature_names])
            concept_drift = calculate_concept_drift(
                reference_predictions=reference_predictions,
                reference_actual=labeled_reference_df["RUL"],
                current_predictions=labeled_window_df["predicted_rul"],
                current_actual=labeled_window_df["RUL"],
                threshold=concept_drift_threshold,
            )
            concept_drift["status"] = "calculated"
        else:
            concept_drift = calculate_concept_drift_from_mae(
                reference_mae=reference_mae,
                current_predictions=labeled_window_df["predicted_rul"],
                current_actual=labeled_window_df["RUL"],
                threshold=concept_drift_threshold,
            )

    actual_window_size = min(int(window_size), len(predictions))
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_dataset": reference_source,
        "current_dataset": f"prediction_window_last_{actual_window_size}",
        "data_path": str(data_path),
        "thresholds": {
            "data_drift": data_drift_threshold,
            "target_drift": target_drift_threshold,
            "concept_drift": concept_drift_threshold,
        },
        "threshold_source": threshold_source,
        "data_drift": data_drift,
        "target_drift": target_drift,
        "concept_drift": concept_drift,
        "prediction_summary": prediction_summary,
    }
    return save_drift_report(report, reports_dir)


def build_prediction_window_dataframe(predictions, feature_names):
    rows = []
    for prediction in reversed(predictions):
        input_payload = prediction.get("input", {})
        if not all(feature in input_payload for feature in feature_names):
            continue
        row = {feature: float(input_payload[feature]) for feature in feature_names}
        row["RUL"] = prediction.get("actual_rul")
        if "unit" in input_payload and input_payload["unit"] is not None:
            row["unit"] = float(input_payload["unit"])
        if row["RUL"] is not None:
            row["RUL"] = float(row["RUL"])
        row["predicted_rul"] = float(prediction["predicted_rul"])
        rows.append(row)
    return pd.DataFrame(rows, columns=["unit", *feature_names, "RUL", "predicted_rul"])


def build_clean_reference_window(train_reference_df, window_df, feature_names):
    if (
        window_df.empty
        or "unit" not in window_df.columns
        or window_df["unit"].isna().any()
    ):
        reference_df = train_reference_df.reset_index(drop=True)
        reference_df.attrs["source"] = "train_reference_fallback"
        return reference_df

    key_columns = ["unit", "cycle"]
    reference_columns = [
        *key_columns,
        *[feature for feature in feature_names if feature not in key_columns],
        "RUL",
    ]
    lookup = train_reference_df[reference_columns].copy()
    keys = window_df[key_columns].copy()
    keys["_window_index"] = window_df.index
    reference_window = keys.merge(
        lookup,
        on=key_columns,
        how="left",
        validate="many_to_one",
    ).sort_values("_window_index")

    if reference_window["RUL"].isna().any():
        reference_df = train_reference_df.reset_index(drop=True)
        reference_df.attrs["source"] = "train_reference_fallback"
        return reference_df

    reference_window = reference_window.drop(columns=["_window_index"]).reset_index(
        drop=True
    )
    reference_window.attrs["source"] = "clean_train_matching_prediction_window"
    return reference_window


def build_prediction_window_summary(window_df, labeled_window_df):
    summary = {
        "window_rows": int(len(window_df)),
        "labeled_rows": int(len(labeled_window_df)),
        "predicted_rul_mean": None,
        "actual_rul_mean": None,
        "absolute_error_mae": None,
        "absolute_error_p95": None,
    }
    if not window_df.empty:
        summary["predicted_rul_mean"] = float(window_df["predicted_rul"].mean())
    if not labeled_window_df.empty:
        errors = (
            labeled_window_df["predicted_rul"].reset_index(drop=True)
            - labeled_window_df["RUL"].reset_index(drop=True)
        ).abs()
        summary["actual_rul_mean"] = float(labeled_window_df["RUL"].mean())
        summary["absolute_error_mae"] = float(errors.mean())
        summary["absolute_error_p95"] = float(errors.quantile(0.95))
    return summary


def calculate_concept_drift_from_mae(
    reference_mae,
    current_predictions,
    current_actual,
    threshold=0.3,
):
    current_mae = _mean_absolute_error(current_predictions, current_actual)
    if reference_mae is None:
        return {
            "drift_detected": False,
            "score": 0.0,
            "threshold": threshold,
            "status": "skipped",
            "reason": "reference MAE is not available",
            "reference_mae": None,
            "current_mae": current_mae,
        }

    if reference_mae == 0:
        score = 0.0 if current_mae == 0 else MAX_DISPLAY_DRIFT_SCORE
    else:
        score = max(0.0, (current_mae - reference_mae) / reference_mae)

    return {
        "drift_detected": bool(score > threshold),
        "score": score,
        "threshold": threshold,
        "reference_mae": float(reference_mae),
        "current_mae": current_mae,
        "status": "calculated",
    }


def _skipped_data_drift(threshold, reason, current_rows, min_window_size):
    return {
        "drift_detected": False,
        "score": 0.0,
        "threshold": threshold,
        "drifted_features_count": 0,
        "drifted_features": [],
        "features": {},
        "status": "skipped",
        "reason": reason,
        "current_rows": current_rows,
        "min_window_size": min_window_size,
    }


def _skipped_target_drift(threshold, reason, current_rows, min_window_size):
    return {
        "drift_detected": False,
        "score": 0.0,
        "threshold": threshold,
        "details": {
            "status": "skipped",
            "score": 0.0,
            "threshold": threshold,
            "drifted": False,
            "reference_mean": None,
            "current_mean": None,
            "reference_std": None,
        },
        "status": "skipped",
        "reason": reason,
        "current_rows": current_rows,
        "min_window_size": min_window_size,
    }


def _skipped_concept_drift(threshold, reason, current_rows, min_window_size):
    return {
        "drift_detected": False,
        "score": 0.0,
        "threshold": threshold,
        "reference_mae": None,
        "current_mae": None,
        "status": "skipped",
        "reason": reason,
        "current_rows": current_rows,
        "min_window_size": min_window_size,
    }


def load_train_pair_for_data_drift(data_path):
    return (
        load_train_dataset(data_path, DATA_DRIFT_REFERENCE_DATASET),
        load_train_dataset(data_path, DATA_DRIFT_CURRENT_DATASET),
    )


def load_train_pair_for_target_drift(data_path):
    return (
        load_train_dataset(data_path, TARGET_DRIFT_REFERENCE_DATASET),
        load_train_dataset(data_path, TARGET_DRIFT_CURRENT_DATASET),
    )


def load_official_test_pair_for_concept_drift(data_path):
    return (
        load_official_test_last_rows(data_path, CONCEPT_DRIFT_REFERENCE_DATASET),
        load_official_test_last_rows(data_path, CONCEPT_DRIFT_CURRENT_DATASET),
    )


def load_train_dataset(data_path, dataset_id):
    data_dir = Path(data_path)
    return add_train_rul(_read_cmapss_file(data_dir / f"train_{dataset_id}.txt"))


def load_official_test_last_rows(data_path, dataset_id):
    data_dir = Path(data_path)
    test = _read_cmapss_file(data_dir / f"test_{dataset_id}.txt")
    rul = pd.read_csv(
        data_dir / f"RUL_{dataset_id}.txt",
        sep=r"\s+",
        header=None,
        names=["RUL"],
    )
    last_rows, y_true = get_official_test_last_rows(test, rul)
    official_rows = last_rows.copy()
    official_rows["RUL"] = y_true.astype(float)
    return official_rows


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
    mean_shift = abs(current_mean - reference_mean)

    if abs(reference_std) <= FLOAT_TOLERANCE:
        if mean_shift <= FLOAT_TOLERANCE:
            score = 0.0
            status = "skipped"
        else:
            score = MAX_DISPLAY_DRIFT_SCORE
            status = "calculated"
    else:
        score = mean_shift / reference_std
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
    if model is None:
        return {
            "drift_detected": False,
            "score": 0.0,
            "threshold": threshold,
            "status": "skipped",
            "reason": "model is not available",
        }

    reference_features = reference_df[feature_names]
    current_features = current_df[feature_names]
    if scaler is None:
        reference_predictions = model.predict(reference_features)
        current_predictions = model.predict(current_features)
    else:
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
