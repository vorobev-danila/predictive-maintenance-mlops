SUPPORTED_SCENARIOS = {"data_drift", "target_drift", "concept_drift", "all"}
SCENARIO_DRIFT_TYPES = {
    "data_drift": {"data_drift"},
    "target_drift": {"target_drift"},
    "concept_drift": {"concept_drift"},
    "all": {"data_drift", "target_drift", "concept_drift"},
}
MAX_SIMULATION_INTENSITY = 1.0
DATA_SHIFT_MULTIPLIER = 1.0


def validate_scenario(scenario):
    if scenario not in SUPPORTED_SCENARIOS:
        supported = ", ".join(sorted(SUPPORTED_SCENARIOS))
        raise ValueError(
            f"Unsupported drift simulation scenario: {scenario}. Use: {supported}"
        )


def apply_simulation_scenario(
    current_df,
    scenario,
    intensity,
    random_state=42,
):
    validate_scenario(scenario)
    bounded_intensity = max(0.0, min(float(intensity), MAX_SIMULATION_INTENSITY))
    active_drift_types = get_active_drift_types(scenario, bounded_intensity)
    simulated = current_df.copy()

    if "data_drift" in active_drift_types:
        simulated["sensor2"] = (
            simulated["sensor2"] + 2.0 * DATA_SHIFT_MULTIPLIER * bounded_intensity
        )
        simulated["sensor4"] = (
            simulated["sensor4"] + 20.0 * DATA_SHIFT_MULTIPLIER * bounded_intensity
        )
        simulated["sensor11"] = (
            simulated["sensor11"] + 1.0 * DATA_SHIFT_MULTIPLIER * bounded_intensity
        )
        simulated["sensor15"] = (
            simulated["sensor15"] + 0.12 * DATA_SHIFT_MULTIPLIER * bounded_intensity
        )

    if "target_drift" in active_drift_types:
        multiplier = max(0.05, 1.0 - 0.9 * bounded_intensity)
        simulated["RUL"] = (simulated["RUL"] * multiplier).clip(lower=0)

    if "concept_drift" in active_drift_types:
        shuffled = (
            simulated["RUL"]
            .sample(
                frac=1.0,
                random_state=random_state,
            )
            .reset_index(drop=True)
        )
        simulated["RUL"] = (
            (1.0 - bounded_intensity) * simulated["RUL"].reset_index(drop=True)
            + bounded_intensity * shuffled
        ).to_numpy()

    return simulated


def get_active_drift_types(scenario, intensity):
    validate_scenario(scenario)
    bounded_intensity = max(0.0, min(float(intensity), MAX_SIMULATION_INTENSITY))
    if bounded_intensity <= 0.0:
        return set()
    return SCENARIO_DRIFT_TYPES[scenario]


def focus_result_on_scenario(report, scenario, intensity=None):
    if intensity is None:
        intensity = report.get("intensity", MAX_SIMULATION_INTENSITY)
    active_drift_types = get_active_drift_types(scenario, intensity)
    report["active_drift_types"] = sorted(active_drift_types)

    if "data_drift" not in active_drift_types:
        report["data_drift"] = {
            **report["data_drift"],
            "drift_detected": False,
            "score": 0.0,
            "drifted_features_count": 0,
            "drifted_features": [],
            "status": "not_applicable_for_scenario",
        }

    if "target_drift" not in active_drift_types:
        report["target_drift"] = {
            **report["target_drift"],
            "drift_detected": False,
            "score": 0.0,
            "status": "not_applicable_for_scenario",
        }

    if "concept_drift" not in active_drift_types:
        report["concept_drift"] = {
            **report["concept_drift"],
            "drift_detected": False,
            "score": 0.0,
            "status": "not_applicable_for_scenario",
        }

    return report
