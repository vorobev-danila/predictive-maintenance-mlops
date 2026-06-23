import argparse
import json
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd
from monitoring.drift import load_cmapss_dataset
from monitoring.drift_simulation import (
    MAX_SIMULATION_INTENSITY,
    SUPPORTED_SCENARIOS,
    apply_simulation_scenario,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream simulated FD001 drift rows to the prediction API."
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8080")
    parser.add_argument("--data-path", default="data/raw")
    parser.add_argument("--dataset-id", default="FD001")
    parser.add_argument(
        "--scenario", choices=sorted(SUPPORTED_SCENARIOS), default="all"
    )
    parser.add_argument("--windows", type=int, default=7)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="Zero-based row offset for the first streamed window.",
    )
    parser.add_argument(
        "--intensity",
        type=float,
        default=None,
        help="Fixed simulation intensity. If omitted, intensity grows by window.",
    )
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    return parser.parse_args()


def post_json(api_url, path, payload):
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{api_url.rstrip('/')}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def stream_simulation(args):
    clean_train_df, _ = load_cmapss_dataset(args.data_path, args.dataset_id)

    while True:
        for window_index in range(1, args.windows + 1):
            intensity = (
                args.intensity
                if args.intensity is not None
                else calculate_window_intensity(window_index, args.windows)
            )
            window_df = build_simulated_window(
                clean_train_df,
                scenario=args.scenario,
                intensity=intensity,
                window_index=window_index,
                window_size=args.window_size,
                start_row=args.start_row,
                random_state=42 + window_index,
            )
            for _, row in window_df.iterrows():
                payload = _row_to_prediction_payload(row)
                result = post_json(args.api_url, "/predict", payload)
                predicted_rul = float(result["rul"])
                actual_rul = float(payload["actual_rul"])
                absolute_error = abs(predicted_rul - actual_rul)
                print(
                    f"{args.dataset_id} scenario={args.scenario} "
                    f"window={window_index}/{args.windows} "
                    f"intensity={intensity:.2f} "
                    f"unit={payload['unit']:.0f} "
                    f"cycle={payload['cycle']:.0f} "
                    f"predicted={predicted_rul:.2f} "
                    f"actual={actual_rul:.2f} "
                    f"absolute_error={absolute_error:.2f}"
                )
                time.sleep(args.delay)

            post_json(
                args.api_url,
                "/drift/run",
                {
                    "dataset_id": args.dataset_id,
                    "scenario": args.scenario,
                    "intensity": intensity,
                },
            )

        if not args.loop:
            break


def _select_window(df, window_index, window_size, start_row=0):
    start = (int(start_row) + (window_index - 1) * window_size) % len(df)
    end = start + window_size
    if end <= len(df):
        return df.iloc[start:end]
    return pd.concat([df.iloc[start:], df.iloc[: end - len(df)]])


def build_simulated_window(
    clean_train_df,
    scenario,
    intensity,
    window_index,
    window_size,
    random_state,
    start_row=0,
):
    clean_window = _select_window(
        clean_train_df,
        window_index=window_index,
        window_size=window_size,
        start_row=start_row,
    )
    # Apply simulation after slicing so concept drift shuffles RUL inside the
    # same window. That preserves target distribution while breaking X -> y.
    return apply_simulation_scenario(
        clean_window,
        scenario=scenario,
        intensity=intensity,
        random_state=random_state,
    )


def calculate_window_intensity(window_index, total_windows):
    if int(total_windows) <= 1:
        return 0.0
    progress = (int(window_index) - 1) / (int(total_windows) - 1)
    return MAX_SIMULATION_INTENSITY * progress


def _row_to_prediction_payload(row):
    payload = {
        "unit": float(row["unit"]),
        "cycle": float(row["cycle"]),
        "setting1": float(row["setting1"]),
        "setting2": float(row["setting2"]),
        "setting3": float(row["setting3"]),
        "actual_rul": float(row["RUL"]),
    }
    for sensor_index in range(1, 22):
        payload[f"sensor{sensor_index}"] = float(row[f"sensor{sensor_index}"])
    return payload


def main():
    args = parse_args()
    try:
        stream_simulation(args)
    except (URLError, TimeoutError) as error:
        raise SystemExit(f"Prediction API is unavailable: {error}") from error


if __name__ == "__main__":
    main()
