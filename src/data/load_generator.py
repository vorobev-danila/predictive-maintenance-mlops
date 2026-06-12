import argparse
import json
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from data.data_loader import iter_official_test_rows, iter_train_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Send real CMAPSS official test rows to the prediction API."
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8080")
    parser.add_argument("--data-path", default="data/raw")
    parser.add_argument("--dataset-id", default="FD001")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--unit-id", type=int, default=1)
    parser.add_argument("--delay", type=float, default=7.0)
    parser.add_argument("--loop", action="store_true")
    return parser.parse_args()


def post_prediction(api_url, payload):
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{api_url.rstrip('/')}/predict",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def stream_predictions(args):
    while True:
        rows_sent = 0
        row_iterator = (
            iter_train_rows if args.split == "train" else iter_official_test_rows
        )
        rows = row_iterator(
            data_path=args.data_path, dataset_id=args.dataset_id, unit_id=args.unit_id
        )
        for payload in rows:
            result = post_prediction(args.api_url, payload)
            predicted_rul = float(result["rul"])
            actual_rul = float(payload["actual_rul"])
            error = abs(predicted_rul - actual_rul)
            rows_sent += 1
            print(
                f"{args.dataset_id} split={args.split} unit={args.unit_id} "
                f"cycle={payload['cycle']:.0f} "
                f"predicted={predicted_rul:.2f} "
                f"actual={actual_rul:.2f} "
                f"absolute_error={error:.2f}"
            )
            time.sleep(args.delay)

        if rows_sent == 0:
            raise ValueError(
                f"No rows found for dataset_id={args.dataset_id}, unit_id={args.unit_id}"
            )
        if not args.loop:
            break


def main():
    args = parse_args()
    try:
        stream_predictions(args)
    except (URLError, TimeoutError) as error:
        raise SystemExit(f"Prediction API is unavailable: {error}") from error


if __name__ == "__main__":
    main()
