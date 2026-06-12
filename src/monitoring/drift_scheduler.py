import argparse
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def parse_args():
    parser = argparse.ArgumentParser(
        description="Periodically trigger drift calculation through the public API."
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8080")
    parser.add_argument("--dataset-id", default="FD001")
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--retry-interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def run_drift(api_url, dataset_id):
    body = json.dumps({"dataset_id": dataset_id}).encode("utf-8")
    request = Request(
        f"{api_url.rstrip('/')}/drift/run",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    args = parse_args()
    while True:
        sleep_seconds = args.interval
        try:
            result = run_drift(args.api_url, args.dataset_id)
            report = result["report"]
            print(
                "drift_run "
                f"status={result['status']} "
                f"data_score={report['data_drift']['score']:.4f} "
                f"target_score={report['target_drift']['score']:.4f} "
                f"concept_score={report['concept_drift']['score']:.4f}",
                flush=True,
            )
        except (HTTPError, URLError, TimeoutError) as error:
            print(f"drift_run failed: {error}", flush=True)
            sleep_seconds = args.retry_interval

        if args.once:
            break
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
