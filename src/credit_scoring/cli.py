"""Command-line entry point: ``credit-scoring {train,tune,serve}``."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence

from credit_scoring.config import get_settings
from credit_scoring.models import SUPPORTED_MODELS


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="credit-scoring", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train, evaluate and save model(s).")
    train.add_argument("--model", choices=[*SUPPORTED_MODELS, "all"], default=None)
    train.add_argument("--split", choices=["group", "row"], default=None)
    train.add_argument("--no-benchmark", action="store_true", help="Skip latency benchmark.")

    tune = sub.add_parser("tune", help="Grid-search hyperparameters with grouped CV.")
    tune.add_argument("--model", choices=SUPPORTED_MODELS, required=True)

    serve = sub.add_parser("serve", help="Run the FastAPI server.")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = _parse_args(argv)
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.command == "train":
        from credit_scoring.train import prepare_data, train_model

        if args.split:
            settings = settings.model_copy(update={"split_strategy": args.split})
        names = SUPPORTED_MODELS if args.model == "all" else [args.model or settings.model_name]
        data = prepare_data(settings)
        summary = {}
        for name in names:
            result = train_model(settings, name, data, benchmark=not args.no_benchmark)
            summary[name] = {
                k: round(result.metrics[k], 4) for k in ("accuracy", "f1_weighted", "cohen_kappa")
            }
        print(json.dumps(summary, indent=2))
    elif args.command == "tune":
        from credit_scoring.train import tune_model

        print(json.dumps(tune_model(settings, args.model), indent=2, default=str))
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("credit_scoring.api:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
