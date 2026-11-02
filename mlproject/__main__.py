import argparse
import json
from pathlib import Path
from .core import predict, read_data, train


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a leakage-safe ML pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--data", required=True)
    fit.add_argument("--config", default="project.json")
    fit.add_argument("--output", default="artifacts")
    fit.add_argument("--task", choices=["classification", "regression"])
    fit.add_argument("--target", help="Override the target for a single-output task")
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--folds", type=int, default=5)
    fit.add_argument("--test-size", type=float, default=0.2)
    fit.add_argument("--jobs", type=int, default=1)
    fit.add_argument("--tune", action="store_true")
    fit.add_argument("--importance", action="store_true")
    fit.add_argument("--group-column", help="Keep all rows for an entity in the same split and CV fold")
    infer = sub.add_parser("predict")
    infer.add_argument("--model", required=True)
    infer.add_argument("--data", required=True)
    infer.add_argument("--output", default="predictions.csv")
    args = vars(parser.parse_args())
    command = args.pop("command")
    try:
        df = read_data(args.pop("data"))
        if command == "train":
            config = json.loads(Path(args.pop("config")).read_text(encoding="utf-8"))
            report = train(df, config, **args)
            print(json.dumps({"selected_model": report["selected_model"], "holdout": report["holdout"]}, indent=2))
        else:
            result = predict(args["model"], df)
            result.to_csv(args["output"], index=False)
            print(f"Saved {len(result)} predictions to {args['output']}")
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
