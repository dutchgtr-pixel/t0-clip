"""Command-line entry point; paths in configs are relative to the config file."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="marketneural")
    sub = parser.add_subparsers(dest="command", required=True)
    synthetic = sub.add_parser("synthetic", help="Generate a clearly labelled synthetic fixture")
    synthetic.add_argument("--output", type=Path, required=True)
    synthetic.add_argument("--n", type=int, default=1200)
    synthetic.add_argument("--seed", type=int, default=2026)
    synthetic.add_argument("--regime", choices=["ph", "nonlinear"], default="nonlinear")
    benchmark = sub.add_parser("benchmark", help="Fit/select on TRAIN/SVAL, then score frozen models on TEST")
    benchmark.add_argument("--config", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "synthetic":
        from .synthetic import generate
        print(json.dumps(generate(args.output, args.n, args.seed, args.regime), indent=2))
    else:
        from .benchmark import run
        summary = run(args.config, args.output)
        print(json.dumps({"study_kind": summary["study_kind"],
                          "test_ibs": {k: v["mean_test_ibs"] for k, v in summary["results"].items()}}, indent=2))


if __name__ == "__main__":
    main()
