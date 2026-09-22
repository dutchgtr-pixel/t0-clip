"""Audit paired, retained binary decisions without exporting private rows.

This is a retrospective analysis of fixed predictions. Bootstrap intervals do
not include model fitting, hyperparameter search or experiment selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read_decisions(path, keys, label, prediction, invariants=()):
    path = Path(path)
    data = path.read_bytes()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = set(keys) | {label, prediction} | set(invariants)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing required columns")
        rows = {}
        for row in reader:
            key = tuple(row[c] for c in keys)
            if any(not x for x in key) or key in rows:
                raise ValueError("Keys must be complete and unique")
            try:
                y, p = float(row[label]), float(row[prediction])
            except ValueError as exc:
                raise ValueError("Labels and predictions must be binary") from exc
            if y not in (0.0, 1.0) or p not in (0.0, 1.0):
                raise ValueError("Labels and predictions must be binary")
            rows[key] = (int(y), int(p), tuple(row[c] for c in invariants))
    if not rows:
        raise ValueError("Empty cohort")
    return rows, {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def metrics(y, p):
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tn = int(np.sum((y == 0) & (p == 0)))
    return {"n": len(y), "positives": tp + fn, "tp": tp, "fp": fp,
            "fn": fn, "tn": tn, "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def compare(models, reference, *, keys, label, prediction, invariants=(),
            bootstrap=5000, seed=20260922):
    if len(models) < 2 or reference not in models:
        raise ValueError("Provide at least two models and a valid reference")
    if bootstrap < 1:
        raise ValueError("Bootstrap count must be positive")
    loaded = {name: read_decisions(path, keys, label, prediction, invariants)
              for name, path in models.items()}
    baseline = loaded[reference][0]
    order = sorted(baseline)
    y = np.array([baseline[k][0] for k in order])
    preds = {}
    for name, (rows, _) in loaded.items():
        if set(rows) != set(baseline):
            raise ValueError("Cohorts do not contain exactly the same keys")
        if any(rows[k][0] != baseline[k][0] or rows[k][2] != baseline[k][2]
               for k in order):
            raise ValueError("Labels or invariant fields differ on matched keys")
        preds[name] = np.array([rows[k][1] for k in order])
    result = {
        "schema_version": 1,
        "scope": "Retrospective paired analysis of fixed retained binary predictions",
        "matching": {"n": len(y), "unique_complete_keys": True,
                     "identical_key_sets": True, "identical_labels": True,
                     "identical_invariants": True, "invariant_count": len(invariants)},
        "privacy": "Only aggregate counts, metrics and source hashes; no row data or source paths",
        "reference": reference,
        "models": {name: dict(metrics(y, p), source=loaded[name][1])
                   for name, p in preds.items()},
        "all_positive_reference": metrics(y, np.ones(len(y), dtype=int)),
        "bootstrap": {"resamples": bootstrap, "seed": seed,
                      "method": "Percentile interval from IID paired row resampling",
                      "interval_level": 0.95,
                      "scope": "Conditional on fitted models, selected thresholds and retained cohort; excludes fitting, search, selection and temporal dependence uncertainty; intervals are not multiplicity-adjusted"},
        "contrasts": {}, "all_positive_contrasts": {}}
    rng = np.random.default_rng(seed)
    diffs = {name: [] for name in preds if name != reference}
    constant_diffs = {name: [] for name in preds}
    for _ in range(bootstrap):
        idx = rng.integers(0, len(y), size=len(y))
        sampled_f1 = {name: metrics(y[idx], p[idx])["f1"] for name, p in preds.items()}
        ref_f1 = sampled_f1[reference]
        constant_f1 = metrics(y[idx], np.ones(len(idx), dtype=int))["f1"]
        for name in diffs:
            diffs[name].append(sampled_f1[name] - ref_f1)
        for name in constant_diffs:
            constant_diffs[name].append(sampled_f1[name] - constant_f1)
    ref_correct = preds[reference] == y
    for name, distribution in diffs.items():
        correct = preds[name] == y
        lo, hi = np.quantile(distribution, [0.025, 0.975])
        result["contrasts"][name] = {
            "f1_difference": result["models"][name]["f1"] - result["models"][reference]["f1"],
            "f1_difference_interval": [float(lo), float(hi)],
            "candidate_only_correct": int(np.sum(correct & ~ref_correct)),
            "reference_only_correct": int(np.sum(~correct & ref_correct)),
            "both_correct": int(np.sum(correct & ref_correct)),
            "both_wrong": int(np.sum(~correct & ~ref_correct))}
    for name, distribution in constant_diffs.items():
        lo, hi = np.quantile(distribution, [0.025, 0.975])
        result["all_positive_contrasts"][name] = {
            "f1_difference": result["models"][name]["f1"] - result["all_positive_reference"]["f1"],
            "f1_difference_interval": [float(lo), float(hi)]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="NAME=CSV")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--keys", nargs="+", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--invariants", nargs="*", default=[])
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pairs = [item.split("=", 1) for item in args.model]
    if any(len(item) != 2 or not item[0] or not item[1] for item in pairs):
        parser.error("Each --model must be NAME=CSV")
    if len({item[0] for item in pairs}) != len(pairs):
        parser.error("Model names must be unique")
    result = compare(dict(pairs), args.reference, keys=args.keys, label=args.label,
                     prediction=args.prediction, invariants=args.invariants,
                     bootstrap=args.bootstrap, seed=args.seed)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
