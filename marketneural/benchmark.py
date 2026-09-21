"""One shared chronological protocol for classical and neural survival models."""
from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import platform
import time

import numpy as np

from .data import attach_features, read_table, sha256_file, split_temporally
from .metrics import SurvivalMetrics, operating_metrics, paired_ibs_intervals, select_threshold
from .models import create_model


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def source_digest():
    root = Path(__file__).parent
    h = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def run(config_path: Path, output: Path):
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Results already exist; choose a new output directory. Do not overwrite a scored test run.")
    output.mkdir(parents=True, exist_ok=True)
    numeric = config["data"]["numeric_columns"]
    categorical = config["data"].get("categorical_columns", [])
    table_path = (config_path.parent/config["data"]["table"]).resolve()
    vector_path = config["data"].get("vectors")
    vector_path = (config_path.parent/vector_path).resolve() if vector_path else None
    df = read_table(table_path, numeric, categorical)
    cohorts, cohort_audit = split_temporally(df, config["protocol"])
    preprocessing = attach_features(cohorts, numeric, categorical, vector_path)
    train, val, test = (cohorts[name] for name in ("train", "validation", "test"))
    times = np.asarray(config["evaluation"]["times_hours"], float)
    horizon = float(config["evaluation"]["decision_horizon_hours"])
    if not np.isfinite(horizon) or horizon <= 0 or horizon > times[-1]:
        raise ValueError("Decision horizon must be positive and within the evaluation grid")
    # The decision horizon is evaluated directly, never interpolated through a
    # discrete hazard grid or substituted with a different requested horizon.
    seeds = config.get("seeds", [42])
    if not seeds or len(set(seeds)) != len(seeds) or not all(isinstance(s, int) for s in seeds):
        raise ValueError("Declare a nonempty unique integer seed list")
    models = config["models"]
    if not models or "coxph" not in models:
        raise ValueError("A benchmark must include Cox PH as a prespecified reference")
    metric = SurvivalMetrics(train.duration, train.event, times,
                             config["evaluation"].get("min_censor_survival", .05))
    provenance = {
        "package_source_sha256": source_digest(), "config_sha256": sha256_file(config_path),
        "table_sha256": sha256_file(table_path),
        "vectors_sha256": sha256_file(vector_path) if vector_path else None,
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "scikit-survival", "torch")},
    }
    # Record the intent before any model fitting or validation selection.
    public_config = json.loads(json.dumps(config))
    public_config["data"]["table"] = "<input-table; see content hash>"
    if vector_path:
        public_config["data"]["vectors"] = "<input-vectors; see content hash>"
    write_json(output/"protocol.json", {"configuration": public_config, "provenance": provenance,
                                      "cohort_audit": cohort_audit, "preprocessing": preprocessing})
    selected = {}
    fitted = {}
    threshold_cfg = config["evaluation"].get("operating_point", {})
    for name, candidates in models.items():
        if not candidates:
            raise ValueError(f"Empty candidate list for {name}")
        records, candidates_fitted = [], []
        for index, parameters in enumerate(candidates):
            fits, per_seed = [], []
            for seed in seeds:
                print(f"fit {name} candidate={index} seed={seed}", flush=True)
                start = time.perf_counter()
                model = create_model(name, random_state=seed, **parameters)
                model.fit(train.features, train.duration, train.event,
                          validation=(val.features, val.duration, val.event))
                seconds = time.perf_counter()-start
                prediction = model.predict_survival(val.features, times)
                scores = metric.evaluate(val.duration, val.event, prediction)
                probability = 1-model.predict_survival(val.features, [horizon])[:, 0]
                threshold = select_threshold(val.duration, val.event, probability, horizon, **threshold_cfg)
                per_seed.append({"seed": seed, "fit_seconds": seconds, "validation": scores,
                                 "operating_point": threshold,
                                 "best_epoch": getattr(model, "best_epoch_", None)})
                fits.append(model)
            record = {"candidate": index, "parameters": parameters, "seeds": per_seed,
                      "mean_validation_ibs": float(np.mean([s["validation"]["integrated_brier_score"] for s in per_seed]))}
            records.append(record)
            candidates_fitted.append(fits)
        best = min(range(len(records)), key=lambda i: (records[i]["mean_validation_ibs"], i))
        selected[name] = {"selected_candidate": best, "selected_parameters": candidates[best],
                          "candidate_results": records,
                          "selection_rule": "lowest mean SVAL IBS across prespecified seeds; candidate order breaks exact ties"}
        fitted[name] = candidates_fitted[best]
    # Freeze every family before any test prediction/metric. This local artifact
    # is auditable, not an access-control boundary against a user rerunning tests.
    write_json(output/"selection.json", selected)
    frozen_selection_hash = sha256_file(output/"selection.json")
    results, primary_row_losses = {}, {}
    for name in models:
        record = selected[name]
        per_seed = []
        chosen = record["candidate_results"][record["selected_candidate"]]
        for seed_index, (seed, model) in enumerate(zip(seeds, fitted[name])):
            prediction = model.predict_survival(test.features, times)
            score = metric.evaluate(test.duration, test.event, prediction)
            probability = 1-model.predict_survival(test.features, [horizon])[:, 0]
            threshold = chosen["seeds"][seed_index]["operating_point"]["threshold"]
            operating = operating_metrics(test.duration, test.event, probability, horizon, threshold)
            per_seed.append({"seed": seed, "test": score, "test_operating_point": operating})
            if seed_index == 0:
                primary_row_losses[name] = metric.per_row_ibs(test.duration, test.event, prediction)
        ibs = [s["test"]["integrated_brier_score"] for s in per_seed]
        results[name] = {"selected_candidate": record["selected_candidate"],
                         "selected_parameters": record["selected_parameters"],
                         "mean_test_ibs": float(np.mean(ibs)), "std_test_ibs_across_seeds": float(np.std(ibs)),
                         "seed_results": per_seed}
    bootstrap_cfg = config["evaluation"].get("bootstrap", {})
    bootstrap = paired_ibs_intervals(primary_row_losses, reference="coxph", **bootstrap_cfg)
    summary = {
        "schema_version": 1, "study_kind": config.get("study_kind", "unclassified"),
        "claim_scope": ("Synthetic pipeline demonstration; no real-world superiority inference"
                        if config.get("study_kind") == "synthetic_smoke" else "Controlled benchmark output; interpretation requires cohort and provenance review"),
        "provenance": provenance, "frozen_selection_sha256": frozen_selection_hash,
        "cohort_audit": cohort_audit, "results": results,
        "paired_bootstrap": {"reference": "coxph", "primary_seed": seeds[0],
                             "repetitions": bootstrap_cfg.get("repetitions", 1000),
                             "seed": bootstrap_cfg.get("seed", 2026),
                             "scope": "Paired entity resampling, conditional on fitted models and TRAIN censor weights; no training uncertainty or multiplicity correction",
                             "comparisons": bootstrap},
    }
    write_json(output/"summary.json", summary)
    return summary
