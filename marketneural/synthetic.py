"""Synthetic reproducibility fixture, never evidence about a real marketplace."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def generate(output: Path, n=1200, seed=2026, regime="nonlinear"):
    if n < 300 or regime not in {"ph", "nonlinear"}:
        raise ValueError("Use at least 300 rows and regime ph or nonlinear")
    output.mkdir(parents=True, exist_ok=True)
    targets = [output/"cohort.csv", output/"vectors.npz", output/"manifest.json"]
    if any(path.exists() for path in targets):
        raise FileExistsError("Synthetic output already exists; choose a new directory")
    rng = np.random.default_rng(seed)
    tabular = rng.normal(size=(n, 4))
    text = rng.normal(size=(n, 4)).astype(np.float32)
    image = rng.normal(size=(n, 2, 3)).astype(np.float32)
    report = rng.normal(size=(n, 2, 3)).astype(np.float32)
    image_mask = rng.random((n, 2)) > .15
    report_mask = rng.random((n, 2)) > .2
    category = rng.integers(0, 3, size=n)
    signal = .65*tabular[:, 0] - .35*tabular[:, 1] + .3*text[:, 0] + .2*category
    signal += .25*image[:, 0, 0]*image_mask[:, 0] + .2*report[:, 0, 0]*report_mask[:, 0]
    shape = np.full(n, 1.25)
    if regime == "nonlinear":
        signal += .6*tabular[:, 2]*text[:, 1] + .4*np.sin(tabular[:, 3])
        shape = np.where(tabular[:, 3] > 0, .8, 1.7)  # deliberately non-PH
    event_time = 95 * (-np.log(rng.uniform(.00001, .99999, n)) / np.exp(signal))**(1/shape)
    # Independent observation censoring; split-specific administrative censoring
    # is added later by the common benchmark data contract.
    censor_time = rng.uniform(12, 420, size=n)
    observed = np.minimum(event_time, censor_time)
    start = pd.Timestamp("2025-01-01", tz="UTC")
    decision = start + pd.to_timedelta(rng.uniform(0, 110, n), unit="D")
    row_ids = np.array([f"synthetic-{i:06d}" for i in range(n)])
    frame = pd.DataFrame({
        "row_id": row_ids, "entity_id": row_ids, "decision_time": decision,
        "feature_observed_at": decision-pd.Timedelta(hours=1),
        "observed_until": decision+pd.to_timedelta(observed, unit="h"),
        "event": (event_time <= censor_time).astype(int),
        **{f"x{i}": tabular[:, i] for i in range(4)}, "category": [f"group-{i}" for i in category],
    })
    frame.to_csv(targets[0], index=False)
    np.savez_compressed(targets[1], row_ids=row_ids, text=text, image=image, report=report,
                        image_mask=image_mask, report_mask=report_mask)
    targets[2].write_text(json.dumps({"kind": "synthetic", "n": n, "seed": seed, "regime": regime,
                                   "purpose": "Pipeline smoke demonstration; not an empirical superiority experiment"}, indent=2)+"\n")
    return {"kind": "synthetic", "n": n, "seed": seed, "regime": regime}
