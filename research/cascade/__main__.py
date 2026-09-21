"""Run a small, entirely artificial end-to-end three-stage training example."""
import argparse
import json

import numpy as np
import torch

from . import CascadeEstimator, FeatureBatch, RoutingPolicy, StageEstimator
from .legacy_core import TrainConfig


def smoke(seed=42):
    torch.set_num_threads(2)
    rng = np.random.default_rng(seed)
    n = 48
    features = FeatureBatch(rng.normal(size=(n, 4)).astype(np.float32),
                            rng.integers(0, 4, (n, 2)),
                            rng.normal(size=(n, 768)).astype(np.float32),
                            rng.normal(size=(n, 512)).astype(np.float32))
    duration = np.tile([24., 60., 100., 200., 480., 650., 80., 300.], 6)
    event = np.tile([1, 1, 1, 1, 1, 0, 0, 0], 6)
    config = TrainConfig(d_model=16, n_latents=4, fusion_layers=1, n_heads=2,
                         n_experts=2, n_bins=16, text_tokens=2, img_tokens=2,
                         tab_num_tokens=2, tab_cat_tokens=2, dropout=0., attn_dropout=0.)
    stages = {stage: [StageEstimator(stage, config, seed=seed+stage, epochs=2, batch_size=16)]
              for stage in (0, 1, 2)}
    cascade = CascadeEstimator(stages).fit(features.take(slice(0, 32)), duration[:32], event[:32],
                                           validation=(features.take(slice(32, 40)), duration[32:40], event[32:40]),
                                           policy=RoutingPolicy(.999, 0., .5))
    result = cascade.predict(features.take(slice(40, None)))
    labels, counts = np.unique(result["bucket"], return_counts=True)
    return {"purpose": "synthetic software execution; not historical model accuracy",
            "seed": seed, "training_rows": 32, "development_rows": 8, "test_rows": 8,
            "thresholds": "prespecified routing exercise; not selected for accuracy",
            "stage_audit": cascade.fit_audit_, "bucket_counts": dict(zip(labels.tolist(), counts.tolist())),
            "stage_parameters": {stage: sum(p.numel() for p in members[0].model_.parameters())
                                  for stage, members in stages.items()},
            "finite_required_scores": bool(np.isfinite(result["p_fast72"]).all())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.smoke:
        parser.error("Choose --smoke; real cohorts require an explicit caller-side temporal protocol")
    print(json.dumps(smoke(args.seed), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
