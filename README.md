# t0-clip

t0-clip is a production-shaped, closed-loop data + database governance + machine learning system.
It is designed to turn continuously changing, real-world longitudinal data into leak-safe (T0-correct)
certified feature-store surfaces that power high-accuracy predictive models.

This repository is a public, platform-agnostic release. It demonstrates the system architecture, governance
primitives, and modeling stack. Users must adapt the acquisition/connectors to their own data sources.

## Key docs
- System Spine (closed-loop orchestration + what runs when): docs/SYSTEM_SPINE.md
- Complexity Hotspots (radon + gocyclo): docs/COMPLEXITY_HOTSPOTS.md

## What you are looking at (high level)
This is not "a model". This is a full closed-loop pipeline:

1) Acquisition writes raw and canonical operational records.
2) Auditors and validators detect inconsistency, drift, and state errors.
3) A self-healing DB layer reconciles and repairs state with deterministic governance.
4) Feature stores are built as modular surfaces and exposed through certified entrypoints only.
5) Models train/score from certified surfaces (survival + tail gates + meta learners).
6) Results and audits feed back into governance (baselines, certification registry, drift checks).

## Core ideas
### 1) T0 correctness (time-leakage prevention)
All feature computation and model consumption is anchored to a decision time T0. No future information is allowed.
The system treats leakage as a systems problem: contracts, certified entrypoints, and guardrails prevent it.

### 2) Certified feature stores (fail-closed governance)
Models do not query raw tables directly. They consume certified entrypoint views only.
Certification is enforced using audit tables, view-definition baselines, and dataset-hash baselines.
Training and scoring can be blocked if a store is stale, drifting, or non-compliant.

### 3) Closed-loop self-healing operational database
Operational data is continuously validated, repaired/reconciled, and re-certified.
The goal is to prevent silent degradation: if upstream data changes or becomes inconsistent, the system detects it,
repairs state (under deterministic rules), and re-establishes certified surfaces.

### 4) Tail-aware modeling (where real systems break)
The modeling stack is designed for long-tail outcomes and decision boundaries:
- censoring-aware survival modeling (time-to-event)
- tail gates / boundary classifiers
- anchors and priors (robust under sparsity)
- optional stacking/meta learners for higher performance

## What is included in the public release
### Orchestration (what runs)
Airflow DAGs and job containers that execute the closed-loop cycle:
- ingestion/discovery
- reverse monitoring / state updates
- DB-healing / quality-control
- multimodal labeling pipelines (image)
- daily/periodic auditors (sanity + tail/stale detection)

See: docs/SYSTEM_SPINE.md

### Feature stores (SQL + docs)
A modular set of feature-store blocks built in SQL, with certified entrypoints and guards.
The repo includes audit/certification primitives such as:
- audit.t0_cert_registry
- audit.t0_viewdef_baseline
- audit.t0_dataset_hash_baseline
- audit.require_certified_strict(...)
- audit.assert_t0_certified_*()

And multiple certified entrypoints such as:
- ml.survival_feature_store_t0_v1_v
- ml.geo_feature_store_t0_v1_v
- ml.fusion_feature_store_t0_v1_v
- ml.woe_anchor_feature_store_t0_v1_v

### Multimodal enrichment (embeddings + labeling)
- Text embeddings (SBERT-style) pipelines
- Image embeddings (OpenCLIP/ViT-style vectors) pipelines
- Image labeling pipelines for feature extraction and quality signals

### Modeling
- Survival/tail modeling code (trainer/controller scripts)
- Meta learner pipeline (stacking/blending style)
- Supporting utilities for evaluation, calibration, and artifacts

### Papers
A set of thesis-style documents explaining leakage control, governance, tail capture, and empirical results.

## What this is NOT
- Not a one-click installer.
- Not a "point at any website and run" tool.
- Platform-agnostic: you must implement/adapt acquisition and operational configs for your data sources.

## Repository layout (orientation)
- docker/                container/runtime scaffolding
- etl/dags/              orchestration DAGs
- infra/jobs/            acquisition, auditors, labeling, QC, repair loops
- feature-stores/        modular store definitions and certification logic
- modeling/              survival/tail models, meta learners, embedding pipelines
- governance/            certification and drift-tolerant audit logic
- papers/                thesis-style documentation

## How to evaluate "power" quickly
If you want an objective view of what this system is capable of, start here:
- docs/SYSTEM_SPINE.md (what runs, cadence, closed-loop behavior)
- feature-stores/ (certified entrypoints + guardrails + baselines)
- docs/COMPLEXITY_HOTSPOTS.md (where complexity concentrates: controllers, not random sprawl)

