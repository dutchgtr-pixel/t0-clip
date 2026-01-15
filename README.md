# t0-clip

## Key docs
- **System Spine (closed-loop orchestration):** docs/SYSTEM_SPINE.md
- **Complexity Hotspots (radon + gocyclo):** docs/COMPLEXITY_HOTSPOTS.md

	0-clip is a production-shaped system for converting continuously changing real-world data into leak-safe (T0-correct)
certified feature-store surfaces that power high-accuracy predictive models. It is designed as a closed loop: the
operational database is continuously validated, repaired/reconciled, and re-certified so training and inference cannot
silently drift.

## What this is
- A closed-loop "self-healing database" setup that feeds ML models through governed, auditable pipelines.
- A T0 certification and feature-store governance layer (fail-closed eligibility, contracts, baselines, drift checks).
- A multi-model, tail-aware modeling stack built for real decision boundaries on messy, longitudinal data.

## What this is NOT
- Not a one-click installer.
- Not a "point at any website and run" tool.
- Platform-agnostic: users must implement/adapt acquisition and operational configs for their own data sources.

## Architecture (closed loop)
See docs/SYSTEM_SPINE.md for the orchestration spine and schedules.

## Repository layout (orientation)
- docker/ - container/runtime scaffolding
- etl/dags/ - orchestration DAGs
- infra/jobs/ - acquisition, auditors, labeling, QC, repair loops
- eature-stores/ - modular store definitions and certification logic
- modeling/ - survival/tail models, meta-learners, embedding pipelines
- papers/ - thesis-style documentation
