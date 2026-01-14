# Time-on-the-Market (TOM) / Survival Feature Store — T0 & Leakage-Safe Certification Package

**Package purpose**: Update the existing end-to-end TOM feature-store documentation so it accurately reflects the **T0 / leakage-safe certification system** now implemented in the a Postgres database, and provide a replicable operating model (refresh, certify, guard training).

**Scope of certification (this package)**:
- **Certified entrypoint (feature store)**: `ml.survival_feature_store_t0_v1_v`
- **Training alias (guarded)**: `ml.survival_feature_store_train_v` (guarded view; fails hard if certification is stale or code drift is detected)

Everything else in the database is **either upstream raw fact**, **supporting dimension**, or **legacy/uncertified** unless explicitly called out as certified.

---

## 1. Deep analysis of the existing merged document (what must change)

Your `Full Documentation_ merged.txt` is a useful history, but it does **not** reflect the current certified architecture. The key gaps:

### 1.1 Time-of-query leakage and “current” selectors
The legacy TOM pipeline relies on objects that are **as-of-now** rather than **as-of-T0**, including:
- `ml.tom_speed_anchor_v1_mv` (contained `CURRENT_DATE` logic; explicit time-of-query leak)
- `ref.geo_mapping_current` and `ml.geo_dim_super_metro_v4_current_v` (select “current release”; stable only until release changes)

**Fix implemented**: replace these dependencies with T0-safe versions:
- `ml.tom_speed_anchor_asof_v1_mv` (anchored by `anchor_day`; embargo proof: no sales on/after anchor)
- `ref.geo_mapping_pinned_super_metro_v4_v1` + `ml.geo_dim_super_metro_v4_pinned_v1` + `ml.geo_dim_super_metro_v4_t0_v1` (release pinned; drift is detectable and requires recertification)

### 1.2 Missing SLA gating for image/damage features
The merged document describes feature joins without enforcing **SLA-based gating**. If image pipelines finish late, features can “appear later” for past rows unless gated.

**Fix implemented**:
- `ml.iphone_image_features_unified_t0_v1_mv` adds `img_within_sla` and nulls out image feature columns when not within SLA.
- `ml.v_damage_fusion_features_v2_scored_t0_v1_mv` gates damage features on the same SLA condition.

### 1.3 Labels leaking into feature closure
Gate B / firewall checks showed labels (`ml.tom_labels_v1_mv`) in the dependency closure of at least one training-style object (e.g., `ml.v_anchor_speed_train_base_rich_v1` in your dependency output). Labels must not be embedded in feature-store entrypoints if you want credible certification.

**Fix implemented**:
- Certification scope is enforced on the **feature-store entrypoint** (`ml.survival_feature_store_t0_v1_v`) and training reads go through the guarded `ml.survival_feature_store_train_v`.
- Training can still join labels downstream in training code, but **the certified feature store does not depend on label MVs**.

### 1.4 Dynamic falsification proof and guardrails absent
The merged document lacks:
- A repeatable **dataset hash** proof across time.
- A **viewdef baseline** (definition hash) closure to catch silent DDL changes.
- A **training guard** that fails hard if certification is stale or viewdef drift is detected.

**Fix implemented**:
- `audit.t0_dataset_hash_baseline` + `audit.dataset_sha256(...)` + multi-day baselines
- `audit.t0_viewdef_baseline` pinned closure viewdefs
- `audit.t0_cert_registry` + `audit.require_certified(_strict)` + `audit.cert_guard(...)`
- Guarded training view `ml.survival_feature_store_train_v` (cert guard embedded)

### 1.5 Operationalization missing
The merged doc includes refresh logic, but it refreshes legacy objects and does not incorporate “refresh → recertify → enforce guard” as an atomic operation.

**Fix implemented**:
- `audit.refresh_and_certify_survival_v1(p_raise boolean)` refreshes MVs in dependency order, runs certification, and (optionally) fails if not certified.

---

## 2. What is now true (and what is not)

### 2.1 “Are we leak-safe?”
For the **certified entrypoints**:
- Yes, **under the assumptions encoded in the assertions**:
  - No forbidden objects in closure (labels/current/live/legacy)
  - No time-of-query tokens in closure
  - Speed anchor embargo proof holds
  - AI enrichment bounded (as-of window)
  - Image/damage features are SLA gated and **do not populate when `img_within_sla=false`**
  - Dynamic hash baselines match for multiple historical days
  - Viewdef baselines match (no drift)

### 2.2 “Is the entire DB leak-safe?”
No. Only the **certified entrypoints** are claimable as leak-safe. Legacy objects remain in the DB for backward compatibility, but are commented as **NOT T0 SAFE** and should be access-controlled away from training roles.

---

## 3. What you should do next (operationally)
1. Treat `audit.refresh_and_certify_survival_v1(true)` as the **only supported refresh path** for training.
2. Ensure training reads **only** `ml.survival_feature_store_train_v` (guarded view), not the raw T0 view and not any legacy objects.
3. Version-control this package + the extracted SQL viewdefs; recertify whenever upstream logic changes.

---

## 4. Files in this package (8-part markdown set)

1. `01_Executive_Summary.md` (this file)
2. `02_Architecture_and_Data_Contracts.md`
3. `03_Phases_and_Proofs.md`
4. `04_Remediation_and_T0_DDL.md`
5. `05_Certification_Guardrails_and_Access_Control.md`
6. `06_Refresh_Runbook_Concurrent_Refresh.md`
7. `07_Appendix_Original_Documentation_Part1.md` (verbatim, updated only where strictly necessary)
8. `08_Appendix_Original_Documentation_Part2.md` (verbatim continuation)

