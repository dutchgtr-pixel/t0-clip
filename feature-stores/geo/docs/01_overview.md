# 1. Overview

## 1.1 Purpose

The geo store provides stable, low-dimensional geographic context for each listing to improve:
- liquidity priors,
- region/metro-specific sell-through behavior,
- market segmentation in downstream models.

This store is designed as a **release-versioned mapping system**:
- you load a mapping “release” into reference tables,
- you mark exactly one release as current,
- you then **pin** that release for training to make training deterministic and T0-safe.

## 1.2 Key idea for leak-proofing

A “current release” pointer is operationally useful, but it is not deterministic over time.
For certification, training must not depend on a mutable “current” pointer.

Therefore, we create:
- `ref.geo_mapping_pinned_super_metro_v4_v1` which returns exactly one constant `release_id`.

All certified training objects depend on the pinned release, not the moving pointer.

## 1.3 Scope of certification

We certify the DB geo dimension and its entrypoint:

- `ml.geo_feature_store_t0_v1_v` (entrypoint including `edited_date`)
- `ml.geo_dim_super_metro_v4_t0_train_v` (guarded consumer view)

Trainer-generated OHE and priors are derived from this certified store at runtime.
