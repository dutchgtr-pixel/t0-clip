# Python Notes

This package includes the exact `train_slow21_gate_classifier_pg.py` present in this workspace at packaging time.

Key sections to review:
- WOE banding / WOE fit: `compute_woe_anchor_p_slow21(...)`
- OOF driver: `compute_woe_anchor_oof(...)`
- Artifact persistence: `persist_woe_anchor_artifacts_v1(...)`
- Main training flow: search for `[woe]` log lines and the call to `persist_woe_anchor_artifacts_v1`.

CLI knobs:
- `--woe_folds` (default 5)
- `--woe_eps` (default 0.5)
- `--woe_band_schema_version` (default 1)
- `--disable_woe_oof`
- `--disable_woe_persist`
