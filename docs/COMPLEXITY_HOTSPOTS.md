# Complexity Hotspots

This document lists the highest cyclomatic-complexity hotspots for the public release.
High-complexity functions in this repository are primarily *pipeline controllers* (multi-stage orchestrators)
for training, audits, and closed-loop self-healing jobs. Complexity is intentionally concentrated in these
controllers; the broader codebase remains lower-complexity on average.

## Python (Radon CC) — Top hotspots

| Score | Grade | Function | File |
|---:|:---:|---|---|
| 166 | F | main | modeling\slow21\train_slow21_gate_classifier.py |
| 160 | F | pipeline_run | modeling\meta\fast24_flagger3.py |
| 67 | F | run_mask23_tuning | modeling\meta\fast24_flagger3.py |
| 58 | F | main | infra\jobs\quality-control\db-healing\src\dedupe_ai_sql.py |
| 46 | F | main | infra\jobs\quality-control\db-healing\src\battery_refit_llm_v2.py |
| 42 | F | xgb_aft_train_predict | modeling\slow21\train_slow21_gate_classifier.py |
| 37 | E | _apply_cli_hf_overrides | modeling\meta\fast24_flagger3.py |
| 33 | E | harvest_features | modeling\meta\fast24_flagger3.py |
| 32 | E | run_stage1_tuning | modeling\meta\fast24_flagger3.py |
| 32 | E | apply_mask23_once | modeling\meta\fast24_flagger3.py |
| 31 | E | scrape_listing_images | infra\jobs\labeling\image-pipeline\src\scrape_images_playwright.py |
| 30 | D | load_damage_fusion_features | modeling\slow21\train_slow21_gate_classifier.py |
| 29 | D | run_loop | modeling\embeddings\sbert\sbert_vec_upsert_title_desc_caption.py |
| 29 | D | compute_woe_anchor_p_slow21 | modeling\slow21\train_slow21_gate_classifier.py |
| 26 | D | soft_gate_penalty | modeling\meta\fast24_flagger3.py |
| 25 | D | main | modeling\embeddings\image\build_img_vec512_openclip_pg.py |
| 24 | D | call_model_for_listing | infra\jobs\labeling\image-pipeline\src\label_damage.py |
| 23 | D | process_one | infra\jobs\quality-control\db-healing\src\battery_refit_llm_v2.py |
| 22 | D | _pick_title_desc_from_html_text | infra\jobs\quality-control\db-healing\src\dedupe_ai_sql.py |
| 21 | D | call_battery_eval_both | infra\jobs\quality-control\db-healing\src\battery_refit_llm_v2.py |

## Go (gocyclo) — Top hotspots

| Score | Function | File:Line |
|---:|---|---|
| 93 | reverseOnce | infra\jobs\scrapers\reverse-monitor\src\reverse_monitor.go:1436 |
| 83 | auditStaleOnce | infra\jobs\auditors\older21days-tail-audit\src\audit_older21days.go:727 |
| 49 | parseFlags | infra\jobs\auditors\status-sanity-audit\src\status_sanity_audit.go:314 |
| 45 | runAudit | infra\jobs\auditors\status-sanity-audit\src\status_sanity_audit.go:772 |
| 41 | parseGenericJSONPayload | infra\jobs\auditors\status-sanity-audit\src\status_sanity_audit.go:1415 |
| 41 | maybeApply | infra\jobs\auditors\status-sanity-audit\src\status_sanity_audit.go:1799 |
| 32 | runRepair | infra\jobs\scrapers\reverse-monitor\src\reverse_monitor.go:2147 |
| 27 | buildAuditRow | infra\jobs\auditors\status-sanity-audit\src\status_sanity_audit.go:1663 |
| 26 | auditOne | infra\jobs\auditors\status-sanity-audit\src\status_sanity_audit.go:1544 |
| 25 | runAuditStatus | infra\jobs\scrapers\reverse-monitor\src\reverse_monitor.go:1905 |
| 23 | scrapeOnce | infra\jobs\scrapers\discovery\src\fetchd.go:1752 |
| 23 | scrapeOnce | infra\jobs\scrapers\discovery\src\main.go:1393 |
| 23 | runTest | infra\jobs\scrapers\reverse-monitor\src\reverse_monitor.go:2064 |
| 20 | selectAndRun | infra\jobs\quality-control\db-healing\src\post_sold_audit.go:887 |
| 20 | consumeDetails | infra\jobs\scrapers\discovery\src\fetchd.go:1572 |
| 20 | consumeDetails | infra\jobs\scrapers\discovery\src\main.go:1213 |
| 19 | smartGET | infra\jobs\scrapers\reverse-monitor\src\reverse_monitor.go:458 |
| 17 | (*HTTPAdapter).smartGET | infra\jobs\auditors\older21days-tail-audit\src\marketplace_adapter.go:495 |
| 17 | (*AutoTuner).Recalc | infra\jobs\scrapers\discovery\src\fetchd.go:817 |
| 17 | (*AutoTuner).Recalc | infra\jobs\scrapers\discovery\src\main.go:458 |

