"""Offline SQL provenance, syntax, dependency and evidence-integrity checks."""
import json
from pathlib import Path
import tempfile

from research.feature_store_ddl.inventory import ROOT, build_inventory, canonical_hash, inspect_sql


def test_all_public_sql_and_explicit_query_fragments_parse():
    report = build_inventory()
    assert report['counts']['parser_rejected'] == 0
    assert report['counts']['fragment_parsed'] == 2
    assert report['counts']['sql_files'] >= 79


def test_archived_export_hashes_match_and_no_seed_rows_are_exported():
    folder = ROOT / 'research/feature_store_ddl'
    provenance = json.loads((folder / 'provenance.json').read_text())
    for entry in provenance['sources']:
        if entry['export_path']:
            assert canonical_hash(folder / entry['export_path']) == entry['export_sha256_lf']
    channel = (folder / 'archived/40_channel_base.sql').read_text()
    assert 'INSERT INTO ml.power_seller_fingerprint_v1' not in channel


def test_dependency_inspection_reports_actual_qualified_relations():
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        path = root / 'example.sql'
        path.write_text('CREATE VIEW f.features AS SELECT x.id FROM source_input.records x;')
        entry = inspect_sql(path, root)
        assert entry['defines'] == ['f.features']
        assert entry['directly_references'] == ['source_input.records']
        path.write_text('CREATE VIEW f.features AS SELECT FROM ;')
        assert inspect_sql(path, root)['status'] == 'parser_rejected'


def test_aggregate_evidence_denominators_and_policy_counts_are_consistent():
    evidence = json.loads((ROOT / 'research/leakage/aggregate_evidence.json').read_text())
    for cohort in evidence['lifecycle_availability']:
        assert sum(row['total'] for row in cohort['duration_buckets']) == cohort['rows']
        assert sum(row['missing'] for row in cohort['duration_buckets']) == cohort['missing_pattern_rows']
        assert all(0 <= row['missing'] <= row['total'] for row in cohort['duration_buckets'])
        diagnostic = cohort['missingness_only_rule']
        assert sum(diagnostic[k] for k in ('tp', 'fp', 'fn', 'tn')) == cohort['rows']
        assert diagnostic['fn'] == 0
        assert abs(diagnostic['f1'] - 2 * diagnostic['tp'] / (2 * diagnostic['tp'] + diagnostic['fp'])) < 1e-12
    for name in ('sval', 'holdout', 'forward'):
        rows = evidence['locked_policy'][name]
        assert rows['zero_duration']['n'] + rows['positive_duration']['n'] == rows['overall']['n']
        for split in ('overall', 'zero_duration', 'positive_duration'):
            r = rows[split]
            assert r['tp'] + r['fp'] + r['fn'] + r['tn'] == r['n']
            assert abs(r['precision'] - r['tp'] / (r['tp'] + r['fp'])) < 1e-12
