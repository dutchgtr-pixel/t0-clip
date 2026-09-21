# Reproducing the cohort reconciliation

The [aggregate receipt](cohort_reconciliation.json) separates three results:

- The earlier 941-row AFT evaluation, covering 6-13 January, reports F1 0.8462.
- The later AFT run, covering 13-20 January, reports F1 0.8571 on 971 rows.
- Restricting its saved predictions to the 964 exact neural evaluation keys gives
  AFT F1 0.855967. All common duration labels match.

The final neural meta-ensemble reports F1 0.9209 in the associated holdout stream.
That result is retained execution output; its complete final decision vector has
not been recovered for a newly computed paired uncertainty analysis.

## Private prediction-file audit

Keep the input files and column mapping outside the public release. Configure
`left` and `right` objects with `id`, `origin`, `duration`, `prediction` and
optionally `event`, each naming the corresponding input column. Durations and
predictions are hours; origins require explicit timezones. See the script's help
for a neutral mapping example.

```bash
python scripts/audit_shared_cohort.py \
  --left-csv inputs/tree.csv --right-csv inputs/neural.csv \
  --columns-json inputs/columns.json --horizon 504 \
  --output results/shared-cohort.json
```

The audit rejects duplicate keys, mismatched common durations, invalid numbers
and horizon labels unresolved by censoring. It reports only hashes, counts,
time ranges and aggregate confusion matrices. It never chooses a threshold.
Its fixed rule is predicted duration greater than 504 hours: this is not an
automatic substitute for a dedicated survival-probability or meta-ensemble
decision. The saved earlier neural duration forecast and later tail classifier
are explicitly different artifacts in the receipt.

## Aggregate sensitivity calculation

```bash
python scripts/analyze_cohort_sensitivity.py \
  --delete-count 23 --output results/sensitivity.json
```

This calculation uses public aggregate inputs. It infers the unique integer
neural matrix compatible with the rounded metrics and surrounding cohort counts,
then enumerates every feasible deletion allocation. Its 0.899408 lower bound is
a conditional counterfactual, not an observed paired score or confidence limit.
The earlier 941-row report is a different dated window, so the script also flags
the failed positive-count condition for treating that report as a subset.

Run the synthetic regression checks with
`python -m pytest -q tests/test_cohort_reconciliation.py`. No private input is
needed for those tests. Researchers seeking the retained observation data should
follow the [data-access policy](../../DATA_ACCESS.md).
