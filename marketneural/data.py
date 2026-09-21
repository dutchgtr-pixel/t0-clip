"""Fail-closed temporal cohort construction and training-only preprocessing.

``feature_observed_at`` is an attestation about the latest underlying evidence,
not proof that the supplied vectors or aggregates were historically available.
An immutable upstream snapshot and transformation provenance are still needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REQUIRED = {"row_id", "entity_id", "decision_time", "feature_observed_at", "observed_until", "event"}
FORBIDDEN_FEATURE_TOKENS = {"duration", "event", "sold", "outcome", "label", "target", "gmc"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc(value):
    parsed = pd.to_datetime(value, utc=True, errors="raise")
    if np.asarray(pd.isna(parsed)).any():
        raise ValueError("Timestamp fields cannot contain NaT or missing instants")
    return parsed


@dataclass
class Cohort:
    frame: pd.DataFrame
    duration: np.ndarray
    event: np.ndarray
    features: dict[str, np.ndarray] | None = None


def read_table(path: Path, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    if path.suffix.lower() != ".csv":
        raise ValueError("The public contract accepts CSV tables; export other formats explicitly")
    df = pd.read_csv(path, dtype={"row_id": str, "entity_id": str})
    missing = (REQUIRED | set(numeric) | set(categorical)) - set(df)
    if missing:
        raise ValueError(f"Missing table columns: {sorted(missing)}")
    selected = numeric + categorical
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Declare a nonempty, unique numeric/categorical feature allowlist")
    for name in selected:
        tokens = set(name.lower().replace("-", "_").split("_"))
        if name in REQUIRED or tokens & FORBIDDEN_FEATURE_TOKENS:
            raise ValueError(f"Forbidden potential outcome feature: {name}")
    if df[list(REQUIRED)].isna().any().any():
        raise ValueError("Required contract fields cannot be null")
    if df.row_id.duplicated().any() or df.row_id.str.strip().eq("").any() or df.entity_id.str.strip().eq("").any():
        raise ValueError("row_id must be unique and row/entity identifiers nonempty")
    if (df.row_id != df.row_id.str.strip()).any() or (df.entity_id != df.entity_id.str.strip()).any():
        raise ValueError("Identifiers cannot contain surrounding whitespace")
    for col in ("decision_time", "feature_observed_at", "observed_until"):
        df[col] = utc(df[col])
    if not df.event.isin([0, 1]).all():
        raise ValueError("event must contain only 0 or 1")
    if (df.feature_observed_at > df.decision_time).any():
        raise ValueError("Future feature evidence: feature_observed_at exceeds decision_time")
    if (df.observed_until <= df.decision_time).any():
        raise ValueError("Zero/negative follow-up requires an explicit upstream endpoint audit")
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if np.isinf(df[col]).any():
            raise ValueError(f"Infinite numeric feature: {col}")
    return df


def split_temporally(df: pd.DataFrame, protocol: dict) -> tuple[dict[str, Cohort], dict]:
    boundaries = {key: utc(protocol[key]) for key in ("train_start", "validation_start", "test_start", "test_end", "test_as_of")}
    if not all(boundaries[a] < boundaries[b] for a, b in zip(boundaries, list(boundaries)[1:])):
        raise ValueError("Require train_start < validation_start < test_start < test_end < test_as_of")
    specs = {
        "train": (boundaries["train_start"], boundaries["validation_start"], boundaries["validation_start"]),
        "validation": (boundaries["validation_start"], boundaries["test_start"], boundaries["test_start"]),
        "test": (boundaries["test_start"], boundaries["test_end"], boundaries["test_as_of"]),
    }
    cohorts, audit = {}, {"boundaries": {k: v.isoformat() for k, v in boundaries.items()}}
    identities = {}
    for name, (start, end, cutoff) in specs.items():
        frame = df.loc[(df.decision_time >= start) & (df.decision_time < end)].copy()
        if frame.empty:
            raise ValueError(f"Empty temporal {name} cohort")
        identities[name] = set(frame.entity_id)
        # Strict cutoffs: an event at the fit/selection instant is not yet known.
        effective_end = frame.observed_until.clip(upper=cutoff)
        event = frame.event.astype(bool) & (frame.observed_until < cutoff)
        duration = (effective_end - frame.decision_time).dt.total_seconds().to_numpy() / 3600
        cohorts[name] = Cohort(frame, duration, event.to_numpy())
        audit[name] = {
            "rows": len(frame), "events": int(event.sum()),
            "administratively_censored": int((frame.observed_until >= cutoff).sum()),
            "label_cutoff": cutoff.isoformat(), "unique_entities": len(identities[name]),
            "repeated_entity_rows": int(frame.entity_id.duplicated().sum()),
        }
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if identities[a] & identities[b]:
            raise ValueError(f"Entity leakage between {a} and {b}; construct disjoint entities upstream")
    if any(cohorts[k].frame.entity_id.duplicated().any() for k in cohorts):
        raise ValueError("This benchmark requires one landmark per entity; cluster/repeated-landmark protocols need a separate design")
    audit["excluded_outside_windows"] = len(df) - sum(len(c.frame) for c in cohorts.values())
    return cohorts, audit


class Preprocessor:
    """Median imputation, scaling and categorical vocabulary fitted on TRAIN only."""

    def __init__(self, numeric, categorical):
        self.numeric, self.categorical = list(numeric), list(categorical)
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=True)
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)

    def _cats(self, frame):
        return frame[self.categorical].fillna("__MISSING__").astype(str)

    def fit(self, frame):
        if self.numeric:
            self.scaler.fit(self.imputer.fit_transform(frame[self.numeric]))
        if self.categorical:
            self.encoder.fit(self._cats(frame))
        return self

    def transform(self, frame):
        parts = []
        if self.numeric:
            parts.append(self.scaler.transform(self.imputer.transform(frame[self.numeric])))
        if self.categorical:
            parts.append(self.encoder.transform(self._cats(frame)))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def metadata(self):
        # Values/vocabularies can identify a private dataset: release counts/hash only.
        result = {"fit_partition": "train", "numeric_columns": self.numeric, "categorical_columns": self.categorical}
        if self.numeric:
            result["numeric_dimensions_after_imputation"] = len(self.scaler.mean_)
            result["numeric_statistics_sha256"] = hashlib.sha256(
                self.imputer.statistics_.tobytes() + self.scaler.mean_.tobytes() + self.scaler.scale_.tobytes()
            ).hexdigest()
        if self.categorical:
            result["category_counts"] = [len(x) for x in self.encoder.categories_]
            result["category_vocabulary_sha256"] = hashlib.sha256(
                repr([x.tolist() for x in self.encoder.categories_]).encode()
            ).hexdigest()
        return result


def attach_features(cohorts, numeric, categorical, vector_path=None):
    prep = Preprocessor(numeric, categorical).fit(cohorts["train"].frame)
    vectors = {}
    lookup = None
    if vector_path is not None:
        with np.load(vector_path, allow_pickle=False) as archive:
            if "row_ids" not in archive:
                raise ValueError("Vector archive requires row_ids for explicit alignment")
            ids = archive["row_ids"].astype(str)
            if ids.ndim != 1 or len(set(ids)) != len(ids):
                raise ValueError("Vector row_ids must be one-dimensional and unique")
            allowed = {"text", "image", "report", "image_mask", "report_mask"}
            if set(archive.files) - allowed - {"row_ids"}:
                raise ValueError("Unknown vector archive fields")
            vectors = {key: archive[key] for key in archive.files if key != "row_ids"}
            if any(len(a) != len(ids) for a in vectors.values()):
                raise ValueError("Vector row counts do not match row_ids")
            lookup = {value: i for i, value in enumerate(ids)}
    for cohort in cohorts.values():
        cohort.features = {"tabular": prep.transform(cohort.frame)}
        if lookup is not None:
            try:
                indices = np.array([lookup[x] for x in cohort.frame.row_id])
            except KeyError as error:
                raise ValueError("Table row missing from vector archive") from error
            cohort.features.update({key: a[indices] for key, a in vectors.items()})
    return prep.metadata()
