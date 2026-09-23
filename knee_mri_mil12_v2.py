#!/usr/bin/env python3
"""
knee_mri_mil12_v2.py

12-abnormality weakly supervised / hierarchical MIL trainer with robust scanner-domain splitting
for heterogeneous knee MRI studies.

Key design
----------
A label belongs to the STUDY, not to each slice.

Hierarchy:
    Study -> Series -> Slices

For each study:
  1. A CNN encodes sampled slices.
  2. Target-specific attention pools slices within each series.
  3. Target-specific attention pools series within the study.
  4. The network outputs 12 study-level logits/probabilities.

Weak labels
-----------
Expected targets:
    ACL
    MCL
    Medial Meniscus
    Lateral Meniscus
    Medial OA
    Lateral OA
    PF OA
    Effusion
    Synovitis
    Baker's
    Contusion
    Fracture

The label file may contain:
    ACL_P, MCL_P, ..., Fracture_P
or, for reference data:
    ACL, MCL, ..., Fracture

IMPORTANT:
    p = 0.5 means "not addressed / no useful report evidence".
    It is retained as 0.5 in the dataset, but given zero training weight:

        information_weight = 2 * abs(p - 0.5)

Thus:
    p=0.0 or 1.0 -> weight 1.0
    p=0.2 or 0.8 -> weight 0.6
    p=0.3 or 0.7 -> weight 0.4
    p=0.5        -> weight 0.0

Input
-----
1. Census SQLite checkpoint from knee_mri_census_v2.py.
2. Weak-label Excel/CSV created from the report-mining stage.

Recommended first run
---------------------
Audit only:

python knee_mri_mil12_v2.py audit \
  --census "/mnt/y/data_RSNA_4_Study/results/knee_mri_census.xlsx.checkpoint.sqlite" \
  --labels "/mnt/y/data_RSNA_4_Study/WeakLabels.xlsx" \
  --labels-sheet "WeakLabels" \
  --run-dir "/mnt/y/data_RSNA_4_Study/results/mil12_v1"

Then train:

python knee_mri_mil12_v2.py train \
  --census "/mnt/y/data_RSNA_4_Study/results/knee_mri_census.xlsx.checkpoint.sqlite" \
  --labels "/mnt/y/data_RSNA_4_Study/WeakLabels.xlsx" \
  --labels-sheet "WeakLabels" \
  --run-dir "/mnt/y/data_RSNA_4_Study/results/mil12_v1" \
  --epochs 20 \
  --max-series 8 \
  --slices-per-series 12 \
  --num-workers 0

Notes
-----
- Start with --num-workers 0 because large WSL multiprocessing jobs can fill /dev/shm.
- The model never assigns a study label directly to every slice.
- Train/validation/test splitting is always by StudyInstanceUID.
- --split-mode scanner holds out normalized scanner domains
  (canonical vendor + canonical model + field strength).
- --split-mode source now aborts if fewer than 3 source groups exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd


TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

TARGET_TO_SAFE = {
    "ACL": "ACL",
    "MCL": "MCL",
    "Medial Meniscus": "Medial_Meniscus",
    "Lateral Meniscus": "Lateral_Meniscus",
    "Medial OA": "Medial_OA",
    "Lateral OA": "Lateral_OA",
    "PF OA": "PF_OA",
    "Effusion": "Effusion",
    "Synovitis": "Synovitis",
    "Baker's": "Bakers",
    "Contusion": "Contusion",
    "Fracture": "Fracture",
}


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def information_weight(p: np.ndarray) -> np.ndarray:
    """0.5 = unknown => zero information weight."""
    return np.clip(2.0 * np.abs(p - 0.5), 0.0, 1.0)


def auc_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    AUROC without sklearn.
    Uses the Mann-Whitney / rank definition and handles ties.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=float)

    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        # ranks are 1-based; ties receive average rank
        avg_rank = ((i + 1) + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    sum_pos_ranks = ranks[pos].sum()
    u = sum_pos_ranks - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Label loading
# ----------------------------------------------------------------------

def read_labels(path: Path, sheet: str | None) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Label file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        df = pd.read_excel(path, sheet_name=sheet or 0, dtype=str, keep_default_na=False)
    elif suffix == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    else:
        raise SystemExit("Labels must be .xlsx/.xlsm/.xls or .csv")

    if "StudyInstanceUID" not in df.columns:
        raise SystemExit("Label file must contain StudyInstanceUID")

    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str).str.strip()
    df = df[df["StudyInstanceUID"] != ""].copy()
    df = df.drop_duplicates("StudyInstanceUID", keep="first")

    out = pd.DataFrame({"StudyInstanceUID": df["StudyInstanceUID"]})

    missing = []
    for target in TARGETS:
        p_col = f"{target}_P"
        if p_col in df.columns:
            source_col = p_col
        elif target in df.columns:
            source_col = target
        else:
            missing.append(target)
            continue

        vals = pd.to_numeric(df[source_col], errors="coerce")
        vals = vals.clip(0.0, 1.0)
        out[target] = vals

    if missing:
        raise SystemExit(
            "Missing target columns for: " + ", ".join(missing) +
            "\nExpected either <Target>_P or <Target>."
        )

    out = out.dropna(subset=TARGETS, how="all").copy()

    # Important: a missing target cell remains unknown = 0.5.
    out[TARGETS] = out[TARGETS].fillna(0.5)

    return out


# ----------------------------------------------------------------------
# Census loading
# ----------------------------------------------------------------------

REQUIRED_SLICE_COLS = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "File",
    "Plane",
    "SeriesNumber",
    "InstanceNumber",
    "SeriesDescription",
    "SourceTopFolder",
    "Manufacturer",
    "ManufacturerModelName",
    "MagneticFieldStrength",
]


def load_census_slice_index(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        raise SystemExit(f"Census SQLite checkpoint does not exist: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        table_names = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "slices" not in table_names:
            raise SystemExit(f"No 'slices' table in census database: {db_path}")

        cols = {
            row[1] for row in conn.execute("PRAGMA table_info(slices)")
        }
        missing = [c for c in REQUIRED_SLICE_COLS if c not in cols]
        if missing:
            raise SystemExit(
                "Census database is missing required columns: " + ", ".join(missing)
            )

        q = """
            SELECT
                StudyInstanceUID,
                SeriesInstanceUID,
                SOPInstanceUID,
                File,
                Plane,
                SeriesNumber,
                InstanceNumber,
                SeriesDescription,
                SourceTopFolder,
                Manufacturer,
                ManufacturerModelName,
                MagneticFieldStrength
            FROM slices
        """
        sl = pd.read_sql_query(q, conn)
    finally:
        conn.close()

    sl["StudyInstanceUID"] = sl["StudyInstanceUID"].astype(str)
    sl["SeriesInstanceUID"] = sl["SeriesInstanceUID"].astype(str)
    sl["File"] = sl["File"].astype(str)
    sl["Plane"] = sl["Plane"].fillna("Unknown").astype(str)
    sl["SourceTopFolder"] = sl["SourceTopFolder"].fillna("").astype(str)
    sl["SeriesDescription"] = sl["SeriesDescription"].fillna("").astype(str)
    sl["Manufacturer"] = sl["Manufacturer"].fillna("").astype(str)
    sl["ManufacturerModelName"] = sl["ManufacturerModelName"].fillna("").astype(str)
    sl["MagneticFieldStrength"] = pd.to_numeric(
        sl["MagneticFieldStrength"], errors="coerce"
    )
    sl["SeriesNumber"] = pd.to_numeric(sl["SeriesNumber"], errors="coerce")
    sl["InstanceNumber"] = pd.to_numeric(sl["InstanceNumber"], errors="coerce")

    return sl


# ----------------------------------------------------------------------
# Scanner-domain normalization
# ----------------------------------------------------------------------

def _clean_token(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.replace("_", " ").strip().upper()
    return " ".join(s.split())


def canonical_vendor(x: Any) -> str:
    s = _clean_token(x)
    if "SIEMENS" in s:
        return "SIEMENS"
    if "PHILIPS" in s:
        return "PHILIPS"
    if s == "GEHC" or "GE MEDICAL" in s or s.startswith("GE "):
        return "GE"
    if "TOSHIBA" in s:
        return "TOSHIBA"
    if "CANON" in s:
        return "CANON"
    if "FUJIFILM" in s:
        return "FUJIFILM"
    if "HITACHI" in s:
        return "HITACHI"
    return s if s else "UNKNOWN_VENDOR"


def canonical_model(vendor: str, x: Any) -> str:
    s = _clean_token(x)
    if not s:
        return "UNKNOWN_MODEL"

    if vendor == "SIEMENS" and s.startswith("MAGNETOM "):
        s = s[len("MAGNETOM "):].strip()

    if vendor == "GE" and s.startswith("SIGNA "):
        s = s[len("SIGNA "):].strip()

    return s if s else "UNKNOWN_MODEL"


def canonical_field_strength(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return "UNK_T"

    if not np.isfinite(v) or v <= 0:
        return "UNK_T"

    # These data contain common values such as 1.16, 1.5 and 3.0 T.
    return f"{v:.2f}T"


def scanner_domain_from_values(
    manufacturer: Any,
    model: Any,
    field_strength: Any,
) -> str:
    vendor = canonical_vendor(manufacturer)
    model2 = canonical_model(vendor, model)
    field = canonical_field_strength(field_strength)
    return f"{vendor}|{model2}|{field}"


def add_scanner_domain(sl: pd.DataFrame) -> pd.DataFrame:
    sl = sl.copy()
    sl["ScannerDomain"] = [
        scanner_domain_from_values(m, model, fs)
        for m, model, fs in zip(
            sl["Manufacturer"],
            sl["ManufacturerModelName"],
            sl["MagneticFieldStrength"],
        )
    ]
    return sl


# ----------------------------------------------------------------------
# Study manifest
# ----------------------------------------------------------------------

@dataclass
class SeriesRecord:
    series_uid: str
    plane: str
    series_number: float
    series_description: str
    files: List[str]
    instance_numbers: List[float]


@dataclass
class StudyRecord:
    study_uid: str
    source: str
    scanner_domain: str
    series: List[SeriesRecord]
    targets: np.ndarray


def build_study_records(
    sl: pd.DataFrame,
    labels: pd.DataFrame,
) -> Tuple[List[StudyRecord], pd.DataFrame]:
    label_map = labels.set_index("StudyInstanceUID")[TARGETS]

    imaging_uids = set(sl["StudyInstanceUID"])
    label_uids = set(labels["StudyInstanceUID"])
    common = imaging_uids & label_uids

    if not common:
        raise SystemExit("No StudyInstanceUID overlap between census and label file.")

    sl2 = sl[sl["StudyInstanceUID"].isin(common)].copy()
    sl2 = add_scanner_domain(sl2)

    records: List[StudyRecord] = []
    manifest_rows = []

    for study_uid, sdf in sl2.groupby("StudyInstanceUID", sort=False):
        series_records = []

        source_values = [x for x in sdf["SourceTopFolder"].astype(str).unique() if x]
        source = source_values[0] if source_values else ""

        scanner_mode = sdf["ScannerDomain"].mode()
        scanner_domain = (
            str(scanner_mode.iloc[0])
            if len(scanner_mode)
            else "UNKNOWN_VENDOR|UNKNOWN_MODEL|UNK_T"
        )

        for series_uid, g in sdf.groupby("SeriesInstanceUID", sort=False):
            g = g.copy()

            # Stable within-series ordering.
            if g["InstanceNumber"].notna().any():
                g = g.sort_values(["InstanceNumber", "File"], na_position="last")
            else:
                g = g.sort_values("File")

            plane_mode = g["Plane"].mode()
            plane = str(plane_mode.iloc[0]) if len(plane_mode) else "Unknown"

            sn = g["SeriesNumber"].dropna()
            series_number = float(sn.iloc[0]) if len(sn) else float("nan")

            desc = ""
            desc_vals = [x for x in g["SeriesDescription"].astype(str).unique() if x]
            if desc_vals:
                desc = desc_vals[0]

            sr = SeriesRecord(
                series_uid=str(series_uid),
                plane=plane,
                series_number=series_number,
                series_description=desc,
                files=g["File"].astype(str).tolist(),
                instance_numbers=g["InstanceNumber"].fillna(np.nan).astype(float).tolist(),
            )
            series_records.append(sr)

            manifest_rows.append({
                "StudyInstanceUID": study_uid,
                "SeriesInstanceUID": str(series_uid),
                "SourceTopFolder": source,
                "ScannerDomain": scanner_domain,
                "Manufacturer": str(g["Manufacturer"].mode().iloc[0]) if len(g["Manufacturer"].mode()) else "",
                "ManufacturerModelName": str(g["ManufacturerModelName"].mode().iloc[0]) if len(g["ManufacturerModelName"].mode()) else "",
                "MagneticFieldStrength": float(g["MagneticFieldStrength"].dropna().median()) if g["MagneticFieldStrength"].notna().any() else np.nan,
                "Plane": plane,
                "SeriesNumber": series_number,
                "SeriesDescription": desc,
                "SliceCount": len(g),
            })

        target_vec = label_map.loc[study_uid].to_numpy(dtype=np.float32)

        records.append(
            StudyRecord(
                study_uid=study_uid,
                source=source,
                scanner_domain=scanner_domain,
                series=series_records,
                targets=target_vec,
            )
        )

    manifest = pd.DataFrame(manifest_rows)
    return records, manifest


def _group_split_score(
    stats: Dict[str, Dict[str, Any]],
    target_n: Dict[str, float],
    target_info: Dict[str, np.ndarray],
    target_pos: Dict[str, np.ndarray],
) -> float:
    score = 0.0
    for name in ("train", "val", "test"):
        tn = max(target_n[name], 1.0)
        score += ((stats[name]["n"] - target_n[name]) / tn) ** 2

        ti = np.maximum(target_info[name], 1.0)
        tp = np.maximum(target_pos[name], 1.0)

        score += 0.30 * float(np.mean(
            ((stats[name]["info"] - target_info[name]) / ti) ** 2
        ))
        score += 0.20 * float(np.mean(
            ((stats[name]["pos"] - target_pos[name]) / tp) ** 2
        ))

    # A held-out split based on one gigantic group is legal but less useful.
    # Prefer at least two domains in validation/test when feasible.
    for name in ("val", "test"):
        if stats[name]["groups"] < 2:
            score += 5.0

    if stats["train"]["groups"] < 2:
        score += 20.0

    return score


def grouped_holdout_split(
    records: List[StudyRecord],
    key_name: str,
    key_fn,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    restarts: int = 800,
) -> Tuple[List[StudyRecord], List[StudyRecord], List[StudyRecord]]:
    by_group: Dict[str, List[StudyRecord]] = defaultdict(list)
    for r in records:
        key = str(key_fn(r) or "").strip()
        if not key:
            key = f"__UNKNOWN_{key_name.upper()}__"
        by_group[key].append(r)

    if len(by_group) < 3:
        sizes = sorted(
            ((k, len(v)) for k, v in by_group.items()),
            key=lambda x: x[1],
            reverse=True,
        )
        details = ", ".join(f"{k}={n}" for k, n in sizes[:10])
        raise SystemExit(
            f"--split-mode {key_name} requires at least 3 distinct groups, "
            f"but only {len(by_group)} were found. Groups: {details}"
        )

    fractions = {
        "train": 1.0 - val_fraction - test_fraction,
        "val": val_fraction,
        "test": test_fraction,
    }

    total_n = len(records)
    global_info = np.zeros(len(TARGETS), dtype=np.float64)
    global_pos = np.zeros(len(TARGETS), dtype=np.float64)
    for r in records:
        y = r.targets.astype(np.float64)
        w = 2.0 * np.abs(y - 0.5)
        global_info += w
        global_pos += w * y

    target_n = {k: total_n * f for k, f in fractions.items()}
    target_info = {k: global_info * f for k, f in fractions.items()}
    target_pos = {k: global_pos * f for k, f in fractions.items()}

    group_payload = {}
    for key, rs in by_group.items():
        info = np.zeros(len(TARGETS), dtype=np.float64)
        pos = np.zeros(len(TARGETS), dtype=np.float64)
        for r in rs:
            y = r.targets.astype(np.float64)
            w = 2.0 * np.abs(y - 0.5)
            info += w
            pos += w * y
        group_payload[key] = {
            "n": len(rs),
            "info": info,
            "pos": pos,
        }

    rng = random.Random(seed)
    best_assignment = None
    best_score = float("inf")

    names = ("train", "val", "test")
    keys = list(by_group)

    for restart in range(max(1, int(restarts))):
        # Large groups are placed first, with modest jitter to explore
        # several near-optimal group combinations.
        ordered = sorted(
            keys,
            key=lambda k: (
                group_payload[k]["n"] * (0.85 + 0.30 * rng.random())
            ),
            reverse=True,
        )

        stats = {
            name: {
                "n": 0,
                "info": np.zeros(len(TARGETS), dtype=np.float64),
                "pos": np.zeros(len(TARGETS), dtype=np.float64),
                "groups": 0,
            }
            for name in names
        }
        assignment = {}

        for key in ordered:
            payload = group_payload[key]
            candidate_scores = []

            for name in names:
                # Temporarily place the domain in this split.
                stats[name]["n"] += payload["n"]
                stats[name]["info"] += payload["info"]
                stats[name]["pos"] += payload["pos"]
                stats[name]["groups"] += 1

                sc = _group_split_score(
                    stats, target_n, target_info, target_pos
                )
                candidate_scores.append((sc, rng.random(), name))

                stats[name]["n"] -= payload["n"]
                stats[name]["info"] -= payload["info"]
                stats[name]["pos"] -= payload["pos"]
                stats[name]["groups"] -= 1

            _, _, chosen = min(candidate_scores)

            assignment[key] = chosen
            stats[chosen]["n"] += payload["n"]
            stats[chosen]["info"] += payload["info"]
            stats[chosen]["pos"] += payload["pos"]
            stats[chosen]["groups"] += 1

        sc = _group_split_score(stats, target_n, target_info, target_pos)
        if sc < best_score:
            best_score = sc
            best_assignment = assignment.copy()

    if best_assignment is None:
        raise RuntimeError("Failed to construct grouped holdout split.")

    parts = {"train": [], "val": [], "test": []}
    for key, rs in by_group.items():
        parts[best_assignment[key]].extend(rs)

    if not parts["train"] or not parts["val"] or not parts["test"]:
        raise SystemExit(
            f"{key_name} split produced an empty partition; aborting rather than training."
        )

    return parts["train"], parts["val"], parts["test"]


def split_records(
    records: List[StudyRecord],
    seed: int,
    val_fraction: float,
    test_fraction: float,
    mode: str,
) -> Tuple[List[StudyRecord], List[StudyRecord], List[StudyRecord], pd.DataFrame]:
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise SystemExit("val_fraction and test_fraction must be >=0 and sum to <1.")

    rng = random.Random(seed)

    if mode == "source":
        train, val, test = grouped_holdout_split(
            records,
            key_name="source",
            key_fn=lambda r: r.source,
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )

    elif mode == "scanner":
        train, val, test = grouped_holdout_split(
            records,
            key_name="scanner",
            key_fn=lambda r: r.scanner_domain,
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )

    else:
        shuffled = records[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = int(round(n * test_fraction))
        n_val = int(round(n * val_fraction))
        test = shuffled[:n_test]
        val = shuffled[n_test:n_test + n_val]
        train = shuffled[n_test + n_val:]

    split_rows = []
    for name, part in [("train", train), ("val", val), ("test", test)]:
        for r in part:
            row = {
                "StudyInstanceUID": r.study_uid,
                "split": name,
                "SourceTopFolder": r.source,
                "ScannerDomain": r.scanner_domain,
                "SeriesCount": len(r.series),
                "TotalSlices": sum(len(s.files) for s in r.series),
            }
            for k, t in enumerate(TARGETS):
                row[f"{t}_P"] = float(r.targets[k])
                row[f"{t}_W"] = float(2.0 * abs(r.targets[k] - 0.5))
            split_rows.append(row)

    return train, val, test, pd.DataFrame(split_rows)


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------

def choose_series(
    series_list: List[SeriesRecord],
    max_series: int,
    train_mode: bool,
    rng: random.Random,
) -> List[SeriesRecord]:
    if max_series <= 0 or len(series_list) <= max_series:
        return list(series_list)

    # Preserve plane diversity first.
    by_plane: Dict[str, List[SeriesRecord]] = defaultdict(list)
    for s in series_list:
        by_plane[s.plane].append(s)

    chosen = []
    planes = list(by_plane.keys())
    if train_mode:
        rng.shuffle(planes)
    else:
        planes = sorted(planes)

    # One series per plane first.
    for p in planes:
        candidates = by_plane[p][:]
        if train_mode:
            rng.shuffle(candidates)
            chosen.append(candidates[0])
        else:
            candidates.sort(
                key=lambda x: (
                    math.inf if math.isnan(x.series_number) else x.series_number,
                    x.series_uid,
                )
            )
            chosen.append(candidates[0])
        if len(chosen) >= max_series:
            return chosen[:max_series]

    remaining = [s for s in series_list if s not in chosen]
    if train_mode:
        rng.shuffle(remaining)
    else:
        remaining.sort(
            key=lambda x: (
                x.plane,
                math.inf if math.isnan(x.series_number) else x.series_number,
                x.series_uid,
            )
        )

    chosen.extend(remaining[: max_series - len(chosen)])
    return chosen


def evenly_spaced_indices(n: int, k: int) -> List[int]:
    if n <= 0:
        return []
    if k <= 0 or n <= k:
        return list(range(n))
    idx = np.linspace(0, n - 1, k)
    return sorted(set(int(round(x)) for x in idx))


def jitter_indices(n: int, k: int, rng: random.Random) -> List[int]:
    if n <= 0:
        return []
    if k <= 0 or n <= k:
        idx = list(range(n))
        rng.shuffle(idx)
        return sorted(idx)

    # Divide the stack into k bins and sample one slice from each bin.
    edges = np.linspace(0, n, k + 1)
    out = []
    for i in range(k):
        lo = int(math.floor(edges[i]))
        hi = int(math.ceil(edges[i + 1])) - 1
        lo = max(0, min(lo, n - 1))
        hi = max(lo, min(hi, n - 1))
        out.append(rng.randint(lo, hi))
    return sorted(set(out))


# ----------------------------------------------------------------------
# Torch / image code is imported lazily so "audit" stays lightweight.
# ----------------------------------------------------------------------

def import_training_stack():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import Dataset, DataLoader
    except ModuleNotFoundError as e:
        raise SystemExit(
            "PyTorch is required for training.\n"
            "Install a CUDA-enabled PyTorch build appropriate for your system."
        ) from e

    try:
        import torchvision
        from torchvision.models import resnet18, ResNet18_Weights
    except ModuleNotFoundError as e:
        raise SystemExit("torchvision is required: pip install torchvision") from e

    try:
        import pydicom
    except ModuleNotFoundError as e:
        raise SystemExit("pydicom is required: pip install pydicom") from e

    return torch, nn, F, Dataset, DataLoader, torchvision, resnet18, ResNet18_Weights, pydicom


def make_training_classes():
    torch, nn, F, Dataset, DataLoader, torchvision, resnet18, ResNet18_Weights, pydicom = import_training_stack()

    class StudyDataset(Dataset):
        def __init__(
            self,
            records: List[StudyRecord],
            max_series: int,
            slices_per_series: int,
            image_size: int,
            train_mode: bool,
            seed: int,
        ):
            self.records = records
            self.max_series = max_series
            self.slices_per_series = slices_per_series
            self.image_size = image_size
            self.train_mode = train_mode
            self.seed = seed
            self.epoch = 0

        def set_epoch(self, epoch: int):
            self.epoch = int(epoch)

        def __len__(self):
            return len(self.records)

        def _read_slice(self, path: str):
            ds = pydicom.dcmread(path, force=False)
            arr = ds.pixel_array.astype(np.float32)

            slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
            intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
            arr = arr * slope + intercept

            # MONOCHROME1 means larger values display darker; invert.
            photo = str(getattr(ds, "PhotometricInterpretation", "")).upper()
            if photo == "MONOCHROME1":
                arr = arr.max() + arr.min() - arr

            finite = np.isfinite(arr)
            if not finite.any():
                arr = np.zeros_like(arr, dtype=np.float32)
            else:
                vals = arr[finite]
                lo, hi = np.percentile(vals, [1, 99])
                if not np.isfinite(lo):
                    lo = float(vals.min())
                if not np.isfinite(hi):
                    hi = float(vals.max())
                if hi <= lo:
                    hi = lo + 1.0
                arr = np.clip(arr, lo, hi)
                arr = (arr - lo) / (hi - lo)

            x = torch.from_numpy(arr).float()
            if x.ndim == 3:
                # Defensive: use the first frame if a multiframe object appears.
                x = x[0]
            x = x.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            x = x.squeeze(0)  # [1,H,W]
            x = x.repeat(3, 1, 1)

            # ImageNet normalization for pretrained ResNet.
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype)[:, None, None]
            std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype)[:, None, None]
            x = (x - mean) / std
            return x

        def __getitem__(self, index: int):
            rec = self.records[index]
            rng = random.Random(self.seed + self.epoch * 1_000_003 + index)

            selected_series = choose_series(
                rec.series,
                self.max_series,
                self.train_mode,
                rng,
            )

            series_tensors = []
            series_meta = []

            for s in selected_series:
                n = len(s.files)
                if self.train_mode:
                    idx = jitter_indices(n, self.slices_per_series, rng)
                else:
                    idx = evenly_spaced_indices(n, self.slices_per_series)

                imgs = []
                kept_files = []
                kept_indices = []

                for j in idx:
                    try:
                        imgs.append(self._read_slice(s.files[j]))
                        kept_files.append(s.files[j])
                        kept_indices.append(j)
                    except Exception:
                        # One unreadable pixel object should not destroy a study.
                        continue

                if not imgs:
                    continue

                series_tensors.append(torch.stack(imgs, dim=0))
                series_meta.append({
                    "SeriesInstanceUID": s.series_uid,
                    "Plane": s.plane,
                    "SeriesDescription": s.series_description,
                    "Files": kept_files,
                    "WithinSeriesIndices": kept_indices,
                })

            if not series_tensors:
                raise RuntimeError(f"No readable pixel slices for study {rec.study_uid}")

            y = torch.tensor(rec.targets, dtype=torch.float32)
            w = 2.0 * torch.abs(y - 0.5)
            w = torch.clamp(w, 0.0, 1.0)

            return {
                "study_uid": rec.study_uid,
                "source": rec.source,
                "scanner_domain": rec.scanner_domain,
                "series": series_tensors,
                "series_meta": series_meta,
                "targets": y,
                "weights": w,
            }

    class HierarchicalMIL12(nn.Module):
        def __init__(
            self,
            embedding_dim: int = 512,
            attention_dim: int = 128,
            pretrained: bool = True,
            freeze_encoder: bool = True,
            encoder_batch: int = 32,
        ):
            super().__init__()
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.encoder = backbone

            self.proj = (
                nn.Identity()
                if in_features == embedding_dim
                else nn.Linear(in_features, embedding_dim)
            )

            self.embedding_dim = embedding_dim
            self.num_targets = len(TARGETS)
            self.encoder_batch = int(encoder_batch)

            # Target-specific slice attention.
            self.slice_attn_h = nn.Linear(embedding_dim, attention_dim)
            self.slice_attn_v = nn.Linear(attention_dim, self.num_targets, bias=False)

            # Target-specific series attention.
            self.series_attn_h = nn.Linear(embedding_dim, attention_dim)
            self.series_attn_v = nn.Linear(attention_dim, self.num_targets, bias=False)

            # One classifier vector per target.
            self.classifier_weight = nn.Parameter(
                torch.empty(self.num_targets, embedding_dim)
            )
            self.classifier_bias = nn.Parameter(torch.zeros(self.num_targets))
            nn.init.xavier_uniform_(self.classifier_weight)

            if freeze_encoder:
                for p in self.encoder.parameters():
                    p.requires_grad = False

        def encode_slices(self, x):
            chunks = []
            for start in range(0, x.shape[0], self.encoder_batch):
                z = self.encoder(x[start:start + self.encoder_batch])
                z = self.proj(z)
                chunks.append(z)
            return torch.cat(chunks, dim=0)

        def forward(self, series_list):
            """
            series_list: list of tensors, each [n_slices, 3, H, W]

            Returns:
                logits: [12]
                attention dictionary
            """
            target_series_embeddings = []
            slice_attention_out = []

            for x in series_list:
                emb = self.encode_slices(x)  # [N,D]

                # [N,A] -> [N,K]
                a = torch.tanh(self.slice_attn_h(emb))
                scores = self.slice_attn_v(a)
                alpha = torch.softmax(scores, dim=0)

                # For each target k: sum_n alpha[n,k] * emb[n,:]
                # -> [K,D]
                z_series = torch.einsum("nk,nd->kd", alpha, emb)

                target_series_embeddings.append(z_series)
                slice_attention_out.append(alpha)

            # [S,K,D]
            zs = torch.stack(target_series_embeddings, dim=0)

            # Target-specific series attention.
            h = torch.tanh(self.series_attn_h(zs))   # [S,K,A]
            # series_attn_v maps A -> K, but we only want diagonal target scores.
            # Compute all K scores and select matching target.
            all_scores = self.series_attn_v(h)       # [S,K,K]
            idx = torch.arange(self.num_targets, device=zs.device)
            s_scores = all_scores[:, idx, idx]       # [S,K]
            beta = torch.softmax(s_scores, dim=0)    # across series

            # [K,D]
            study_target_emb = torch.einsum("sk,skd->kd", beta, zs)

            logits = (
                (study_target_emb * self.classifier_weight).sum(dim=1)
                + self.classifier_bias
            )

            return logits, {
                "slice_attention": slice_attention_out,
                "series_attention": beta,
            }

    return torch, nn, F, Dataset, DataLoader, StudyDataset, HierarchicalMIL12


# ----------------------------------------------------------------------
# Loss / metrics
# ----------------------------------------------------------------------

def weighted_soft_bce(torch, F, logits, targets, weights):
    """
    Soft-label BCE with information weights.

    p=0.5 stays in targets, but w=0 makes it non-informative.
    """
    per_target = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    denom = weights.sum()
    if float(denom.detach().cpu()) <= 0:
        # Preserve graph with zero loss.
        return logits.sum() * 0.0
    return (per_target * weights).sum() / denom


def summarize_label_distribution(records: List[StudyRecord]) -> pd.DataFrame:
    arr = np.stack([r.targets for r in records], axis=0)
    rows = []
    for k, target in enumerate(TARGETS):
        p = arr[:, k]
        rows.append({
            "Target": target,
            "N": len(p),
            "MeanP": float(np.mean(p)),
            "P<=0.20": int((p <= 0.20).sum()),
            "0.20<P<0.50": int(((p > 0.20) & (p < 0.50)).sum()),
            "P=0.50": int(np.isclose(p, 0.50).sum()),
            "0.50<P<0.80": int(((p > 0.50) & (p < 0.80)).sum()),
            "P>=0.80": int((p >= 0.80).sum()),
            "MeanInfoWeight": float(information_weight(p).mean()),
        })
    return pd.DataFrame(rows)


def evaluate_model(
    torch,
    F,
    model,
    loader,
    device,
    high_conf_threshold: float,
):
    model.eval()

    losses = []
    all_targets = []
    all_logits = []
    all_weights = []
    all_uids = []

    with torch.no_grad():
        for sample in loader:
            series = [x.to(device, non_blocking=True) for x in sample["series"]]
            y = sample["targets"].to(device)
            w = sample["weights"].to(device)

            logits, _ = model(series)
            loss = weighted_soft_bce(torch, F, logits, y, w)

            losses.append(float(loss.detach().cpu()))
            all_targets.append(y.detach().cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())
            all_weights.append(w.detach().cpu().numpy())
            all_uids.append(sample["study_uid"])

    targets = np.stack(all_targets)
    logits = np.stack(all_logits)
    probs = sigmoid_np(logits)
    weights = np.stack(all_weights)

    metrics = []
    for k, target in enumerate(TARGETS):
        p = targets[:, k]
        pred = probs[:, k]

        # Evaluate AUC only where weak labels are sufficiently informative.
        informative = (p <= (1.0 - high_conf_threshold)) | (p >= high_conf_threshold)
        y = (p[informative] > 0.5).astype(int)
        score = pred[informative]

        auc = auc_rank(y, score) if informative.sum() > 1 else float("nan")

        metrics.append({
            "Target": target,
            "AUC_high_conf": auc,
            "N_high_conf": int(informative.sum()),
            "N_pos_high_conf": int(y.sum()) if len(y) else 0,
            "MeanPred": float(pred.mean()),
            "MeanLabelP": float(p.mean()),
            "MeanInfoWeight": float(weights[:, k].mean()),
        })

    return float(np.mean(losses)) if losses else float("nan"), pd.DataFrame(metrics), {
        "uids": all_uids,
        "targets": targets,
        "probs": probs,
        "weights": weights,
    }


# ----------------------------------------------------------------------
# Attention export
# ----------------------------------------------------------------------

def export_attention(
    torch,
    model,
    records: List[StudyRecord],
    dataset,
    device,
    out_csv: Path,
    top_slices: int,
):
    """
    Export top attended series and slices for each target/study.
    Intended for audit, not as ground-truth localization.
    """
    model.eval()
    rows = []

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            series = [x.to(device) for x in sample["series"]]
            logits, att = model(series)
            probs = torch.sigmoid(logits).cpu().numpy()

            beta = att["series_attention"].detach().cpu().numpy()  # [S,K]

            for k, target in enumerate(TARGETS):
                s_idx = int(np.argmax(beta[:, k]))
                s_meta = sample["series_meta"][s_idx]
                alpha = att["slice_attention"][s_idx][:, k].detach().cpu().numpy()

                order = np.argsort(-alpha)[:max(1, top_slices)]
                top_files = [s_meta["Files"][j] for j in order]
                top_weights = [float(alpha[j]) for j in order]

                rows.append({
                    "StudyInstanceUID": sample["study_uid"],
                    "Target": target,
                    "PredictedProbability": float(probs[k]),
                    "TargetProbability": float(sample["targets"][k]),
                    "InformationWeight": float(sample["weights"][k]),
                    "TopSeriesInstanceUID": s_meta["SeriesInstanceUID"],
                    "TopSeriesPlane": s_meta["Plane"],
                    "TopSeriesDescription": s_meta["SeriesDescription"],
                    "SeriesAttention": float(beta[s_idx, k]),
                    "TopSliceFiles": " | ".join(top_files),
                    "TopSliceAttention": " | ".join(f"{x:.6f}" for x in top_weights),
                })

    ensure_parent(out_csv)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------

def prepare_inputs(args):
    seed_everything(args.seed)

    labels = read_labels(Path(args.labels), args.labels_sheet)
    print(f"Label studies: {len(labels):,}")

    print("Loading census slice index ...")
    sl = load_census_slice_index(Path(args.census))
    print(
        f"Census: {sl['StudyInstanceUID'].nunique():,} studies | "
        f"{sl['SeriesInstanceUID'].nunique():,} series | "
        f"{len(sl):,} slices"
    )

    records, series_manifest = build_study_records(sl, labels)
    print(f"Studies with both imaging and labels: {len(records):,}")

    train, val, test, split_df = split_records(
        records,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        mode=args.split_mode,
    )

    return records, train, val, test, split_df, series_manifest


def command_audit(args):
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    records, train, val, test, split_df, series_manifest = prepare_inputs(args)

    label_summary = summarize_label_distribution(records)

    study_rows = []
    for r in records:
        row = {
            "StudyInstanceUID": r.study_uid,
            "SourceTopFolder": r.source,
            "ScannerDomain": r.scanner_domain,
            "SeriesCount": len(r.series),
            "TotalSlices": sum(len(s.files) for s in r.series),
            "Planes": ", ".join(sorted(set(s.plane for s in r.series))),
        }
        for k, target in enumerate(TARGETS):
            row[f"{target}_P"] = float(r.targets[k])
            row[f"{target}_W"] = float(2.0 * abs(r.targets[k] - 0.5))
        study_rows.append(row)

    study_manifest = pd.DataFrame(study_rows)

    split_df.to_csv(run_dir / "split.csv", index=False)
    series_manifest.to_csv(run_dir / "series_manifest.csv", index=False)
    study_manifest.to_csv(run_dir / "study_manifest.csv", index=False)
    label_summary.to_csv(run_dir / "label_summary.csv", index=False)

    domain_rows = []
    for split_name, part in [("train", train), ("val", val), ("test", test)]:
        counts = pd.Series(
            [r.scanner_domain for r in part], dtype="object"
        ).value_counts()
        for domain, n in counts.items():
            domain_rows.append({
                "split": split_name,
                "ScannerDomain": domain,
                "Studies": int(n),
            })
    scanner_split = pd.DataFrame(domain_rows)
    scanner_split.to_csv(run_dir / "scanner_domain_split.csv", index=False)

    # Per-split target supervision audit.
    split_label_rows = []
    for split_name, part in [("train", train), ("val", val), ("test", test)]:
        for k, target in enumerate(TARGETS):
            p = np.array([float(r.targets[k]) for r in part], dtype=float)
            w = 2.0 * np.abs(p - 0.5)
            split_label_rows.append({
                "split": split_name,
                "Target": target,
                "N": len(p),
                "MeanP": float(np.mean(p)) if len(p) else np.nan,
                "MeanInfoWeight": float(np.mean(w)) if len(w) else np.nan,
                "InformativeN_WgtGT0": int(np.sum(w > 0)),
                "StrongLow_Ple0.2": int(np.sum(p <= 0.2)),
                "StrongHigh_Pge0.8": int(np.sum(p >= 0.8)),
            })
    pd.DataFrame(split_label_rows).to_csv(
        run_dir / "split_label_summary.csv", index=False
    )

    print("\nSplit")
    print(f"  train: {len(train):,}")
    print(f"  val  : {len(val):,}")
    print(f"  test : {len(test):,}")

    if args.split_mode == "scanner":
        print("\nScanner domains by split")
        for split_name, part in [("train", train), ("val", val), ("test", test)]:
            domains = sorted(set(r.scanner_domain for r in part))
            print(f"  {split_name:5}: {len(domains):2d} domains")
            for d in domains:
                n = sum(1 for r in part if r.scanner_domain == d)
                print(f"           {n:4d}  {d}")

    print("\n12-target label summary")
    print(label_summary.to_string(index=False))

    print(f"\nAudit files written to: {run_dir}")
    print("No images were decoded and no model was trained.")


def command_train(args):
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    records, train_records, val_records, test_records, split_df, series_manifest = prepare_inputs(args)

    if len(train_records) == 0 or len(val_records) == 0:
        raise SystemExit("Train and validation splits must both be non-empty.")

    # Save manifests before training.
    split_df.to_csv(run_dir / "split.csv", index=False)
    series_manifest.to_csv(run_dir / "series_manifest.csv", index=False)
    summarize_label_distribution(records).to_csv(
        run_dir / "label_summary.csv", index=False
    )

    torch, nn, F, Dataset, DataLoader, StudyDataset, HierarchicalMIL12 = make_training_classes()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    if args.precision == "bf16" and device.type == "cuda":
        amp_dtype = torch.bfloat16
        use_amp = True
    elif args.precision == "fp16" and device.type == "cuda":
        amp_dtype = torch.float16
        use_amp = True
    else:
        amp_dtype = torch.float32
        use_amp = False

    train_ds = StudyDataset(
        train_records,
        max_series=args.max_series,
        slices_per_series=args.slices_per_series,
        image_size=args.image_size,
        train_mode=True,
        seed=args.seed,
    )
    val_ds = StudyDataset(
        val_records,
        max_series=args.max_series,
        slices_per_series=args.slices_per_series,
        image_size=args.image_size,
        train_mode=False,
        seed=args.seed + 10_000,
    )
    test_ds = StudyDataset(
        test_records,
        max_series=args.max_series,
        slices_per_series=args.slices_per_series,
        image_size=args.image_size,
        train_mode=False,
        seed=args.seed + 20_000,
    )

    # batch_size=1 because a study contains a variable hierarchy of series/slices.
    # custom collate returns the single study dict directly.
    collate_one = lambda batch: batch[0]

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_one,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_one,
        persistent_workers=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_one,
        persistent_workers=False,
    )

    model = HierarchicalMIL12(
        embedding_dim=args.embedding_dim,
        attention_dim=args.attention_dim,
        pretrained=not args.no_pretrained,
        freeze_encoder=args.freeze_encoder,
        encoder_batch=args.encoder_batch,
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = None
    if use_amp and amp_dtype == torch.float16:
        scaler = torch.amp.GradScaler("cuda")

    config = vars(args).copy()
    config["targets"] = TARGETS
    config["device"] = str(device)
    config["torch_version"] = torch.__version__
    if device.type == "cuda":
        config["gpu"] = torch.cuda.get_device_name(0)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    history_rows = []
    best_val = float("inf")
    best_path = run_dir / "best.pt"

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        model.train()

        epoch_start = time.perf_counter()
        running = []
        optimizer.zero_grad(set_to_none=True)

        for step, sample in enumerate(train_loader, start=1):
            series = [x.to(device, non_blocking=True) for x in sample["series"]]
            y = sample["targets"].to(device)
            w = sample["weights"].to(device)

            if use_amp:
                with torch.autocast(
                    device_type="cuda",
                    dtype=amp_dtype,
                    enabled=True,
                ):
                    logits, _ = model(series)
                    loss = weighted_soft_bce(torch, F, logits, y, w)
                    loss = loss / args.accum_steps
            else:
                logits, _ = model(series)
                loss = weighted_soft_bce(torch, F, logits, y, w)
                loss = loss / args.accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % args.accum_steps == 0 or step == len(train_loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running.append(float(loss.detach().cpu()) * args.accum_steps)

            if step % args.log_every == 0:
                elapsed = time.perf_counter() - epoch_start
                print(
                    f"Epoch {epoch:02d}/{args.epochs} | "
                    f"study {step:,}/{len(train_loader):,} | "
                    f"loss {np.mean(running[-args.log_every:]):.4f} | "
                    f"elapsed {format_seconds(elapsed)}"
                )

        train_loss = float(np.mean(running)) if running else float("nan")
        val_loss, val_metrics, _ = evaluate_model(
            torch, F, model, val_loader, device, args.high_conf_threshold
        )

        macro_auc = float(
            np.nanmean(val_metrics["AUC_high_conf"].to_numpy(dtype=float))
        )

        epoch_elapsed = time.perf_counter() - epoch_start

        print(
            f"\nEpoch {epoch:02d} complete | "
            f"train loss={train_loss:.4f} | "
            f"val loss={val_loss:.4f} | "
            f"macro AUC(high-conf)={macro_auc:.4f} | "
            f"time={format_seconds(epoch_elapsed)}"
        )

        print(
            val_metrics[
                ["Target", "AUC_high_conf", "N_high_conf", "N_pos_high_conf"]
            ].to_string(index=False)
        )
        print()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "macro_auc_high_conf": macro_auc,
            "epoch_seconds": epoch_elapsed,
        }
        for _, m in val_metrics.iterrows():
            safe = TARGET_TO_SAFE[m["Target"]]
            row[f"auc_{safe}"] = m["AUC_high_conf"]
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(run_dir / "history.csv", index=False)

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "targets": TARGETS,
            "val_loss": val_loss,
            "macro_auc_high_conf": macro_auc,
        }
        torch.save(checkpoint, run_dir / "last.pt")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, best_path)
            val_metrics.to_csv(run_dir / "best_val_metrics.csv", index=False)
            print(f"Saved new best checkpoint: {best_path}")

    # ------------------------------------------------------------------
    # Final best-checkpoint evaluation
    # ------------------------------------------------------------------
    print("\nLoading best checkpoint for final evaluation ...")
    best = torch.load(best_path, map_location=device)
    model.load_state_dict(best["model"])

    val_loss, val_metrics, val_raw = evaluate_model(
        torch, F, model, val_loader, device, args.high_conf_threshold
    )
    val_metrics.to_csv(run_dir / "final_val_metrics.csv", index=False)

    if len(test_records):
        test_loss, test_metrics, test_raw = evaluate_model(
            torch, F, model, test_loader, device, args.high_conf_threshold
        )
        test_metrics.to_csv(run_dir / "test_metrics.csv", index=False)
        print(f"Test loss: {test_loss:.4f}")
        print(
            test_metrics[
                ["Target", "AUC_high_conf", "N_high_conf", "N_pos_high_conf"]
            ].to_string(index=False)
        )

    # Attention audit on validation set.
    print("\nExporting validation attention audit ...")
    export_attention(
        torch,
        model,
        val_records,
        val_ds,
        device,
        run_dir / "val_attention_top_slices.csv",
        top_slices=args.top_attention_slices,
    )

    print(f"\nTraining complete. Results: {run_dir}")
    print(
        "Important: attention is an audit/localization clue, not a slice-level ground-truth label."
    )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def add_common_args(p):
    p.add_argument("--census", required=True,
                   help="SQLite checkpoint from knee_mri_census_v2.py")
    p.add_argument("--labels", required=True,
                   help="Weak-label Excel/CSV with StudyInstanceUID and 12 targets")
    p.add_argument("--labels-sheet", default="WeakLabels",
                   help="Excel sheet containing labels (default: WeakLabels)")
    p.add_argument("--run-dir", required=True,
                   help="Output directory")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument(
        "--split-mode",
        choices=["random", "source", "scanner"],
        default="random",
        help=(
            "Study-level random split, whole-source holdout, or normalized "
            "scanner-domain holdout (vendor + model + field strength)"
        ),
    )


def build_parser():
    ap = argparse.ArgumentParser(
        description="12-target hierarchical MIL for weakly labeled knee MRI studies"
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_audit = sub.add_parser(
        "audit",
        help="Join imaging hierarchy to all 12 labels and write manifests; no training"
    )
    add_common_args(p_audit)

    p_train = sub.add_parser(
        "train",
        help="Train the 12-target hierarchical MIL model"
    )
    add_common_args(p_train)

    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--max-series", type=int, default=8,
                         help="Maximum series sampled per study; <=0 means all")
    p_train.add_argument("--slices-per-series", type=int, default=12,
                         help="Slices sampled per selected series; <=0 means all")
    p_train.add_argument("--image-size", type=int, default=224)
    p_train.add_argument("--embedding-dim", type=int, default=512)
    p_train.add_argument("--attention-dim", type=int, default=128)
    p_train.add_argument("--encoder-batch", type=int, default=32,
                         help="Number of slice images encoded at once on GPU")
    p_train.add_argument("--freeze-encoder", action="store_true",
                         help="Freeze ResNet encoder and train attention/classifiers only")
    p_train.add_argument("--no-pretrained", action="store_true",
                         help="Do not use ImageNet-pretrained ResNet18")
    p_train.add_argument("--lr", type=float, default=2e-4)
    p_train.add_argument("--weight-decay", type=float, default=1e-4)
    p_train.add_argument("--grad-clip", type=float, default=5.0)
    p_train.add_argument("--accum-steps", type=int, default=4,
                         help="Accumulate gradients across N studies")
    p_train.add_argument("--num-workers", type=int, default=0,
                         help="Start at 0 under WSL; raise cautiously")
    p_train.add_argument("--precision", choices=["fp32", "fp16", "bf16"],
                         default="bf16")
    p_train.add_argument("--log-every", type=int, default=25)
    p_train.add_argument("--high-conf-threshold", type=float, default=0.8,
                         help="For AUROC: use p<=1-t or p>=t")
    p_train.add_argument("--top-attention-slices", type=int, default=3)

    return ap


def main():
    args = build_parser().parse_args()

    if args.command == "audit":
        command_audit(args)
    elif args.command == "train":
        command_train(args)
    else:
        raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()
