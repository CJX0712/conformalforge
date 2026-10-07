"""File loaders for user-supplied datasets (CSV).

Author: 晨星
"""

from __future__ import annotations

import csv
import os

import numpy as np

from ..core.errors import DataError
from ..core.types import Dataset


def load_csv(path: str, label_col: str | int = -1) -> Dataset:
    """Load a CSV into a Dataset. ``label_col`` may be a name or integer index."""
    if not os.path.exists(path):
        raise DataError(f"file not found: {path}")
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise DataError("empty csv")
    header = rows[0]
    body = rows[1:]
    if isinstance(label_col, str):
        if label_col not in header:
            raise DataError(f"label column {label_col!r} not in header")
        li = header.index(label_col)
    else:
        li = label_col % len(header)
    feat_idx = [i for i in range(len(header)) if i != li]
    X = np.array([[float(r[i]) for i in feat_idx] for r in body], dtype=np.float64)
    y_raw = [r[li] for r in body]
    try:
        y = np.array([float(v) for v in y_raw], dtype=np.float64)
    except ValueError:
        classes = sorted(set(y_raw))
        lookup = {c: i for i, c in enumerate(classes)}
        y = np.array([lookup[v] for v in y_raw], dtype=np.int64)
    return Dataset(X, y)


def make_synthetic_then_split(*args, **kwargs) -> tuple[Dataset, Dataset, Dataset]:
    raise DataError("use generators in synthetic.py instead")
