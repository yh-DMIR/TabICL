#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch benchmark script for TabICL
Compatible with directory structure:
root/
  tabzilla_csv/
    OpenML-ID-10/
      OpenML-ID-10_train.csv
      OpenML-ID-10_test.csv
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# 🔴 核心变化：使用 TabICL
from tabicl import TabICLClassifier


TARGET_CANDIDATES = [
    "target", "label", "class", "y",
    "TARGET", "Label", "Class", "Y",
]


def find_dataset_pairs(root: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for train_path in root.rglob("*_train.csv"):
        test_path = train_path.with_name(train_path.name.replace("_train.csv", "_test.csv"))
        if test_path.exists():
            pairs.append((train_path, test_path))
    return sorted(pairs, key=lambda x: str(x[0]))


def infer_target_column(train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    for c in TARGET_CANDIDATES:
        if c in train_df.columns:
            return c

    extra = [c for c in train_df.columns if c not in test_df.columns]
    if len(extra) == 1:
        return extra[0]

    return train_df.columns[-1]


def sanitize_dataset_id(train_path: Path) -> str:
    m = re.search(r"(OpenML-ID-\d+)", str(train_path))
    return m.group(1) if m else train_path.parent.name


@dataclass
class ResultRow:
    dataset_id: str
    n_train: int
    n_test: int
    n_features: int
    n_classes: Optional[int]
    accuracy: Optional[float]
    f1_weighted: Optional[float]
    logloss: Optional[float]
    fit_seconds: float
    predict_seconds: float
    status: str
    error: Optional[str]


def _normalize_local_ckpt_path(model_path: Optional[str]) -> Optional[str]:
    """Resolve model_path to an absolute path and validate existence."""
    if not model_path:
        return None
    mp = Path(model_path).expanduser()
    try:
        mp = mp.resolve()
    except Exception:
        # resolve may fail on some weird FS; keep as-is
        pass
    if not mp.exists():
        raise FileNotFoundError(f"Local checkpoint not found: {mp}")
    return str(mp)


def run_one_dataset(
    train_csv: Path,
    test_csv: Path,
    clf_kwargs: Dict,
) -> ResultRow:
    dataset_id = sanitize_dataset_id(train_csv)

    try:
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)

        target_col = infer_target_column(train_df, test_df)

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]

        if target_col in test_df.columns:
            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]
        else:
            X_test = test_df
            y_test = None

        clf = TabICLClassifier(**clf_kwargs)

        t0 = time.time()
        clf.fit(X_train, y_train)
        fit_s = time.time() - t0

        t1 = time.time()
        y_pred = clf.predict(X_test)
        pred_s = time.time() - t1

        if y_test is not None:
            acc = accuracy_score(y_test, y_pred)
            f1w = f1_score(y_test, y_pred, average="weighted")

            try:
                proba = clf.predict_proba(X_test)
                ll = log_loss(y_test, proba, labels=clf.classes_)
            except Exception:
                ll = None

            n_classes = clf.n_classes_
        else:
            acc = f1w = ll = None
            n_classes = y_train.nunique()

        return ResultRow(
            dataset_id,
            len(X_train),
            len(X_test),
            X_train.shape[1],
            n_classes,
            acc,
            f1w,
            ll,
            fit_s,
            pred_s,
            "ok",
            None,
        )

    except Exception as e:
        return ResultRow(
            dataset_id, 0, 0, 0, None,
            None, None, None, 0.0, 0.0,
            "fail", f"{type(e).__name__}: {e}",
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="tabicl_results.csv")

    # 🔥 TabICL 核心参数
    ap.add_argument(
        "--checkpoint-version",
        type=str,
        default="tabicl-classifier-v1.1-0506.ckpt",
        choices=[
            "tabicl-classifier-v1.1-0506.ckpt",
            "tabicl-classifier-v1-0208.ckpt",
        ],
        help="Choose TabICL pretrained checkpoint",
    )

    # ✅ 新增：本地 ckpt 路径（离线/无外网机器必备）
    ap.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to local TabICL checkpoint (.ckpt). If provided, TabICL will load from disk and will NOT auto-download.",
    )

    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-estimators", type=int, default=32)
    ap.add_argument("--norm-methods", default="none,power")
    ap.add_argument("--feat-shuffle", default="latin")
    ap.add_argument("--no-class-shift", action="store_true")
    ap.add_argument("--softmax-temp", type=float, default=0.9)
    ap.add_argument("--no-average-logits", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    root = Path(args.root)
    pairs = find_dataset_pairs(root)

    norm_methods = [x.strip() for x in args.norm_methods.split(",") if x.strip()]

    model_path = _normalize_local_ckpt_path(args.model_path)

    clf_kwargs = dict(
        n_estimators=args.n_estimators,
        norm_methods=norm_methods,
        feat_shuffle_method=args.feat_shuffle,
        class_shift=not args.no_class_shift,
        softmax_temperature=args.softmax_temp,
        average_logits=not args.no_average_logits,
        use_amp=not args.no_amp,
        batch_size=args.batch_size,
        device=args.device,
        random_state=args.random_state,
        verbose=args.verbose,
        checkpoint_version=args.checkpoint_version,
    )

    # ✅ 如果提供本地 ckpt，则传入并禁用自动下载（避免任何联网尝试）
    if model_path is not None:
        clf_kwargs["model_path"] = model_path
        clf_kwargs["allow_auto_download"] = False

    results = []
    for train_csv, test_csv in pairs:
        row = run_one_dataset(train_csv, test_csv, clf_kwargs)
        results.append(row)
        print(f"[{row.status}] {row.dataset_id} acc={row.accuracy}")

    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(args.out, index=False)

    print("\nSaved to:", args.out)
    print(json.dumps(clf_kwargs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
