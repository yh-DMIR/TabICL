#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
8-GPU SERIAL (per-dataset) runner with manual data-parallel inference for TabICL.

Behavior:
- Datasets are processed STRICTLY serially (one dataset at a time).
- For EACH dataset, we spawn N processes (N = --workers, usually 8),
  each binds to one GPU and runs:
    fit(X_train, y_train)  [full train]
    predict_proba(X_test_shard)  [only its shard]
  then the parent concatenates shards to get full proba/pred and computes metrics.

Keeps:
- dataset discovery, missing test reporting
- ResultRow schema and ALL.csv + summary.txt writing format
- TabICL kwargs parsing and printing

Note:
- This is NOT the dynamic queue runner. It's designed for the "few huge datasets dominate" case.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

TARGET_CANDIDATES = [
    "target", "label", "class", "y",
    "TARGET", "Label", "Class", "Y",
]


def sanitize_dataset_id(train_path: Path) -> str:
    m = re.search(r"(OpenML-ID-\d+)", str(train_path))
    return m.group(1) if m else train_path.parent.name


def find_dataset_pairs(root: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for train_path in root.rglob("*_train.csv"):
        test_path = train_path.with_name(train_path.name.replace("_train.csv", "_test.csv"))
        if test_path.exists():
            pairs.append((train_path, test_path))
    return sorted(pairs, key=lambda x: str(x[0]))


def find_missing_test_datasets(root: Path) -> List[str]:
    missing: List[str] = []
    for train_path in root.rglob("*_train.csv"):
        test_path = train_path.with_name(train_path.name.replace("_train.csv", "_test.csv"))
        if not test_path.exists():
            missing.append(sanitize_dataset_id(train_path))
    return sorted(set(missing))


def infer_target_column(train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    for c in TARGET_CANDIDATES:
        if c in train_df.columns:
            return c
    extra = [c for c in train_df.columns if c not in test_df.columns]
    if len(extra) == 1:
        return extra[0]
    return train_df.columns[-1]


def _normalize_local_ckpt_path(model_path: Optional[str]) -> Optional[str]:
    if not model_path:
        return None
    mp = Path(model_path).expanduser()
    try:
        mp = mp.resolve()
    except Exception:
        pass
    if not mp.exists():
        raise FileNotFoundError(f"Local checkpoint not found: {mp}")
    return str(mp)


def _default_all_out(out_dir: Path) -> Path:
    return out_dir / "tabicl_results.ALL.csv"


def _default_summary_txt(out_dir: Path) -> Path:
    return out_dir / "tabicl_results.summary.txt"


def _fmt_hms(seconds: float) -> str:
    if seconds is None:
        return ""
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h}:{m:02d}:{s:02d}"


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


def write_summary_txt(
    out_txt: Path,
    root: Path,
    discovered_pairs: int,
    processed_pairs: int,
    missing_test_ids: List[str],
    failed_ids: List[str],
    avg_acc: Optional[float],
    topn_avgs: Dict[int, float],
    wall_seconds: Optional[float] = None,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
):
    lines: List[str] = []
    lines.append(f"root: {root}")
    lines.append(f"discovered_pairs: {discovered_pairs}")
    lines.append(f"processed_pairs: {processed_pairs}")

    if started_at is not None:
        lines.append(f"started_at: {started_at}")
    if finished_at is not None:
        lines.append(f"finished_at: {finished_at}")
    if wall_seconds is not None:
        lines.append(f"wall_seconds: {wall_seconds:.3f}")
        lines.append(f"wall_time_hms: {_fmt_hms(wall_seconds)}")

    lines.append(f"missing_test_count: {len(missing_test_ids)}")
    if missing_test_ids:
        lines.append("missing_test_datasets: " + ", ".join(missing_test_ids))
    else:
        lines.append("missing_test_datasets: (none)")

    lines.append(f"failed_count: {len(failed_ids)}")
    if failed_ids:
        lines.append("failed_datasets: " + ", ".join(failed_ids))
    else:
        lines.append("failed_datasets: (none)")

    if avg_acc is None:
        lines.append("avg_accuracy_ok: (none)")
    else:
        lines.append(f"avg_accuracy_ok: {avg_acc:.6f}")

    for n in (27, 63, 154):
        if n in topn_avgs:
            lines.append(f"avg_accuracy_ok_top_{n}: {topn_avgs[n]:.6f}")

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------
# 8-GPU data-parallel per-dataset
# -----------------------------

def _split_indices(n: int, parts: int) -> List[Tuple[int, int]]:
    """Return list of (start, end) half-open slices covering [0, n)."""
    if parts <= 0:
        raise ValueError("parts must be > 0")
    base = n // parts
    rem = n % parts
    out = []
    s = 0
    for i in range(parts):
        e = s + base + (1 if i < rem else 0)
        out.append((s, e))
        s = e
    return out


def _dp_worker_predict_proba(
    rank: int,
    gpu_id: int,
    train_csv: str,
    test_csv: str,
    x_test_slice: Tuple[int, int],
    clf_kwargs: Dict[str, Any],
    out_queue,
):
    """
    Worker does:
      - bind to GPU
      - init TabICLClassifier
      - load train/test
      - fit on full train
      - predict_proba on X_test[s:e]
      - put (rank, s, e, proba, classes_, fit_s, pred_s) into out_queue
    """
    try:
        os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        from tabicl import TabICLClassifier  # noqa

        local_kwargs = dict(clf_kwargs)
        # In each worker, "cuda:0" refers to the one visible GPU
        local_kwargs["device"] = "cuda:0"

        clf = TabICLClassifier(**local_kwargs)

        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)

        target_col = infer_target_column(train_df, test_df)

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]

        if target_col in test_df.columns:
            X_test_full = test_df.drop(columns=[target_col])
        else:
            X_test_full = test_df

        s, e = x_test_slice
        X_test = X_test_full.iloc[s:e]

        t0 = time.time()
        clf.fit(X_train, y_train)
        fit_s = time.time() - t0

        t1 = time.time()
        proba = clf.predict_proba(X_test)  # ONLY ONE CALL
        pred_s = time.time() - t1

        classes_ = getattr(clf, "classes_", None)
        # Convert to numpy explicitly for IPC
        proba_np = np.asarray(proba)

        out_queue.put({
            "ok": True,
            "rank": rank,
            "s": s,
            "e": e,
            "proba": proba_np,
            "classes": np.asarray(classes_) if classes_ is not None else None,
            "fit_s": float(fit_s),
            "pred_s": float(pred_s),
            "err": None,
        })
    except Exception:
        out_queue.put({
            "ok": False,
            "rank": rank,
            "s": x_test_slice[0],
            "e": x_test_slice[1],
            "proba": None,
            "classes": None,
            "fit_s": 0.0,
            "pred_s": 0.0,
            "err": traceback.format_exc(),
        })


def run_one_dataset_8gpu_serial(
    train_csv: Path,
    test_csv: Path,
    gpu_ids: List[int],
    clf_kwargs: Dict[str, Any],
    verbose: bool,
) -> ResultRow:
    dataset_id = sanitize_dataset_id(train_csv)

    try:
        # Load once in parent for metrics/shape/target inference and y_test
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        target_col = infer_target_column(train_df, test_df)

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]

        if target_col in test_df.columns:
            X_test_full = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]
        else:
            X_test_full = test_df
            y_test = None

        n_test = len(X_test_full)
        n_parts = len(gpu_ids)
        slices = _split_indices(n_test, n_parts)

        import multiprocessing as mp

        # Queue for results
        out_q: mp.Queue = mp.Queue()

        # Spawn N workers
        procs: List[mp.Process] = []
        for rank, (gpu_id, slc) in enumerate(zip(gpu_ids, slices)):
            p = mp.Process(
                target=_dp_worker_predict_proba,
                args=(rank, gpu_id, str(train_csv), str(test_csv), slc, clf_kwargs, out_q),
                daemon=False,
            )
            p.start()
            procs.append(p)

        # Collect N results
        results = []
        for _ in range(n_parts):
            results.append(out_q.get())

        # Join
        for p in procs:
            p.join()

        # If any worker failed -> mark dataset fail (keep behavior consistent: one row per dataset)
        fails = [r for r in results if not r["ok"]]
        if fails:
            # Take first error for reporting
            err = fails[0]["err"]
            if verbose:
                print(f"[DP-8GPU] FAIL {dataset_id}\n{err}")
            return ResultRow(
                dataset_id=dataset_id,
                n_train=int(len(X_train)),
                n_test=int(len(X_test_full)),
                n_features=int(X_train.shape[1]),
                n_classes=int(y_train.nunique()) if y_test is None else None,
                accuracy=None,
                f1_weighted=None,
                logloss=None,
                fit_seconds=0.0,
                predict_seconds=0.0,
                status="fail",
                error=err,
            )

        # Sort shards by slice start
        results.sort(key=lambda r: r["s"])

        # Validate classes alignment (should be identical)
        classes0 = results[0]["classes"]
        for r in results[1:]:
            if r["classes"] is None or classes0 is None:
                continue
            if r["classes"].shape != classes0.shape or not np.all(r["classes"] == classes0):
                raise RuntimeError(
                    f"classes_ mismatch across ranks for {dataset_id}. "
                    f"Need consistent class ordering to stitch proba safely."
                )

        # Stitch proba
        proba_full = np.concatenate([r["proba"] for r in results], axis=0)

        # Derive y_pred from proba (same as your speed fix)
        y_pred = classes0[np.argmax(proba_full, axis=1)] if classes0 is not None else np.argmax(proba_full, axis=1)

        # Aggregate timings: since workers run in parallel, wall-ish per-stage is max
        fit_s = float(max(r["fit_s"] for r in results))
        pred_s = float(max(r["pred_s"] for r in results))

        if y_test is not None:
            acc = accuracy_score(y_test, y_pred)
            f1w = f1_score(y_test, y_pred, average="weighted")
            try:
                ll = log_loss(y_test, proba_full, labels=classes0)
            except Exception:
                ll = None
            n_classes = int(len(classes0)) if classes0 is not None else int(y_test.nunique())
        else:
            acc = f1w = ll = None
            n_classes = int(y_train.nunique())

        if verbose:
            print(f"[DP-8GPU] OK {dataset_id} acc={acc} fit~{fit_s:.1f}s pred~{pred_s:.1f}s")

        return ResultRow(
            dataset_id=dataset_id,
            n_train=int(len(X_train)),
            n_test=int(len(X_test_full)),
            n_features=int(X_train.shape[1]),
            n_classes=n_classes,
            accuracy=float(acc) if acc is not None else None,
            f1_weighted=float(f1w) if f1w is not None else None,
            logloss=float(ll) if ll is not None else None,
            fit_seconds=float(fit_s),
            predict_seconds=float(pred_s),
            status="ok",
            error=None,
        )

    except Exception as e:
        return ResultRow(
            dataset_id=dataset_id,
            n_train=0,
            n_test=0,
            n_features=0,
            n_classes=None,
            accuracy=None,
            f1_weighted=None,
            logloss=None,
            fit_seconds=0.0,
            predict_seconds=0.0,
            status="fail",
            error=f"{type(e).__name__}: {e}",
        )


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", required=True, help="Root folder containing *_train.csv and *_test.csv")
    ap.add_argument("--out-dir", required=True, help="Output directory for outputs")
    ap.add_argument("--all-out", default=None, help="Path to merged ALL CSV (default: <out-dir>/tabicl_results.ALL.csv)")
    ap.add_argument("--summary-txt", default=None, help="Path to ONE global summary txt (default: <out-dir>/tabicl_results.summary.txt)")
    ap.add_argument("--workers", type=int, default=8, help="Number of GPUs to use for data-parallel (usually 8)")
    ap.add_argument("--gpus", default=None, help="Comma-separated GPU ids to use (default: 0..workers-1)")

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
    ap.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to local TabICL checkpoint (.ckpt). If provided, TabICL loads from disk and will NOT auto-download.",
    )

    # TabICL params
    ap.add_argument("--device", default="cuda:0", help='Device string in workers (ignored; workers use cuda:0)')
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

    # Whole-run timing
    run_start_ts = time.time()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run_start_ts))

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_out = Path(args.all_out) if args.all_out else _default_all_out(out_dir)
    summary_txt = Path(args.summary_txt) if args.summary_txt else _default_summary_txt(out_dir)

    root = Path(args.root)
    missing_test_ids = find_missing_test_datasets(root)
    pairs = find_dataset_pairs(root)
    discovered_pairs = len(pairs)

    if discovered_pairs == 0:
        empty_df = pd.DataFrame(columns=[f.name for f in ResultRow.__dataclass_fields__.values()])
        all_out.parent.mkdir(parents=True, exist_ok=True)
        empty_df.to_csv(all_out, index=False)

        run_end_ts = time.time()
        finished_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run_end_ts))
        wall_seconds = run_end_ts - run_start_ts

        write_summary_txt(
            out_txt=summary_txt,
            root=root,
            discovered_pairs=0,
            processed_pairs=0,
            missing_test_ids=missing_test_ids,
            failed_ids=[],
            avg_acc=None,
            topn_avgs={},
            wall_seconds=wall_seconds,
            started_at=started_at,
            finished_at=finished_at,
        )
        print("No dataset pairs found. Wrote empty outputs.")
        return

    workers = int(args.workers)
    if workers < 1:
        raise ValueError("--workers must be >= 1")

    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip() != ""]
        if len(gpu_ids) != workers:
            raise ValueError(f"--gpus must list exactly {workers} ids, got {len(gpu_ids)}")
    else:
        gpu_ids = list(range(workers))

    norm_methods = [x.strip() for x in args.norm_methods.split(",") if x.strip()]
    model_path = _normalize_local_ckpt_path(args.model_path)

    clf_kwargs: Dict[str, Any] = dict(
        n_estimators=args.n_estimators,
        norm_methods=norm_methods,
        feat_shuffle_method=args.feat_shuffle,
        class_shift=not args.no_class_shift,
        softmax_temperature=args.softmax_temp,
        average_logits=not args.no_average_logits,
        use_amp=not args.no_amp,
        batch_size=args.batch_size,
        # device is forced to cuda:0 inside each worker after HIP_VISIBLE_DEVICES binding
        random_state=args.random_state,
        verbose=False,
        checkpoint_version=args.checkpoint_version,
    )

    if model_path is not None:
        clf_kwargs["model_path"] = model_path
        clf_kwargs["allow_auto_download"] = False

    # Multiprocessing start method
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    rows: List[ResultRow] = []

    # SERIAL over datasets
    for idx, (train_csv, test_csv) in enumerate(pairs, start=1):
        if args.verbose:
            print(f"\n=== [{idx}/{discovered_pairs}] DP-8GPU SERIAL: {sanitize_dataset_id(train_csv)} ===")
        row = run_one_dataset_8gpu_serial(
            train_csv=train_csv,
            test_csv=test_csv,
            gpu_ids=gpu_ids,
            clf_kwargs=clf_kwargs,
            verbose=args.verbose,
        )
        rows.append(row)

    all_df = pd.DataFrame([asdict(r) for r in rows])
    all_out.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(all_out, index=False)

    processed_pairs = int(len(all_df))

    if len(all_df):
        ok_df = all_df[(all_df["status"] == "ok") & all_df["accuracy"].notna()].copy()
    else:
        ok_df = pd.DataFrame(columns=["accuracy", "status", "dataset_id"])

    avg_acc = float(ok_df["accuracy"].mean()) if len(ok_df) > 0 else None

    topn_avgs: Dict[int, float] = {}
    if len(ok_df) > 0:
        ok_sorted = ok_df.sort_values("accuracy", ascending=False, kind="mergesort")
        ok_count = len(ok_sorted)
        for n in (27, 63, 154):
            if ok_count >= n:
                topn_avgs[n] = float(ok_sorted.head(n)["accuracy"].mean())

    failed_ids: List[str] = []
    if len(all_df):
        failed_ids = (
            all_df.loc[all_df["status"] == "fail", "dataset_id"]
            .dropna()
            .astype(str)
            .tolist()
        )
        failed_ids = sorted(set(failed_ids))

    run_end_ts = time.time()
    finished_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run_end_ts))
    wall_seconds = run_end_ts - run_start_ts

    write_summary_txt(
        out_txt=summary_txt,
        root=root,
        discovered_pairs=discovered_pairs,
        processed_pairs=processed_pairs,
        missing_test_ids=missing_test_ids,
        failed_ids=failed_ids,
        avg_acc=avg_acc,
        topn_avgs=topn_avgs,
        wall_seconds=wall_seconds,
        started_at=started_at,
        finished_at=finished_at,
    )

    print("\nSaved merged ALL CSV to:", str(all_out))
    print("Saved summary TXT to:", str(summary_txt))
    print(f"\nTotal wall time: {wall_seconds:.3f}s ({_fmt_hms(wall_seconds)})")
    print("\nTabICL kwargs:")
    print(json.dumps(clf_kwargs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
