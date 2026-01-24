#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dynamic multi-GPU benchmark runner for TabICL (AMD/ROCm friendly).

Key features:
- Dynamic scheduling (work stealing): N worker processes pull datasets from a shared queue.
- Each worker binds to one GPU via HIP_VISIBLE_DEVICES=<gpu_id> and uses device "cuda:0".
- Writes per-worker CSVs, merges into one ALL CSV, and writes ONE global summary TXT:
  - discovered_pairs: number of valid (train,test) pairs found
  - processed_pairs: number of attempted datasets (ok + fail rows written)
  - missing_test_datasets: train exists but test missing
  - failed_datasets: datasets with status=fail (including worker crash marker if any)
  - avg_accuracy_ok: mean accuracy over all OK results
  - avg_accuracy_ok_top_{27,63,154}: mean accuracy of top-N datasets by accuracy (descending),
    computed only if #OK-with-accuracy >= N.
  - wall_seconds: total elapsed wall time of the whole run (from start to end)
  - wall_time_hms: wall time formatted as H:MM:SS
  - started_at / finished_at: timestamps (local time)

Example:
  python benchmark_tabicl_dynamic.py \
    --root limix/tabzilla_csv \
    --out-dir results/v1.1/tabzilla_dynamic \
    --workers 8 \
    --model-path ./tabicl-classifier-v1.1-0506.ckpt \
    --device cuda:0
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
from typing import Optional, Tuple, List, Dict

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# -----------------------------
# Helpers: dataset discovery
# -----------------------------

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
    """Format seconds -> H:MM:SS (rounded to nearest second)."""
    if seconds is None:
        return ""
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h}:{m:02d}:{s:02d}"


# -----------------------------
# Result schema
# -----------------------------

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


# -----------------------------
# Core evaluation (reuses one clf per worker)
# -----------------------------

def run_one_dataset_with_clf(
    clf,
    train_csv: Path,
    test_csv: Path,
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

            n_classes = getattr(clf, "n_classes_", None)
        else:
            acc = f1w = ll = None
            n_classes = int(y_train.nunique())

        return ResultRow(
            dataset_id=dataset_id,
            n_train=int(len(X_train)),
            n_test=int(len(X_test)),
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


# -----------------------------
# Worker process (dynamic queue)
# -----------------------------

def worker_main(
    worker_id: int,
    gpu_id: int,
    task_queue,
    out_csv: str,
    clf_kwargs: Dict,
    verbose: bool,
):
    """
    Each worker:
    - bind to one GPU
    - create TabICLClassifier once
    - pull tasks until sentinel None
    - write its own CSV at the end
    """
    try:
        # Bind GPU (AMD/ROCm)
        os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)

        # Reduce CPU contention
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        # Import inside worker after setting HIP_VISIBLE_DEVICES
        from tabicl import TabICLClassifier  # noqa

        # In each worker, "cuda:0" refers to the one visible GPU
        if not clf_kwargs.get("device"):
            clf_kwargs["device"] = "cuda:0"

        clf = TabICLClassifier(**clf_kwargs)

        rows: List[ResultRow] = []
        while True:
            item = task_queue.get()
            if item is None:
                break

            train_csv, test_csv = item
            row = run_one_dataset_with_clf(clf, Path(train_csv), Path(test_csv))
            rows.append(row)

            if verbose:
                print(f"[worker {worker_id} | gpu {gpu_id}] [{row.status}] {row.dataset_id} acc={row.accuracy}")

        df = pd.DataFrame([asdict(r) for r in rows])
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    except Exception:
        # If a worker crashes, write a marker row for debugging
        err = traceback.format_exc()
        crash_df = pd.DataFrame([{
            "dataset_id": f"__WORKER_CRASH__{worker_id}",
            "n_train": 0,
            "n_test": 0,
            "n_features": 0,
            "n_classes": None,
            "accuracy": None,
            "f1_weighted": None,
            "logloss": None,
            "fit_seconds": 0.0,
            "predict_seconds": 0.0,
            "status": "fail",
            "error": err,
        }])
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        crash_df.to_csv(out_csv, index=False)
        if verbose:
            print(f"[worker {worker_id}] CRASHED:\n{err}")


# -----------------------------
# Summary writer
# -----------------------------

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

    # ---- timing ----
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

    # Top-N by highest accuracy (only if enough OK results exist)
    for n in (27, 63, 154):
        if n in topn_avgs:
            lines.append(f"avg_accuracy_ok_top_{n}: {topn_avgs[n]:.6f}")

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", required=True, help="Root folder containing *_train.csv and *_test.csv")
    ap.add_argument("--out-dir", required=True, help="Output directory for per-worker CSVs and merged results")
    ap.add_argument("--all-out", default=None, help="Path to merged ALL CSV (default: <out-dir>/tabicl_results.ALL.csv)")
    ap.add_argument("--summary-txt", default=None, help="Path to ONE global summary txt (default: <out-dir>/tabicl_results.summary.txt)")
    ap.add_argument("--workers", type=int, default=8, help="Number of worker processes (usually #GPUs)")
    ap.add_argument("--gpus", default=None, help="Comma-separated GPU ids to use (default: 0..workers-1)")

    # TabICL checkpoint
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
    ap.add_argument("--device", default="cuda:0", help='Device string in workers (recommend: "cuda:0")')
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

    # ---- whole-run timing start (covers discovery -> workers -> merge -> summary) ----
    run_start_ts = time.time()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run_start_ts))

    # Reduce CPU contention (good default for multi-proc)
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
        # still write empty outputs + summary
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

    clf_kwargs: Dict = dict(
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
        verbose=False,  # worker logs controlled by args.verbose
        checkpoint_version=args.checkpoint_version,
    )

    if model_path is not None:
        clf_kwargs["model_path"] = model_path
        clf_kwargs["allow_auto_download"] = False

    # --- multiprocessing setup ---
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    task_queue: mp.Queue = mp.Queue()  # unbounded, won't block before workers start

    # Enqueue all tasks
    for train_csv, test_csv in pairs:
        task_queue.put((str(train_csv), str(test_csv)))

    # Sentinels
    for _ in range(workers):
        task_queue.put(None)

    # Start workers
    procs: List[mp.Process] = []
    worker_csv_paths: List[Path] = []
    for wid in range(workers):
        w_csv = out_dir / f"worker_{wid}.csv"
        worker_csv_paths.append(w_csv)
        p = mp.Process(
            target=worker_main,
            args=(wid, gpu_ids[wid], task_queue, str(w_csv), dict(clf_kwargs), args.verbose),
            daemon=False,
        )
        p.start()
        procs.append(p)

    # Wait
    for p in procs:
        p.join()

    # Merge per-worker CSVs
    dfs: List[pd.DataFrame] = []
    for w_csv in worker_csv_paths:
        if w_csv.exists():
            try:
                dfs.append(pd.read_csv(w_csv))
            except Exception:
                continue

    if dfs:
        all_df = pd.concat(dfs, ignore_index=True)
    else:
        all_df = pd.DataFrame(columns=[f.name for f in ResultRow.__dataclass_fields__.values()])

    all_out.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(all_out, index=False)

    # Compute summary
    processed_pairs = int(len(all_df))  # attempted rows (ok + fail)

    if len(all_df):
        ok_df = all_df[(all_df["status"] == "ok") & all_df["accuracy"].notna()].copy()
    else:
        ok_df = pd.DataFrame(columns=["accuracy", "status", "dataset_id"])

    avg_acc = float(ok_df["accuracy"].mean()) if len(ok_df) > 0 else None

    # Top-N averages by highest accuracy
    cutoffs = (27, 63, 154)
    topn_avgs: Dict[int, float] = {}
    if len(ok_df) > 0:
        ok_sorted = ok_df.sort_values("accuracy", ascending=False, kind="mergesort")
        ok_count = len(ok_sorted)
        for n in cutoffs:
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

    # ---- whole-run timing end ----
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

    # Print config for reproducibility
    print("\nSaved per-worker CSVs to:", str(out_dir))
    print("Saved merged ALL CSV to:", str(all_out))
    print("Saved summary TXT to:", str(summary_txt))
    print(f"\nTotal wall time: {wall_seconds:.3f}s ({_fmt_hms(wall_seconds)})")
    print("\nTabICL kwargs:")
    print(json.dumps(clf_kwargs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
