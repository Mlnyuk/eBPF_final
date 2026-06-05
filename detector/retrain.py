#!/usr/bin/env python3
"""
retrain.py
==========
Drift-adaptation retraining with a regression gate.

Pipeline (run by k8s/cronjob-retrain.yaml):
  1. Train a CANDIDATE Isolation Forest on recently archived production features
     (the detector's /archive corpus — see detector/archive.py).
  2. Evaluate the candidate AND the current production model on a fixed labelled
     holdout (results/labeled14/: normal_sample.csv + fault_*.csv), computing
     ROC AUC and per-fault hybrid recall.
  3. GATE: promote the candidate only if it does not regress —
       candidate ROC AUC  >= current ROC AUC - --auc-tol   AND
       min per-fault recall >= --recall-floor
     Otherwise keep the current model.
  4. Write the candidate bundle to --candidate-out and print a JSON verdict.
     Exit 0 = PROMOTE, 3 = REJECT (kept current), other = error.

Training data has a header (archive CSV); the holdout files are headerless
(timestamp,node,namespace,pod,container,<14 features>,label) — matching
threshold_tune.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from detector.model_utils import (  # noqa: E402
    ModelBundle, load_config, get_feature_order, frame_to_matrix,
    raw_anomaly_score,
)
from detector import detect as detect_mod  # noqa: E402

META_COLS = ["timestamp", "node", "namespace", "pod", "container"]

# Streaming/sampling bounds. The retrain corpus grows with archive history, but
# the model only needs a representative sample (IsolationForest subsamples to
# max_samples per tree anyway). We stream each file in CHUNK_ROWS-row blocks and
# keep a uniform reservoir of at most MAX_TRAIN_ROWS rows, so the job's peak RAM
# is bounded by the reservoir — not by how many days of archive have piled up.
CHUNK_ROWS = 100_000
MAX_TRAIN_ROWS = 300_000


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def _iter_chunks(path: Path, chunk_rows: int):
    """Yield DataFrame chunks from a CSV/JSONL archive file, never loading the
    whole file into memory."""
    if path.suffix == ".jsonl":
        reader = pd.read_json(path, lines=True, chunksize=chunk_rows)
    else:
        reader = pd.read_csv(path, chunksize=chunk_rows)  # archive CSV has a header
    for chunk in reader:
        yield chunk


def _reservoir_sample(data_paths: List[str], feature_order: List[str], cap: int,
                      normal_only: bool, seed: int) -> Tuple[np.ndarray, int, int]:
    """Uniform random sample of up to `cap` feature rows streamed across the whole
    corpus (Algorithm A-Res: draw a random key per row, keep the `cap` smallest
    keys). Every row across all files has equal inclusion probability, independent
    of corpus size, and peak memory stays ~`cap` + one chunk.

    Returns (X, n_seen, n_kept) where n_seen counts every streamed row and n_kept
    counts rows surviving the normal-only filter (the reservoir's population).
    """
    rng = np.random.default_rng(seed)
    n_feat = len(feature_order)
    keys = np.empty(0, dtype=np.float64)
    res = np.empty((0, n_feat), dtype=np.float64)
    n_seen = n_kept = 0
    for p in data_paths:
        path = Path(p)
        if not path.exists():
            continue
        for chunk in _iter_chunks(path, CHUNK_ROWS):
            n_seen += len(chunk)
            if normal_only:
                chunk = chunk[_normal_mask(chunk)]
            if chunk.empty:
                continue
            n_kept += len(chunk)
            Xc = frame_to_matrix(chunk, feature_order)
            kc = rng.random(len(Xc))
            keys = np.concatenate([keys, kc])
            res = Xc if res.size == 0 else np.vstack([res, Xc])
            if len(keys) > cap:
                keep = np.argpartition(keys, cap)[:cap]
                keys, res = keys[keep], res[keep]
    return res, n_seen, n_kept


def _normal_mask(df: pd.DataFrame) -> pd.Series:
    """Rows the current model considers normal (is_anomaly == False). Keeps the
    candidate's baseline stats clean so the z-tail side-rule survives — training
    on the full archive lets fault rows inflate the per-feature std and silently
    disables single-axis detection (e.g. cpu-stress)."""
    if "is_anomaly" not in df.columns:
        return pd.Series(True, index=df.index)
    flag = df["is_anomaly"].astype(str).str.lower().isin(["false", "0", "0.0"])
    return flag


def train_candidate(data_paths: List[str], feature_order: List[str],
                    cfg: dict, time_str: str, normal_only: bool = True,
                    max_rows: int = MAX_TRAIN_ROWS, seed: int = 42) -> ModelBundle:
    mcfg = cfg.get("model", {})
    dcfg = cfg.get("detector", {})
    X, n_seen, n_kept = _reservoir_sample(
        data_paths, feature_order, max_rows, normal_only, seed)
    if n_seen == 0:
        raise SystemExit("retrain: no training data found")
    filt = f"normal-only {n_kept}/{n_seen} kept; " if normal_only else f"{n_seen} streamed; "
    print(f"[retrain] {filt}reservoir sampled {len(X)} rows (cap {max_rows})")
    if len(X) < 500:
        raise SystemExit(f"retrain: too few training rows ({len(X)}); aborting")

    contamination = float(mcfg.get("contamination", 0.02))
    model = IsolationForest(
        n_estimators=int(mcfg.get("n_estimators", 200)),
        contamination=contamination,
        max_samples=mcfg.get("max_samples", "auto"),
        random_state=int(mcfg.get("random_state", 42)),
        n_jobs=-1,
    )
    model.fit(X)

    raw = raw_anomaly_score(model, X)
    raw_min, raw_max = float(raw.min()), float(raw.max())
    span = max(raw_max - raw_min, 1e-12)
    norm_train = np.clip((raw - raw_min) / span, 0.0, 1.0)
    threshold = (float(dcfg["score_threshold"]) if dcfg.get("score_threshold") is not None
                 else float(np.quantile(norm_train, 1.0 - contamination)))

    return ModelBundle(
        model=model,
        feature_order=feature_order,
        raw_score_min=raw_min,
        raw_score_max=raw_max,
        contamination=contamination,
        score_threshold=threshold,
        sigma_threshold=float(dcfg.get("sigma_threshold", 10.0)),
        metadata={
            "n_train_samples": int(len(X)),
            "n_rows_seen": int(n_seen),
            "train_means": X.mean(axis=0).tolist(),
            "train_stds": X.std(axis=0).tolist(),
            "sklearn_offset_": float(getattr(model, "offset_", 0.0)),
            "trained_at": time_str,
            "trained_by": "retrain.py",
        },
    )


# --------------------------------------------------------------------------
# Evaluation on the labelled holdout
# --------------------------------------------------------------------------

def _read_labeled(path: Path, feature_order: List[str]) -> pd.DataFrame:
    cols = META_COLS + list(feature_order) + ["label"]
    return pd.read_csv(path, header=None, names=cols)


def evaluate(bundle: ModelBundle, normal_csv: Path, fault_glob: str) -> Dict:
    """ROC AUC (score) + per-fault hybrid recall (score_frame decision)."""
    fo = bundle.feature_order
    neg = _read_labeled(normal_csv, fo)
    fault_paths = [Path(p) for p in glob.glob(fault_glob)]
    pos_frames = [_read_labeled(p, fo) for p in fault_paths if Path(p).exists()]
    if not pos_frames:
        raise SystemExit(f"retrain: no fault files matched {fault_glob}")
    pos = pd.concat(pos_frames, ignore_index=True)

    all_df = pd.concat([neg, pos], ignore_index=True)
    res = detect_mod.score_frame(bundle, all_df)
    score = res["anomaly_score"].to_numpy(dtype=float)
    is_anom = res["is_anomaly"].to_numpy(dtype=bool)
    y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))]).astype(int)

    roc_auc = float(roc_auc_score(y, score)) if len(set(y)) > 1 else float("nan")
    # per-fault hybrid recall
    pos_is_anom = is_anom[len(neg):]
    per_fault = {}
    for ft, grp in pos.reset_index(drop=True).groupby("label"):
        idx = grp.index.to_numpy()
        per_fault[str(ft)] = round(float(pos_is_anom[idx].mean()), 4)
    fpr = float(is_anom[:len(neg)].mean())  # normal-side false-positive rate
    return {
        "roc_auc": round(roc_auc, 4),
        "fault_recall": per_fault,
        "min_fault_recall": round(min(per_fault.values()), 4),
        "normal_fpr": round(fpr, 4),
        "n_normal": int(len(neg)),
        "n_fault": int(len(pos)),
    }


# --------------------------------------------------------------------------

def main() -> None:
    import time
    cfg = load_config()
    feature_order = get_feature_order(cfg)
    cur_default = cfg.get("model", {}).get("path", "models/isolation_forest.pkl")

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True,
                    help="archived feature CSV/JSONL files (training corpus)")
    ap.add_argument("--current-model", default=cur_default,
                    help="current production model bundle (baseline for the gate)")
    ap.add_argument("--holdout-normal", default="results/labeled14/normal_sample.csv")
    ap.add_argument("--holdout-faults", default="results/labeled14/fault_*.csv")
    ap.add_argument("--candidate-out", default="/work/candidate.pkl")
    ap.add_argument("--auc-tol", type=float, default=0.02,
                    help="candidate may be at most this much below current AUC")
    ap.add_argument("--recall-floor", type=float, default=0.80,
                    help="candidate min per-fault recall must be >= this")
    ap.add_argument("--all-rows", action="store_true",
                    help="train on every archived row (default: is_anomaly==False only)")
    ap.add_argument("--max-train-rows", type=int,
                    default=int(os.environ.get("MAX_TRAIN_ROWS", MAX_TRAIN_ROWS)),
                    help="reservoir cap: train on at most this many sampled rows "
                         "(bounds RAM regardless of corpus size)")
    ap.add_argument("--seed", type=int, default=int(os.environ.get("RETRAIN_SEED", 42)),
                    help="reservoir-sampling seed (deterministic candidate)")
    args = ap.parse_args()

    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    verdict: Dict = {"trained_at": ts}

    cand = train_candidate(args.data, feature_order, cfg, ts,
                           normal_only=not args.all_rows,
                           max_rows=args.max_train_rows, seed=args.seed)
    cand_eval = evaluate(cand, Path(args.holdout_normal), args.holdout_faults)
    verdict["candidate"] = cand_eval
    verdict["candidate"]["n_train_samples"] = cand.metadata["n_train_samples"]

    cur_eval = None
    if Path(args.current_model).exists():
        cur = ModelBundle.load(args.current_model)
        cur_eval = evaluate(cur, Path(args.holdout_normal), args.holdout_faults)
    verdict["current"] = cur_eval

    cur_auc = cur_eval["roc_auc"] if cur_eval else 0.0
    auc_ok = cand_eval["roc_auc"] >= cur_auc - args.auc_tol
    recall_ok = cand_eval["min_fault_recall"] >= args.recall_floor
    promote = bool(auc_ok and recall_ok)
    verdict["gate"] = {
        "auc_ok": auc_ok, "recall_ok": recall_ok,
        "auc_tol": args.auc_tol, "recall_floor": args.recall_floor,
        "promote": promote,
    }

    if promote:
        Path(args.candidate_out).parent.mkdir(parents=True, exist_ok=True)
        cand.save(args.candidate_out)
        verdict["candidate_out"] = args.candidate_out

    print(json.dumps(verdict, indent=2))
    sys.exit(0 if promote else 3)


if __name__ == "__main__":
    main()
