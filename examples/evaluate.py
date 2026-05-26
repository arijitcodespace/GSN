#!/usr/bin/env python
"""DyGLib / DyGMamba-aligned evaluation for a pretrained GSN model.

Thin CLI wrapper. All eval logic lives in :mod:`gsn.train.eval` so the
trainer's per-epoch logged numbers and this script's reported numbers
come from the same code path.

This script intentionally separates two things that are easy to conflate:

  - GSN state buckets: controlled by ``--batch_events``; these affect the
    state trajectory and therefore the scores.
  - DyGLib metric batches: controlled by ``--metric_batch_size``; DyGLib
    computes sklearn AP/AUC inside each evaluation DataLoader batch and
    averages the batch metrics.

Aligned DyGLib / DyGMamba choices:

  - random NSS:
      sample one negative destination from unique full-data destination IDs;
      use the positive source as the negative source in evaluation; do not
      repair collisions with the positive destination.

  - inductive NSS:
      sample negative edges with DyGLib's edge/time sampler:
          historical_edges - observed_edges - current_batch_edges
      The negative source can differ from the positive source.
      This is not unseen-destination-node sampling.

  - metrics:
      sigmoid(logits), concatenate [positives, negatives] inside each metric
      batch, compute sklearn AP/AUC, and average batch metrics.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import tensorflow as tf

from gsn.datasets import load_tgb
from gsn.datasets.tgb_loader import TGBSplit
from gsn.train.eval import (
    DyGLibInductiveNegativeSampler,
    DyGLibRandomNegativeSampler,
    evaluate_split,
)
from gsn.train.loop import GSNLinkPredictor


tf.keras.config.disable_traceback_filtering()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _concat_full(
    train: TGBSplit,
    val:   TGBSplit,
    test:  TGBSplit,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.concatenate([
            train.src.astype(np.int64),
            val.src.astype(np.int64),
            test.src.astype(np.int64),
        ]),
        np.concatenate([
            train.dst.astype(np.int64),
            val.dst.astype(np.int64),
            test.dst.astype(np.int64),
        ]),
        np.concatenate([train.ts, val.ts, test.ts]),
    )


def _assign_actual_temperature(
    model: GSNLinkPredictor,
    tau:   float,
) -> None:
    if tau <= 0:
        raise ValueError("--temp must be positive")
    raw_temp = math.log(math.exp(float(tau)) - 1.0)
    if hasattr(model.temp, "assign"):
        model.temp.assign(np.asarray(raw_temp, dtype = np.float32))
    else:
        model.temp = raw_temp


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description     = "DyGLib/DyGMamba-aligned 1-negative binary evaluation for GSN",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required = True)
    parser.add_argument(
        "--epoch",
        default = None,
        type    = int,
        help    = (
            "Optional 1-indexed epoch to load (epoch_NNN.weights.h5). "
            "Requires the run to have been trained with save_every_epoch: true. "
            "Omit to load best.weights.h5 (best-by-val-MRR)."
        ),
    )
    parser.add_argument("--dataset", required = True)
    parser.add_argument("--root",    default  = "data/")
    parser.add_argument(
        "--batch_events",
        default = 32,
        type    = int,
        help    = "GSN state bucket size; affects scores",
    )
    parser.add_argument(
        "--metric_batch_size",
        default = 200,
        type    = int,
        help    = "DyGLib DataLoader batch size used for AP/AUC aggregation",
    )
    parser.add_argument(
        "--temp",
        default = None,
        type    = float,
        help    = "Optional actual temperature tau. Omit to keep checkpoint value.",
    )
    parser.add_argument(
        "--seed",
        default = 0,
        type    = int,
        help    = "Base eval sampler seed; defaults reproduce DyGLib val=0/test=2 seeds",
    )
    parser.add_argument("--split", default = "both", choices = ["val", "test", "both"])
    parser.add_argument("--gpu",   default = None)
    parser.add_argument(
        "--neg_pool",
        default = "dyglib",
        choices = ["dyglib", "full_dst", "train_dst", "all"],
        help    = (
            "Random NSS destination pool. dyglib/full_dst = unique dst nodes "
            "over train+val+test; train_dst = unique train dst only; all = "
            "all node IDs."
        ),
    )
    parser.add_argument(
        "--no_global_diagnostics",
        action = "store_true",
        help   = "Hide strict whole-split global AP/AUC diagnostics",
    )
    parser.add_argument(
        "--no_inductive",
        action = "store_true",
        help   = "Skip the inductive NSS evaluation (random only).",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    tf.keras.utils.set_random_seed(args.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    print(f"Loading dataset '{args.dataset}' from '{args.root}' ...")
    (train, val, test), meta = load_tgb(args.dataset, root = args.root)
    num_nodes     = int(meta["num_nodes"])
    edge_feat_dim = int(meta["edge_feat_dim"])
    print(f"  {num_nodes} nodes | train {len(train)} | val {len(val)} | test {len(test)}")

    full_src, full_dst, full_ts = _concat_full(train, val, test)
    full_unique_dst             = np.unique(full_dst).astype(np.int64)
    train_unique_dst            = np.unique(train.dst.astype(np.int64)).astype(np.int64)

    if args.neg_pool in ("dyglib", "full_dst"):
        random_dst_pool = full_unique_dst
        pool_desc       = f"DyGLib full-data dst pool: {len(random_dst_pool):,} unique dst nodes"
    elif args.neg_pool == "train_dst":
        random_dst_pool = train_unique_dst
        pool_desc       = f"train-only dst pool: {len(random_dst_pool):,} unique dst nodes"
    else:
        random_dst_pool = np.arange(num_nodes, dtype = np.int64)
        pool_desc       = f"all node IDs: {num_nodes:,} nodes"

    print(f"\nLoading model from '{args.checkpoint}' ...")
    model = GSNLinkPredictor.from_pretrained(
        args.checkpoint, edge_feat_dim = edge_feat_dim, epoch = args.epoch
    )
    if args.temp is not None:
        _assign_actual_temperature(model, args.temp)
        print(f"  Overrode actual temperature tau to {args.temp:.6g}")
    total_params = sum(np.prod(weight.shape) for weight in model.trainable_weights)
    print(f"  Trainable parameters: {total_params:,}")

    val_seed                = int(args.seed)
    test_seed               = int(args.seed) + 2
    val_last_observed_time  = float(train.ts[-1])
    test_last_observed_time = float(val.ts[-1])

    print("\nNegative samplers:")
    print(f"  random val/test — {pool_desc}; no collision repair; seeds {val_seed}/{test_seed}")
    if not args.no_inductive:
        print(
            "  inductive val   — DyGLib edge/time sampler; "
            f"last_observed_time=end(train)={val_last_observed_time:g}; seed={val_seed}"
        )
        print(
            "  inductive test  — DyGLib edge/time sampler; "
            f"last_observed_time=end(val)={test_last_observed_time:g}; seed={test_seed}"
        )

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    if args.split in ("val", "both"):
        print("\nEvaluating validation split ...")
        val_random_sampler = DyGLibRandomNegativeSampler(random_dst_pool, seed = val_seed)
        val_inductive_sampler = None if args.no_inductive else DyGLibInductiveNegativeSampler(
            full_src           = full_src,
            full_dst           = full_dst,
            full_ts            = full_ts,
            last_observed_time = val_last_observed_time,
            seed               = val_seed,
        )
        all_results["val"] = evaluate_split(
            model             = model,
            split             = val,
            random_sampler    = val_random_sampler,
            inductive_sampler = val_inductive_sampler,
            batch_events      = args.batch_events,
            metric_batch_size = args.metric_batch_size,
            label             = "Val",
            report_global     = not args.no_global_diagnostics,
            console           = None,
            compute_loss      = True,
        )

    if args.split in ("test", "both"):
        if args.split == "test":
            print("\nNote: --split test starts from checkpoint/end-of-train state, not end-of-val state.")
        print("\nEvaluating test split ...")
        test_random_sampler = DyGLibRandomNegativeSampler(random_dst_pool, seed = test_seed)
        test_inductive_sampler = None if args.no_inductive else DyGLibInductiveNegativeSampler(
            full_src           = full_src,
            full_dst           = full_dst,
            full_ts            = full_ts,
            last_observed_time = test_last_observed_time,
            seed               = test_seed,
        )
        all_results["test"] = evaluate_split(
            model             = model,
            split             = test,
            random_sampler    = test_random_sampler,
            inductive_sampler = test_inductive_sampler,
            batch_events      = args.batch_events,
            metric_batch_size = args.metric_batch_size,
            label             = "Test",
            report_global     = not args.no_global_diagnostics,
            console           = None,
            compute_loss      = True,
        )

    sep = "=" * 78
    print(f"\n{sep}")
    print("  DyGLib/DyGMamba-aligned evaluation  (1 negative per positive)")
    print(sep)
    print(f"  AP/AUC shown first are mean per-batch sklearn metrics, batch_size={args.metric_batch_size}.")
    print(f"  random    — {pool_desc}")
    if not args.no_inductive:
        print("  inductive — edge/time DyGLib sampler, not unseen-destination-node sampling")
    print(sep)

    for split_name, nss_dict in all_results.items():
        print(f"\n  [{split_name.upper()}]")
        for nss_name in ("random", "inductive"):
            if nss_name not in nss_dict:
                continue
            metrics  = nss_dict[nss_name]
            loss_str = f"  loss={metrics['loss']:.4f}" if "loss" in metrics else ""
            print(
                f"    {nss_name:<12}  "
                f"AP={metrics['ap']:.4f}  AUC={metrics['auc']:.4f}  "
                f"Acc={metrics['acc']:.4f}  MRR@1neg={metrics['mrr_1neg']:.4f}  "
                f"(n={int(metrics['n_events']):,}){loss_str}"
            )
            if not args.no_global_diagnostics and "global_ap" in metrics:
                print(
                    f"    {'global diag':<12}  "
                    f"AP={metrics['global_ap']:.4f}  AUC={metrics['global_auc']:.4f}  "
                    f"Acc={metrics['global_acc']:.4f}  "
                    f"MRR@1neg={metrics['global_mrr_1neg']:.4f}"
                )

    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
