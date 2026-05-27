#!/usr/bin/env python
"""Unified GSN training script.

Usage
-----
    python examples/train.py configs/wikipedia.yaml
    python examples/train.py configs/mooc.yaml --epochs 3 --lr 1e-3
    python examples/train.py configs/enron.yaml --weights_dir runs/enron/

All hyperparameters live in the YAML config; CLI flags override individual fields.
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict

# Make the repo root importable when running directly
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import yaml
import tensorflow as tf

tf.keras.config.disable_traceback_filtering()

from gsn.datasets import load_dataset, merge_splits
from gsn.train.loop import GSNLinkPredictor, Trainer, TrainerConfig


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _apply_overrides(cfg: dict, overrides: dict) -> dict:
    """Deep-merge CLI overrides into the flat trainer/model sub-dicts."""
    for key, val in overrides.items():
        if val is None:
            continue
        # Try trainer first, then model, then dataset
        for section in ("trainer", "model", "dataset"):
            if key in cfg.get(section, {}):
                cfg[section][key] = val
                break
        else:
            # Unknown key: add to trainer
            cfg.setdefault("trainer", {})[key] = val
    return cfg

def display_dataclass(d: dataclass, header: Optional[str] = None, span: int = 75):
    import shutil
    width = shutil.get_terminal_size().columns
    
    span = min(span, width)
    
    datadict = asdict(d)
    keys = [k for k, _ in datadict.items()]
    values = [v for _, v in datadict.items()]
    
    max_key_len = max([len(str(k)) for k in keys])
    max_value_len = max([len(str(v)) for v in values])
    field_length = max(max_key_len, max_value_len) + 5
    border = "=" * span
    top_rule = "-" * span
    padded_header = (
                        " " * ((span - len(str(header))) // 2 if header is not None else span // 2) +\
                            (header if header is not None else "")+\
                        " " * ((span - len(str(header))) // 2 if header is not None else span // 2)
                    )
    print(border)
    print(padded_header)
    print(top_rule)
    for (k, v) in datadict.items():
        print(str(k) + " " * (field_length - len(str(k))), ':', " " * (field_length - len(str(v))) + str(v))
    
    print(border)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description = "Train a GSN on a TGB dataset")
    parser.add_argument("config", help = "Path to a YAML config file (e.g. configs/wikipedia.yaml)")
    parser.add_argument("--epochs",         type = int,   default = None)
    parser.add_argument("--initial_epoch",  type = int,   default = None)
    parser.add_argument("--lr",             type = float, default = None)
    parser.add_argument("--hidden",         type = int,   default = None)
    parser.add_argument("--num_layers",     type = int,   default = None)
    parser.add_argument("--batch_events",   type = int,   default = None)
    parser.add_argument("--commit_alpha",   type = float, default = None)
    parser.add_argument("--lambda_wr",      type = float, default = None)
    parser.add_argument("--weights_dir",    type = str,   default = None)
    parser.add_argument("--root",           type = str,   default = None,
                        help = "Override dataset root directory")
    parser.add_argument("--seed",           type = int,   default = None)
    parser.add_argument("--gpu",            type = str,   default = None,
                        help = "Comma-separated GPU indices to use (e.g. '0' or '0,1')")
    parser.add_argument("--checkpoint", type = str, default = None,
                        help = "Specify checkpoint from which to continue training, e.g., ./checkpoints/tgbl-wiki/")
    parser.add_argument("--from_epoch", type = int, default = None,
                        help = "Specify which epoch's checkpoint to take. Defaults to best.weights.h5")
    args = parser.parse_args()

    # GPU selection
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Load config and apply CLI overrides
    raw = _load_yaml(args.config)
    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("config", "gpu", "checkpoint") and v is not None}
    raw = _apply_overrides(raw, overrides)

    ds_cfg  = raw["dataset"]
    mdl_cfg = raw["model"]
    trn_cfg = raw["trainer"]

    name = ds_cfg["name"]
    root = ds_cfg.get("root", "data/")
    
    tf.random.set_seed(raw["trainer"]["seed"])
    tf.keras.utils.set_random_seed(raw["trainer"]["seed"])

    neg_per_pos = int(trn_cfg["val_test_neg_per_pos"])
    # val_test_neg_per_pos=-1 means "all negatives": precompute and attach to splits
    # so that OfficialNegativeSampler is used in the Trainer instead of TrainNegativeSampler.
    precompute_negs = (neg_per_pos == -1)

    print(f"Loading dataset '{name}' from '{root}' ...")
    (train, val, test), meta = load_dataset(
                                                name, root = root, cache = True,
                                                precompute_negatives = precompute_negs,
                                                num_neg_e            = neg_per_pos,
                                                split_by             = "events"
                                            )
    num_nodes = meta["num_nodes"]
    print(f"  {num_nodes} nodes | train {len(train)} | val {len(val)} | test {len(test)}")

    # Optional regime-3 retraining: absorb val into the training stream.
    # Useful only after hyperparameters have been frozen using the train→val
    # signal. Val will still be evaluated each epoch but its metrics become a
    # memorisation check (val_mrr → ~1), NOT a model-selection signal.
    train_on_val = bool(trn_cfg.get("train_on_val", False))
    eval_train = train  # original (un-merged) train; needed for inductive cutoffs
    if train_on_val:
        merged = merge_splits(train, val)
        print(
            "  [train_on_val=True] merging val into training stream: "
            f"train {len(train)} + val {len(val)} = {len(merged)} events. "
            "Val MRR in subsequent logs is informational (memorisation), "
            "not a model-selection signal — use test MRR for evaluation."
        )
        train = merged

    # Validate head_dim consistency
    hidden   = int(mdl_cfg["hidden"])
    num_heads = int(mdl_cfg["num_heads"])
    head_dim  = int(mdl_cfg.get("head_dim", hidden // num_heads))
    assert hidden == num_heads * head_dim, f"hidden ({hidden}) must equal num_heads * head_dim ({num_heads} * {head_dim})"
    
    # Adaptive commit section (optional; defaults to uniform mode if absent)
    ac_cfg = raw.get("adaptive_commit", {})

    if args.checkpoint is not None:
        # Always forward the YAML's adaptive_commit settings as overrides so
        # that (a) an old uniform checkpoint can be upgraded to adaptive_hazard
        # without re-training from scratch, and (b) the current YAML's commit
        # mode is respected even when config.json in the checkpoint is stale.
        config_override = {
                            "pair_recurrence": bool(mdl_cfg.get("pair_recurrence", False)),
                            "pair_recurrence_dim": int(mdl_cfg.get("pair_recurrence_dim", 16)),
                            "pair_recurrence_tau": (
                                float(mdl_cfg["pair_recurrence_tau"])
                                if mdl_cfg.get("pair_recurrence_tau") is not None else None
                            ),
                            "pair_recurrence_undirected": bool(mdl_cfg.get("pair_recurrence_undirected", False)),
                            "pair_recurrence_reset_per_epoch": bool(mdl_cfg.get("pair_recurrence_reset_per_epoch", True)),
                            "query_history": bool(mdl_cfg.get("query_history", False)),
                            "query_history_k": int(mdl_cfg.get("query_history_k", 16)),
                            "query_history_dim": int(mdl_cfg.get("query_history_dim", 16)),
                            "query_history_tau": (
                                float(mdl_cfg["query_history_tau"])
                                if mdl_cfg.get("query_history_tau") is not None else None
                            ),
                            "query_history_undirected": bool(mdl_cfg.get("query_history_undirected", True)),
                            "query_history_reset_per_epoch": bool(mdl_cfg.get("query_history_reset_per_epoch", True)),
                            "commit_mode":     str(ac_cfg.get("commit_mode",     "uniform")),
                            "gate_hidden":     int(ac_cfg.get("gate_hidden",     64)),
                            "gate_layers":     int(ac_cfg.get("gate_layers",     2)),
                            "alpha_min":       float(ac_cfg.get("alpha_min",     1e-4)),
                            "alpha_max":       float(ac_cfg.get("alpha_max",     0.999)),
                            "lambda_min":      float(ac_cfg.get("lambda_min",    1e-5)),
                            "exposure_delta0": float(ac_cfg.get("exposure_delta0", 0.05)),
                            "exposure_cn":     float(ac_cfg.get("exposure_cn",   0.25)),
                      }
        model = GSNLinkPredictor.from_pretrained(
                                                    args.checkpoint,
                                                    edge_feat_dim   = int(meta["edge_feat_dim"]),
                                                   config_override = config_override,
                                                    epoch = args.from_epoch
                                                )
    else:
        print("\nBuilding model ...")
        model = GSNLinkPredictor(
                                    num_nodes        = num_nodes,
                                    hidden           = hidden,
                                    num_heads        = num_heads,
                                    head_dim         = head_dim,
                                    state_dim        = int(mdl_cfg.get("state_dim", 16)),
                                    num_layers       = int(mdl_cfg.get("num_layers", 1)),
                                    embed_dim        = int(mdl_cfg.get("embed_dim", hidden)),
                                    scorer           = str(mdl_cfg.get("scorer", "mlp")),
                                    commit_alpha     = float(mdl_cfg.get("commit_alpha", 0.2)),
                                    time_feat_dim    = int(mdl_cfg.get("time_feat_dim", 8)),
                                    time_scale       = float(mdl_cfg.get("time_scale", 86400.0)),
                                    edge_gate_hidden = int(mdl_cfg.get("edge_gate_hidden", 32)),
                                    dropout             = float(mdl_cfg.get("dropout", 0.0)),
                                    self_loops          = bool(mdl_cfg.get("self_loops", True)),
                                    pre_message         = bool(mdl_cfg.get("pre_message", False)),
                                    conv_cache          = bool(mdl_cfg.get("conv_cache", False)),
                                    conv_cache_dt_decay = (
                                        float(mdl_cfg["conv_cache_dt_decay"])
                                        if mdl_cfg.get("conv_cache_dt_decay") else None
                                    ),
                                    intra_bucket_seq    = bool(mdl_cfg.get("intra_bucket_seq", False)),
                                    conv1d_kernel_size  = int(mdl_cfg.get("conv1d_kernel_size", 4)),
                                    noise_scale         = float(mdl_cfg.get("noise_scale", 0.005)),
                                    id_dim           = int(mdl_cfg.get("id_dim", 0)),
                                    temp             = float(mdl_cfg.get("temp", -2.624)),
                                    pair_recurrence  = bool(mdl_cfg.get("pair_recurrence", False)),
                                    pair_recurrence_dim = int(mdl_cfg.get("pair_recurrence_dim", 16)),
                                    pair_recurrence_tau = (
                                        float(mdl_cfg["pair_recurrence_tau"])
                                        if mdl_cfg.get("pair_recurrence_tau") is not None else None
                                    ),
                                    pair_recurrence_undirected = bool(mdl_cfg.get("pair_recurrence_undirected", False)),
                                    pair_recurrence_reset_per_epoch = bool(mdl_cfg.get("pair_recurrence_reset_per_epoch", True)),
                                    query_history  = bool(mdl_cfg.get("query_history", False)),
                                    query_history_k = int(mdl_cfg.get("query_history_k", 16)),
                                    query_history_dim = int(mdl_cfg.get("query_history_dim", 16)),
                                    query_history_tau = (
                                        float(mdl_cfg["query_history_tau"])
                                        if mdl_cfg.get("query_history_tau") is not None else None
                                    ),
                                    query_history_undirected = bool(mdl_cfg.get("query_history_undirected", True)),
                                    query_history_reset_per_epoch = bool(mdl_cfg.get("query_history_reset_per_epoch", True)),
                                    # Adaptive commit (ignored when commit_mode = "uniform")
                                    commit_mode      = str(ac_cfg.get("commit_mode", "uniform")),
                                    gate_hidden      = int(ac_cfg.get("gate_hidden", 64)),
                                    gate_layers      = int(ac_cfg.get("gate_layers", 2)),
                                    alpha_min        = float(ac_cfg.get("alpha_min", 1e-4)),
                                    alpha_max        = float(ac_cfg.get("alpha_max", 0.999)),
                                    lambda_min       = float(ac_cfg.get("lambda_min", 1e-5)),
                                    exposure_delta0  = float(ac_cfg.get("exposure_delta0", 0.05)),
                                    exposure_cn      = float(ac_cfg.get("exposure_cn", 0.25)),
                                )

    cfg = TrainerConfig(
                            lr              = float(trn_cfg.get("lr", 3e-4)),
                            beta_1          = float(trn_cfg.get("beta_1", 0.9)),
                            beta_2          = float(trn_cfg.get("beta_2", 0.999)),
                            weight_decay    = float(trn_cfg.get("weight_decay", 0.0)),
                            clip_norm       = trn_cfg.get("clip_norm"),
                            loss_fn         = str(trn_cfg.get("loss_fn", "ce")),
                            lambda_wr       = float(trn_cfg.get("lambda_wr", 1e-4)),
                            epochs          = int(trn_cfg.get("epochs", 5)),
                            initial_epoch   = int(trn_cfg.get("initial_epoch", 0)),
                            batch_events    = int(trn_cfg.get("batch_events", 20_000)),
                            accumulate_every     = int(trn_cfg.get("accumulate_every", 1)),
                            train_neg_per_pos    = int(trn_cfg.get("train_neg_per_pos", 1)),
                            val_test_neg_per_pos = int(trn_cfg.get("val_test_neg_per_pos", 49)),
                            seed                 = int(trn_cfg.get("seed", 1337)),
                            weights_dir          = trn_cfg.get("weights_dir"),
                            save_every_epoch     = bool(trn_cfg.get("save_every_epoch", False)),
                            train_on_val         = train_on_val,
                            # Adaptive commit trainer settings
                            lambda_alpha_prior      = float(ac_cfg.get("lambda_alpha_prior", 1e-3)),
                            lambda_alpha_saturation = float(ac_cfg.get("lambda_alpha_saturation", 1e-4)),
                            alpha_warmup_epochs     = int(ac_cfg.get("alpha_warmup_epochs", 2)),
                        )

    display_dataclass(cfg, header = "Training Config")

    trainer = Trainer(
                        model = model,
                        cfg   = cfg,
                        train = train,
                        val   = val,
                        test  = test,
                        meta  = meta,
                        eval_train = eval_train,
                    )

    print(f"\nTraining for {cfg.epochs} epoch(s) ...")
    history = trainer.fit()

    best_val  = max(history["val_mrr"])
    best_test = history["test_mrr"][history["val_mrr"].index(best_val)]
    print(f"\nDone. Best val MRR={best_val:.4f}  (test MRR={best_test:.4f})")


if __name__ == "__main__":
    main()
