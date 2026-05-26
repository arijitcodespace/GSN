# GSN — Graph State Networks

**Temporal link prediction on continuous-time dynamic graphs with persistent
per-node Mamba-2 state and gated message passing.**

GSN treats a temporal interaction stream as a sequence of mini-graph
*snapshots* (one per time bucket) and maintains a persistent recurrent state
for every node across snapshots. Each snapshot runs:

1.  **Read** the per-node state from a `DenseStateTable`.
2.  **Update** the state with a Mamba-2 state-space step driven by the
    current node features and bucket Δt.
3.  **Mix** node representations via a one-hop, edge-gated message passing
    layer (plus optional FFN).
4.  **Score** candidate (src, dst) edges with a learned dot or MLP scorer.
5.  **Commit** the updated state back to the table (uniform or learned
    per-node EMA), ready for the next snapshot.

The framework is built on **TensorFlow / Keras (>=2.15)** and targets the
[TGB](https://tgb.complexdatalab.com/) benchmark datasets, downloaded
directly from the [Zenodo record 7213796](https://zenodo.org/record/7213796).
No `py-tgb` dependency.

---

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Datasets](#datasets)
- [Repository layout](#repository-layout)
- [How it works](#how-it-works)
  - [The snapshot abstraction](#the-snapshot-abstraction)
  - [GSN block](#gsn-block)
  - [State tables](#state-tables)
  - [Commit modes](#commit-modes)
  - [Conv-cache and intra-bucket sequencing](#conv-cache-and-intra-bucket-sequencing)
- [Training](#training)
  - [CLI](#cli)
  - [Resuming from a checkpoint](#resuming-from-a-checkpoint)
  - [Train-on-val (Regime 3)](#train-on-val-regime-3)
- [Evaluation](#evaluation)
  - [Trainer-vs-evaluator parity](#trainer-vs-evaluator-parity)
  - [Negative sampling strategies](#negative-sampling-strategies)
- [Configuration reference](#configuration-reference)
  - [`dataset`](#dataset-section)
  - [`model`](#model-section)
  - [`trainer`](#trainer-section)
  - [`adaptive_commit`](#adaptive_commit-section)
- [Checkpoints and artefacts](#checkpoints-and-artefacts)
- [Reproducibility](#reproducibility)
- [Tips and gotchas](#tips-and-gotchas)
- [Development](#development)
- [Project status](#project-status)

---

## Installation

GSN targets Python ≥ 3.10 and TensorFlow ≥ 2.15. A CUDA-enabled GPU is
strongly recommended.

```bash
# 1. Create a fresh environment (example: conda)
conda create -n gsn python=3.10 -y
conda activate gsn

# 2. Install TensorFlow with GPU support, then install GSN editable
pip install "tensorflow[and-cuda]>=2.15"
pip install -e ".[dev]"      # editable + black/isort/pytest
```

Direct runtime dependencies (declared in `setup.py`):

- `tensorflow >= 2.15`
- `einops >= 0.7`
- `numpy >= 1.24`
- `pyyaml >= 6.0`
- `tqdm >= 4.66`
- `rich >= 13.0` (used for the training/eval progress bars)

The Mamba-2 SSD kernel ships inside the repo at
`gsn/layers/mamba2/` — no separate `mamba_ssm` install is required for GSN.

---

## Quick start

```bash
# Train on UCI for 1 epoch using the bundled config
python examples/train.py configs/uci.yaml --epochs 1

# Train on Wikipedia, override LR and pick GPU 0
python examples/train.py configs/wikipedia.yaml --lr 1e-4 --gpu 0

# Resume from a saved checkpoint (loads best.weights.h5 by default)
python examples/train.py configs/wikipedia.yaml \
    --checkpoint checkpoints/tgbl-wiki/ \
    --from_epoch 5

# DyGLib/DyGMamba-aligned standalone evaluation
python examples/evaluate.py \
    --dataset tgbl-uci \
    --checkpoint examples/checkpoints/tgbl-uci/ \
    --epoch 1 \
    --seed 2020 \
    --batch_events 256 \
    --metric_batch_size 200
```

All hyperparameters live in the YAML files under `configs/`. CLI flags
override individual fields — see [Configuration reference](#configuration-reference).

---

## Datasets

GSN supports the following continuous-time dynamic graph datasets out of
the box. Each one is fetched on first use from the canonical Zenodo
record and cached locally as `.npz`:

| Config key       | Source name | Bipartite? | Default `time_scale` |
|------------------|-------------|------------|----------------------|
| `tgbl-wiki`      | Wikipedia   | yes        | 60.0                 |
| `tgbl-mooc`      | MOOC        | yes        | 30022                |
| `tgbl-uci`       | UCI         | no         | 1.0                  |
| `tgbl-enron`     | Enron       | no         | 86400.0              |
| `tgbl-uslegis`   | USLegis     | no         | dataset-dependent    |
| `tgbl-canparl`   | CanParl     | no         | dataset-dependent    |
| `tgbl-contacts`  | Contacts    | no         | 300.0                |

The loader caches everything under `<root>/<dataset>/cache/` (default
`root: data/`). It will:

1. Try multiple Zenodo download endpoints (CDN + REST API) with retries.
2. Unzip and convert CSV/NPY into a compact `full_data.npz` + `meta.json`.
3. Hand back three `TGBSplit` objects (`train`, `val`, `test`) plus a
   `meta` dict.

If Zenodo is unreachable from your network you can drop a pre-downloaded
zip into `<root>/<dataset>/` and the loader will pick it up.

> **Heads-up.** Datasets and checkpoints are **not** tracked by git;
> see `.gitignore`. Default download root is `data/` relative to the
> *working directory*. Most invocations use `examples/data/` because the
> example scripts `cd` into `examples/` implicitly via the `root: data/`
> entry in each YAML.

---

## Repository layout

```
GSN/
├── configs/                    # YAML configs (one per dataset)
│   ├── wikipedia.yaml
│   ├── mooc.yaml
│   ├── uci.yaml
│   ├── enron.yaml
│   ├── uslegis.yaml
│   ├── canparl.yaml
│   └── contacts.yaml
│
├── examples/
│   ├── train.py                # CLI training entry point
│   ├── evaluate.py             # CLI evaluation entry point
│   └── checkpoints/<dataset>/  # Saved weights + activity buffers + config
│
├── gsn/
│   ├── snapshot.py             # Snapshot dataclass + bucket builder
│   │
│   ├── datasets/
│   │   ├── tgb_loader.py       # Zenodo loader, TGBSplit, merge_splits
│   │   └── negative_sampling.py# Train-time negative samplers
│   │
│   ├── layers/
│   │   ├── mamba2/             # Mamba-2 SSD TF implementation (step mode)
│   │   ├── gsn_block.py        # GSNBlock + PersistentGSNBlock
│   │   ├── edge_gate.py        # Edge gating MLP
│   │   ├── time_encoding.py    # TGAT time embedding
│   │   ├── link_predictor.py   # Dot / MLP scorers
│   │   └── adaptive_commit_gate.py  # Learned per-node α gate
│   │
│   ├── state/
│   │   ├── table.py            # DenseStateTable (persistent SSM state)
│   │   ├── conv_cache.py       # ConvCacheTable (persistent conv1d cache)
│   │   └── activity_buffers.py # NodeActivityBuffers for adaptive gate
│   │
│   ├── train/
│   │   ├── loop.py             # Trainer + GSNLinkPredictor model
│   │   ├── eval.py             # Shared eval module (trainer & CLI both use this)
│   │   ├── loss.py             # ranking_loss + write_penalty_loss
│   │   └── metrics.py          # MRR / AP / AUC helpers
│   │
│   ├── src/
│   │   └── graph_layer.py      # GraphLayerBackbone (positional-arg policing)
│   │
│   └── utils/                  # ops.py and helpers
│
├── setup.py
└── .gitignore
```

---

## How it works

### The snapshot abstraction

`gsn.snapshot.Snapshot` is a plain dataclass representing a single
time-bucket mini-graph:

```python
Snapshot(
    node_ids,   # [N] int64 — global node IDs in this bucket
    edge_src,   # [E] int32 — local source indices (into node_ids)
    edge_dst,   # [E] int32 — local destination indices
    num_nodes,  # N
    t_ref,      # reference timestamp (bucket end)
    dt,         # seconds since previous bucket
    edge_feat,  # [E, F_e] float32 or None
    edge_ts,    # [E] int64 or None — per-edge timestamps
    x,          # [N, F_n] float32 or None — optional node features
)
```

The training loop slices the event stream into buckets of
`trainer.batch_events` events each, builds a `Snapshot.from_events(...)`
per bucket, and threads it through `PersistentGSNBlock`.

For an *evaluation* bucket, the snapshot is built with extra placeholder
node IDs that cover all required negative-sample sources and
destinations, so the forward pass computes embeddings for every node
that will be scored later in the same bucket.

### GSN block

`gsn.layers.gsn_block.GSNBlock` is **stateless** — the caller passes
the per-node SSM state in and gets the updated state out. One block does:

```
[Read state]  →  Mamba-2 SSM step (with optional conv cache)
              →  one-hop edge-gated message passing
              →  optional FFN
              →  [Updated node embeddings, updated state]
```

`PersistentGSNBlock` wraps `GSNBlock` with automatic
`DenseStateTable` / `ConvCacheTable` read-and-write logic, plus
optional `AdaptiveCommitGate` to control how strongly new state
overwrites old.

### State tables

Two non-trainable tables persist across snapshots:

- **`DenseStateTable`** (`gsn/state/table.py`) — the SSM hidden state,
  shape `[num_nodes, num_heads * head_dim * state_dim]`.
- **`ConvCacheTable`** (`gsn/state/conv_cache.py`) — the Mamba-2 causal
  conv1d cache, shape `[num_nodes, conv1d_kernel_size, xbc_channels]`.
  Only allocated when `conv_cache: true`.

Both expose `clone()` / `copy_from()` so the trainer can snapshot
training state before per-epoch evaluation and restore it afterwards —
guaranteeing that eval does not leak information back into training and
that saved checkpoints reflect the *end-of-train* state cleanly.

### Commit modes

After each snapshot the new state `s'` is blended back into the table:

```
S ← (1 − α) · S + α · s'
```

- **`uniform`** — `α = commit_alpha`, a single scalar shared by all
  nodes (`model.commit_alpha` in the YAML). Simple, fast, hyperparam-only.
- **`adaptive_hazard`** — `α_{i,k}` is learned per-node via a small MLP
  taking 7 features per node (Δt, event count, novelty, cosine change,
  etc.) parameterised as a continuous-time hazard:

  ```
  α_{i,k} = α_min + (α_max − α_min) · (1 − exp(−λ_{i,k} · φ_{i,k}))
  ```

  See `gsn/layers/adaptive_commit_gate.py` for the full formulation.
  Enable by setting `adaptive_commit.commit_mode: adaptive_hazard`.

### Conv-cache and intra-bucket sequencing

Two ablation flags toggle key behaviours of the Mamba-2 step:

- **`conv_cache: true`** — keep a persistent Mamba-2 conv1d cache per
  node so that the causal 1-D conv inside the SSM step sees true
  streaming history across snapshots instead of zero-padding every step.
  Required if you want the local conv to actually do anything in step mode.
- **`intra_bucket_seq: true`** — instead of one aggregated SSM step
  per bucket, run the SSM event-by-event within each bucket. Preserves
  intra-bucket temporal order at the cost of throughput.
- **`pre_message: true`** — feed a per-destination neighbour summary
  into the SSM input (gated by a learnable scalar initialised to 0,
  so identity at init). Lets the committed state ingest interaction
  information directly.

These are all per-config and default to `false` in most configs.

---

## Training

### CLI

```text
python examples/train.py CONFIG [options]

required:
  CONFIG                 Path to a YAML config (e.g. configs/uci.yaml)

common overrides:
  --epochs N             Override trainer.epochs
  --initial_epoch N      Override trainer.initial_epoch (for resume)
  --lr FLOAT             Override trainer.lr
  --hidden INT           Override model.hidden
  --num_layers INT       Override model.num_layers
  --batch_events INT     Override trainer.batch_events
  --commit_alpha FLOAT   Override model.commit_alpha
  --lambda_wr FLOAT      Override trainer.lambda_wr (write-penalty weight)
  --weights_dir PATH     Where to save *.weights.h5 (default per-YAML)
  --root PATH            Dataset root (default: data/)
  --seed INT             Override trainer.seed
  --gpu STR              CUDA_VISIBLE_DEVICES, e.g. "0" or "0,1"

resume:
  --checkpoint PATH      Directory containing previous weights
  --from_epoch INT       Specific epoch (default: best.weights.h5)
```

The trainer prints a Rich progress bar per bucket and a one-line summary
per epoch:

```
─── Epoch 1 / 1 ───
Tra…  bucket 164/164 - loss: 0.50 - rank: 0.50 - MRR_1v1: 0.91 - wr: 0.00
Val   bucket  36/36  - loss: 0.66  MRR: 0.91  AP: 0.82  AUC: 0.82
  Val [inductive]   loss: 0.66  MRR: 0.81  AP: 0.63  AUC: 0.62
Test  bucket  36/36  - loss: 0.58  MRR: 0.94  AP: 0.89  AUC: 0.87
  Test [inductive]  loss: 0.58  MRR: 0.83  AP: 0.67  AUC: 0.66
```

### Resuming from a checkpoint

```bash
python examples/train.py configs/uci.yaml \
    --checkpoint examples/checkpoints/tgbl-uci/ \
    --from_epoch 7 \
    --epochs 20
```

The loader replays the saved `config.json` to reconstruct the model
exactly, then `load_weights(skip_mismatch=True)` restores trainable
**and** non-trainable tensors (state table + conv cache + activity
buffers when present). The current YAML's `adaptive_commit` settings are
always re-applied as overrides so you can upgrade a `uniform` checkpoint
to `adaptive_hazard` without retraining from scratch.

### Train-on-val (Regime 3)

Once hyperparameters have been frozen using the train→val signal you can
re-run with `train_on_val: true` to absorb the validation events into
the training stream. Useful for getting a stronger test-time model.
When this flag is set:

- Validation is still computed each epoch but is essentially a
  memorisation check (val MRR → 1).
- Use the **test** MRR for model selection.
- The trainer keeps the original (un-merged) train split as
  `eval_train`, which is what the inductive negative sampler needs for
  its `last_observed_time = end(train)` cutoff. (This subtlety used to
  cause a silent metric drift — now handled automatically.)

---

## Evaluation

### Trainer-vs-evaluator parity

Both the trainer's per-epoch eval and `examples/evaluate.py` go through
the **same** function: `gsn.train.eval.evaluate_split(...)`. This is
the single source of truth for:

- Negative sampling (`DyGLibRandomNegativeSampler`,
  `DyGLibInductiveNegativeSampler`).
- Snapshot construction with the right "extras" packed into
  `snap_pre.node_ids`.
- AP/AUC aggregation (mean per-batch sklearn metrics over chunks of
  `--metric_batch_size`).
- BCE loss computation.

A correctly-saved checkpoint will produce **byte-for-byte identical**
trainer- and evaluator-reported metrics for the same epoch, given the
same `seed`, `batch_events`, and `metric_batch_size`. This is enforced
by the test workflow (`pytest gsn/tests/`).

### CLI

```text
python examples/evaluate.py [options]

required:
  --dataset NAME           e.g. tgbl-uci, tgbl-wiki, ...
  --checkpoint PATH        Directory with weights and config.json

common:
  --epoch INT              Which epoch_NNN.weights.h5 to load (default: best)
  --root PATH              Dataset root (default: data/)
  --batch_events INT       State-bucket size (default: 1024)
  --metric_batch_size INT  sklearn AP/AUC batch (default: 200)
  --temp FLOAT             Override scorer temperature τ
  --seed INT               Eval seed (default: 1337; pair with trainer seed)
  --split {val,test,both}  Default: both
  --neg_pool {dyglib,full_dst,train_dst,all}
                           Random NSS destination pool
  --no_inductive           Skip the inductive NSS eval
  --no_global_diagnostics  Hide the strict whole-split global AP/AUC line
  --gpu STR                CUDA_VISIBLE_DEVICES
```

### Negative sampling strategies

- **Random NSS (`random`)** — for each positive (s, d, t) sample one
  negative destination `d⁻` from the unique full-data destination pool
  (no collision repair with `d`). Reports MRR@1neg.
- **Inductive NSS (`inductive`)** — DyGLib edge/time sampler. Samples
  negative *edges* drawn from `historical_edges − observed_edges −
  current_batch_edges`. This is **not** unseen-destination-node
  sampling; the negative source can differ from the positive source.

The evaluator additionally reports a "global diag" line per strategy:
the strict whole-split AP/AUC (one global threshold over all positives
and negatives), which catches calibration issues that batch-mean metrics
can hide.

---

## Configuration reference

Every config has four top-level sections: `dataset`, `model`, `trainer`,
`adaptive_commit`. The examples below use `configs/uci.yaml` for
reference values; consult the per-dataset YAML for tuned defaults.

### `dataset` section

| Key   | Type | Description |
|-------|------|-------------|
| `name` | str | Dataset key. One of `tgbl-wiki`, `tgbl-mooc`, `tgbl-uci`, `tgbl-enron`, `tgbl-uslegis`, `tgbl-canparl`, `tgbl-contacts`. |
| `root` | str | Local cache root for downloaded files (default `data/`). |

### `model` section

| Key | Type | Description |
|-----|------|-------------|
| `hidden` | int | Model width `d_model`. Must equal `num_heads * head_dim`. |
| `num_heads` | int | Number of Mamba-2 SSM heads. |
| `head_dim` | int | Per-head dimension. |
| `state_dim` | int | SSM state dim per head (`N`). |
| `num_layers` | int | Number of stacked GSN blocks. |
| `embed_dim` | int | Node ID embedding dim fed to the scorer. |
| `scorer` | str | `dot` or `mlp`. |
| `commit_alpha` | float | Uniform EMA commit rate (only used when `commit_mode: uniform`). |
| `time_feat_dim` | int | TGAT time-encoding dimension. |
| `time_scale` | float | Δt normaliser, in seconds. Pick something near the median inter-event gap. |
| `edge_gate_hidden` | int | Hidden units in `EdgeGate`. 0 = linear gate. |
| `dropout` | float | FFN dropout rate. |
| `self_loops` | bool | Add self-loops in message passing. |
| `pre_message` | bool | **B0** ablation: feed neighbour summary into SSM input (gated, init = 0). |
| `conv_cache` | bool | **B** ablation: persistent Mamba-2 conv1d cache across snapshots. |
| `conv_cache_dt_decay` | float \| null | Optional Δt-staleness decay τ applied to the read cache. |
| `intra_bucket_seq` | bool | **C** ablation: per-event SSM stepping within each bucket. |
| `conv1d_kernel_size` | int | Mamba-2 conv kernel width (effective horizon: K × `batch_events` events; only active with `conv_cache`). |
| `noise_scale` | float | Gaussian noise injected into state during training (regulariser). |
| `id_dim` | int | Width of trainable per-node ID embedding (0 to disable). |
| `temp` | float | Scorer temperature (in the unparameterised raw scale; the model applies softplus internally). |

### `trainer` section

| Key | Type | Description |
|-----|------|-------------|
| `lr` | float | Adam learning rate. |
| `beta_1`, `beta_2` | float | Adam momenta. |
| `weight_decay` | float | AdamW-style decoupled weight decay (0 = vanilla Adam). |
| `clip_norm` | float \| null | Global-norm gradient clip. |
| `loss_fn` | str | `ce` (categorical-ish ranking) or `bce`. |
| `lambda_wr` | float | Weight of the write-penalty loss (discourages over-eager state writes). |
| `epochs` | int | Number of training epochs. |
| `initial_epoch` | int | First-epoch index (for resume bookkeeping). |
| `batch_events` | int | Events per snapshot/bucket. |
| `accumulate_every` | int | Gradient accumulation count (1 = no accumulation). |
| `train_neg_per_pos` | int | Negatives per positive during training. |
| `val_test_neg_per_pos` | int | Negatives per positive during eval. Set to 1 for DyGLib-style; -1 means "use precomputed all-negatives" (legacy path). |
| `seed` | int | RNG seed for training + sampler. |
| `weights_dir` | str | Where to save `epoch_NNN.weights.h5`, `best.weights.h5`, `config.json`, and (if applicable) `activity_buffers.npz`. |
| `save_every_epoch` | bool | If true, save a per-epoch checkpoint (useful for post-hoc model selection). |
| `train_on_val` | bool | **Regime 3.** Absorb val into the training stream. Hyperparameters must already be tuned. |

### `adaptive_commit` section

Ignored when `commit_mode: uniform`. Configures the learned per-node α
gate.

| Key | Type | Description |
|-----|------|-------------|
| `commit_mode` | str | `uniform` or `adaptive_hazard`. |
| `gate_hidden` | int | Hidden width of the MLP feeding the hazard rate. |
| `gate_layers` | int | Number of MLP layers (excluding final projection). |
| `alpha_min`, `alpha_max` | float | Bounds on the per-node α (e.g. 1e-4, 0.999). |
| `lambda_min` | float | Lower bound on the hazard rate λ. |
| `exposure_delta0` | float | Exposure floor (prevents φ = 0 at Δt = 0 events). |
| `exposure_cn` | float | Coefficient on the `log(1 + n)` event-count term inside φ. |
| `lambda_alpha_prior` | float | Weight of the α-prior regulariser. |
| `lambda_alpha_saturation` | float | Weight of the α-saturation regulariser (discourages α → bounds). |
| `alpha_warmup_epochs` | int | Number of warmup epochs blending the gate toward a fixed α₀. |

---

## Checkpoints and artefacts

For each training run with `save_every_epoch: true` the trainer writes:

```
<weights_dir>/
├── config.json                  # full model + adaptive_commit config (for resume)
├── best.weights.h5              # best-val-MRR snapshot
├── epoch_001.weights.h5         # per-epoch snapshots
├── epoch_002.weights.h5
├── ...
└── activity_buffers.npz         # only when commit_mode = adaptive_hazard
```

Use `--from_epoch N` on the evaluator or trainer to point at a specific
snapshot. `best.weights.h5` is the model with the highest val MRR seen so
far. All checkpoint files are gitignored.

---

## Reproducibility

- The trainer seeds NumPy / TF / Keras via `trainer.seed`. The standalone
  evaluator additionally enables `tf.config.experimental.enable_op_determinism()`
  for the eval forward pass.
- Negative samplers are seeded explicitly:
  `val_seed = seed`, `test_seed = seed + 2` (random NSS),
  identical seeding for the inductive NSS.
- The `Trainer` snapshots both the state table **and** the conv cache
  before per-epoch evaluation and restores them afterwards. Without
  this, `epoch_N+1` would start with a cache polluted by `epoch_N`'s
  val+test events, which silently hurts training quality on
  `conv_cache: true` configs.

---

## Tips and gotchas

- **`hidden == num_heads * head_dim`** is asserted at construction.
- **`time_scale`** has a much bigger effect than people expect. As a
  rule of thumb, pick something near the median inter-event Δt of the
  dataset. The bundled configs already do this.
- **`batch_events`** trades off temporal fidelity vs throughput. Very
  small buckets preserve order but kill GPU utilisation. Conversely
  very large buckets aggregate too much into one SSM step.
- **`val_test_neg_per_pos: 1`** is the DyGLib/DyGMamba convention and is
  what `evaluate.py` expects by default. Setting it higher activates a
  different, multi-negative-per-positive ranking path.
- **Inductive NSS ≠ unseen-destination-node sampling**. It samples
  negative *edges* drawn from `historical − observed − current_batch`.
  The negative source can differ from the positive source.
- **`train_on_val: true`** changes val MRR from a model-selection
  signal into a memorisation check. Always use **test** MRR for
  decisions in that regime.
- **`pg.ipynb`, `debug.py`** in `examples/` are scratch space. They are
  intentionally not part of the API but are kept tracked.
- The Mamba-2 implementation in `gsn/layers/mamba2/` runs in **step
  mode** (sequence_length = 1) — one snapshot at a time. Enable
  `conv_cache: true` to recover the conv1d's short-range temporal mixing
  across steps; without it the local conv sees only zeros and is
  effectively disabled.

---

## Development

```bash
# Format
black gsn/ examples/
isort gsn/ examples/

# Tests
pytest gsn/tests/
pytest gsn/tests/ -k "test_name"
```

Coding conventions:

- Most modules try relative imports first and fall back to absolute
  imports with a warning. This supports both `import gsn` (package mode)
  and direct script execution.
- `GraphLayerBackbone` (`gsn/src/graph_layer.py`) enforces a fixed
  number of positional call args via `PinArgs` / `num_call_args` to
  prevent silently breaking the API contract.
- `@tf.function` boundaries on the state tables use
  `reduce_retracing=True` and explicit `input_signature` — be careful
  when changing their shapes.

---

## Project status

GSN is research-grade code: the public surface is small and stable
(`Snapshot`, `GSNLinkPredictor`, `Trainer`, `evaluate_split`,
`load_dataset`), but internal layers are evolving. The current focus is:

- Closing the AP/MRR gap against DyGMamba on fine-grained datasets.
- Tuning the adaptive commit gate on UCI, MOOC, Contacts.
- Maintaining strict trainer-vs-evaluator metric parity.

Contributions and bug reports are welcome — please include the YAML
config, the random seed, and (if relevant) the checkpoint that
reproduces the issue.