# Copilot Instructions — GSN (Graph State Networks)

## Project Overview

GSN implements temporal link prediction on dynamic graphs using Mamba-2 SSM (State Space Model) for recurrent node state updates combined with gated message passing. The framework is built on **TensorFlow/Keras** (≥2.15) and targets TGB (Temporal Graph Benchmark) datasets downloaded from Zenodo.

## Build & Run

```bash
# Install (editable, with dev tools)
pip install -e ".[dev]"

# Train on a dataset
python examples/train.py configs/wikipedia.yaml
python examples/train.py configs/mooc.yaml --epochs 3 --lr 1e-3 --gpu 0

# Resume from checkpoint
python examples/train.py configs/wikipedia.yaml --checkpoint checkpoints/tgbl-wiki/

# Run tests
pytest gsn/tests/
pytest gsn/tests/test_something.py -k "test_name"

# Formatting
black gsn/ examples/
isort gsn/ examples/
```

## Architecture

The core pipeline per time bucket (Algorithm 1):

1. **Snapshot** (`gsn/snapshot.py`) — A `Snapshot` dataclass represents a mini-graph for one time bucket, mapping global node IDs to a compact local index space.
2. **DenseStateTable** (`gsn/state/table.py`) — A non-trainable `[num_nodes, state_dim]` tensor storing persistent per-node SSM state. Supports EMA commit via `put(ids, new_states, alpha)`.
3. **GSNBlock** (`gsn/layers/gsn_block.py`) — The core computation block: Mamba-2 SSM step → gated message passing → FFN. It is **stateless**; the caller passes state in/out.
4. **PersistentGSNBlock** — Wraps GSNBlock with automatic DenseStateTable read/write and optional AdaptiveCommitGate.
5. **GSNLinkPredictor** / **Trainer** (`gsn/train/loop.py`) — End-to-end model and training loop. Handles negative sampling, gradient accumulation, MRR/AP evaluation, and checkpointing.

### Key data flow

```
Events → Snapshot.from_events() → PersistentGSNBlock(snap) → node embeddings
    → LinkPredictor (dot/MLP scorer) → logits → ranking_loss
    → commit updated states back to DenseStateTable
```

### Mamba-2 Integration

The SSM core lives in `gsn/layers/mamba2/` and wraps a custom TensorFlow Mamba-2 SSD implementation. It runs in **step mode** (sequence_length=1, num_chunks=1) — one snapshot at a time, not a full sequence.

## Key Conventions

- **Relative + absolute import fallback**: Most modules try relative imports first, then fall back to absolute imports with a warning. This supports both `import gsn` (package mode) and direct script execution.
- **GraphLayerBackbone** (`gsn/src/graph_layer.py`): Base class enforcing a fixed number of positional call args via `PinArgs` context manager or explicit `num_call_args`.
- **Config-driven**: All hyperparameters live in YAML configs (`configs/`). CLI flags override individual fields. The config has four sections: `dataset`, `model`, `trainer`, `adaptive_commit`.
- **Adaptive commit gate** (`adaptive_hazard` mode): A learned per-node α that controls how much new state overwrites old state. Controlled via the `adaptive_commit` config section.
- **State management**: Node states persist across snapshots via `DenseStateTable`. States are reset between epochs. The `NodeActivityBuffers` tracks per-node activity for the adaptive gate.
- **Loss**: `ranking_loss` (CE or BCE over candidate blocks) + optional `write_penalty_loss` + optional alpha regularizers.
- **Metrics**: MRR is the primary evaluation metric; AP and AUC are also available.
- **TF function tracing**: `DenseStateTable` methods use `@tf.function` with `reduce_retracing=True` and explicit `input_signature` for performance.
- **Datasets**: Downloaded from Zenodo (record 7213796), not pip-installed TGB. Supported: `tgbl-wiki`, `tgbl-mooc`, `tgbl-enron`, etc.

## Configuration Reference

See `configs/wikipedia.yaml` for a complete annotated example. Key parameters:

- `model.hidden` / `num_heads` / `head_dim`: Must satisfy `hidden == num_heads * head_dim`
- `model.commit_alpha`: Default uniform commit rate (used when `commit_mode: uniform`)
- `trainer.batch_events`: Number of events per training batch/snapshot
- `adaptive_commit.commit_mode`: `"uniform"` or `"adaptive_hazard"`
