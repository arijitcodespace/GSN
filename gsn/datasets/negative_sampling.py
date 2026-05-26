"""Negative samplers for GSN temporal link prediction.

TGBStyleTrainNegativeSampler — hist_rnd / rnd strategy (training)
TrainNegativeSampler         — simple uniform sampler (eval fallback)
OfficialNegativeSampler      — wraps pre-computed eval negatives
load_official_negatives      — load / compute + cache eval negatives
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# Cache helpers (mirrors tgb_loader.py; kept here to avoid circular import)
# ---------------------------------------------------------------------------

def _cache_dir(root: str, name: str) -> Path:
    p = Path(root) / name / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Training-time negative sampler (hist_rnd / rnd)
# ---------------------------------------------------------------------------

class TGBStyleTrainNegativeSampler:
    """
    TGB-style negative sampler for training.

    Strategies
    ----------
    rnd      : uniform sample from all (or bipartite-dst) nodes, excluding
               same-timestamp positives and optionally self-loops
    hist_rnd : mix of historical neighbours + random non-historical nodes
    """

    def __init__(
                    self,
                    train_src:      np.ndarray,
                    train_dst:      np.ndarray,
                    train_ts:       np.ndarray,
                    num_nodes:      int,
                    num_neg_e:      int,
                    strategy:       str   = "hist_rnd",
                    hist_ratio:     float = 0.5,
                    seed:           int   = 2020,
                    exclude_self:   bool  = True,
                    dst_candidates: Optional[np.ndarray] = None,
                ) -> None:

        assert strategy in ("rnd", "hist_rnd"), f"Unknown strategy: {strategy!r}"
        self.strategy     = strategy
        self.num_neg_e    = int(num_neg_e)
        self.hist_ratio   = float(hist_ratio)
        self.exclude_self = bool(exclude_self)
        self.rng = np.random.default_rng(int(seed))

        # Destination candidate pool (restricted for bipartite graphs)
        if dst_candidates is not None:
            self.all_dst = np.asarray(dst_candidates, dtype = np.int64)
        else:
            self.all_dst = np.arange(int(num_nodes), dtype = np.int64)

        # Positives per (timestamp, source) — for same-timestamp exclusion
        self.pos_ts_edge_dict: Dict[int, Dict[int, np.ndarray]] = {}
        tmp: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        
        for s, d, t in zip(train_src.astype(np.int64), train_dst.astype(np.int64), train_ts.astype(np.int64)):
            tmp[(int(t), int(s))].append(int(d))
        
        for (t, s), lst in tmp.items():
            self.pos_ts_edge_dict.setdefault(int(t), {})[int(s)] = np.asarray(lst, dtype = np.int64)

        # Historical neighbour sets (hist_rnd only)
        self.hist_edge_set_per_node: Dict[int, np.ndarray] = {}
        if strategy == "hist_rnd":
            hist: Dict[int, Set[int]] = defaultdict(set)
            for s, d in zip(train_src.astype(np.int64), train_dst.astype(np.int64)):
                hist[int(s)].add(int(d))
            for s, ds in hist.items():
                self.hist_edge_set_per_node[int(s)] = np.asarray(sorted(ds), dtype = np.int64)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _weighted_choice(self, candidates: np.ndarray, k: int, replace: bool) -> np.ndarray:
        candidates = np.asarray(candidates, dtype=np.int64)
        k = int(k)
        if k <= 0 or candidates.size == 0:
            return np.empty(0, dtype = np.int64)
        if (not replace) and k > candidates.size:
            taken = self.rng.choice(candidates, size = candidates.size, replace = False)
            extra = self.rng.choice(candidates, size = k - candidates.size, replace = True)
            return np.concatenate([taken, extra]).astype(np.int64)
        return self.rng.choice(candidates, size = k, replace = bool(replace)).astype(np.int64)

    def _sample_rnd(self, invalid_dst: np.ndarray, k: int) -> np.ndarray:
        pool = np.setdiff1d(self.all_dst, invalid_dst, assume_unique = False).astype(np.int64)
        return self._weighted_choice(pool, k, replace = False)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def sample_fixed_k(self, u: int, pos_dst_same_ts: np.ndarray) -> np.ndarray:
        """Return negatives for source node `u`.

        pos_dst_same_ts : positive destinations at the current timestamp
                          (and the current positive itself), used for exclusion.

        When num_neg_e == -1 every valid negative destination is returned
        (ragged output — size varies per query).  Otherwise up to num_neg_e
        negatives are returned.
        """
        u               = int(u)
        pos_dst_same_ts = np.asarray(pos_dst_same_ts, dtype = np.int64)

        pos_e = pos_dst_same_ts
        if self.exclude_self:
            pos_e = np.unique(np.concatenate([pos_e, np.asarray([u], dtype = np.int64)]))

        if self.strategy == "rnd":
            pool = np.setdiff1d(self.all_dst, pos_e, assume_unique = False)
            if self.num_neg_e == -1:          # all negatives
                return pool.astype(np.int64)
            if pool.size <= self.num_neg_e:   # fewer available than requested → return all
                return pool.astype(np.int64)
            return self.rng.choice(pool, size = self.num_neg_e, replace = False).astype(np.int64)

        # hist_rnd
        seen = self.hist_edge_set_per_node.get(u, np.empty(0, dtype = np.int64))

        if self.num_neg_e == -1:              # all negatives
            hist_pool = np.setdiff1d(seen, pos_e, assume_unique = False).astype(np.int64)
            rnd_inv   = pos_e if seen.size == 0 else np.unique(np.concatenate([pos_e, seen]))
            rnd_pool  = np.setdiff1d(self.all_dst, rnd_inv, assume_unique = False).astype(np.int64)
            return np.concatenate([hist_pool, rnd_pool]).astype(np.int64)

        max_hist = int(self.num_neg_e * self.hist_ratio)

        hist_pool = np.setdiff1d(seen, pos_e, assume_unique = False)
        hist_take = (
                        self._weighted_choice(hist_pool, max_hist, replace = (max_hist > hist_pool.size))
                        if hist_pool.size > 0 and max_hist > 0
                        else np.empty(0, dtype = np.int64)
                    )

        invalid_rnd = pos_e if seen.size == 0 else np.unique(np.concatenate([pos_e, seen]))
        rnd_need    = self.num_neg_e - int(hist_take.size)
        rnd_take    = (
                        self._sample_rnd(invalid_rnd, rnd_need)
                        if rnd_need > 0
                        else np.empty(0, dtype = np.int64)
                      )

        out = np.concatenate([hist_take, rnd_take]).astype(np.int64)
        if out.size > self.num_neg_e:
            out = self.rng.choice(out, size = self.num_neg_e, replace = False).astype(np.int64)
        return out

    def build_candidates(
                            self,
                            src: np.ndarray,
                            dst: np.ndarray,
                            ts:  Optional[np.ndarray] = None,
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (cand_src, cand_dst, labels, sizes) for a batch.

        Layout per query i: [positive] + [num_neg_e negatives].
        ts is used for same-timestamp positive exclusion; pass None if unknown.
        """
        src = np.asarray(src, dtype = np.int64)
        dst = np.asarray(dst, dtype = np.int64)
        ts  = (np.zeros(src.shape[0], dtype = np.int64)
               if ts is None else np.asarray(ts, dtype = np.int64))

        cand_src_list, cand_dst_list, labels_list, sizes = [], [], [], []

        for i in range(int(src.shape[0])):
            u = int(src[i])
            v = int(dst[i])
            t = int(ts[i])

            t_dict   = self.pos_ts_edge_dict.get(t, {})
            pos_dsts = t_dict.get(u, np.empty(0, dtype = np.int64))
            # Include the current positive in the exclusion set
            pos_dsts = np.unique(np.concatenate([pos_dsts, np.asarray([v], dtype = np.int64)]))

            neg = self.sample_fixed_k(u, pos_dsts)
            Ki  = int(neg.size)

            cand_src_list.append(np.full(1 + Ki, u, dtype = np.int64))
            cand_dst_list.append(np.concatenate([[v], neg]))
            lbl = np.zeros(1 + Ki, dtype = np.float32)
            lbl[0] = 1.0
            labels_list.append(lbl)
            sizes.append(1 + Ki)

        return (
                    np.concatenate(cand_src_list),
                    np.concatenate(cand_dst_list),
                    np.concatenate(labels_list),
                    np.array(sizes, dtype = np.int32),
               )


# ---------------------------------------------------------------------------
# Simple uniform sampler (eval fallback when no official negatives)
# ---------------------------------------------------------------------------

class TrainNegativeSampler:
    """Uniform-random destination sampler.

    Kept for backward compatibility and as a lightweight eval fallback
    when official pre-computed negatives are not available.

    Pass ``dst_pool`` (a sorted int64 array, e.g. ``np.unique(train.dst)``)
    to restrict sampling to a fixed destination-domain pool — required for
    bipartite datasets (tgbl-wiki, tgbl-reddit) so that eval negatives come
    from the same pool the trainer used during training.
    """

    def __init__(
                    self,
                    num_nodes:   int,
                    neg_per_pos: int = 1,
                    seed:        int = 0,
                    dst_pool:    Optional[np.ndarray] = None
                ):
        self.num_nodes   = int(num_nodes)
        self.neg_per_pos = int(neg_per_pos)
        self.rng         = np.random.default_rng(seed)
        self.dst_pool: Optional[np.ndarray] = (
            np.asarray(dst_pool, dtype=np.int64).copy() if dst_pool is not None else None
        )

    def build_candidates(
                            self,
                            src: np.ndarray,
                            dst: np.ndarray,
                            ts: Any = None  # API compatibility
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (cand_src, cand_dst, labels, sizes).

        Layout per query i: [positive] + [neg_per_pos negatives].
        """
        B = src.shape[0]
        K = self.neg_per_pos
        if self.dst_pool is not None:
            neg  = self.rng.choice(self.dst_pool, size=(B, K), replace=True).astype(np.int64)
            mask = neg == dst[:, None]
            if mask.any():
                n_hit     = int(mask.sum())
                resampled = self.rng.choice(self.dst_pool, size=n_hit, replace=True).astype(np.int64)
                neg[mask] = resampled
                mask2     = neg == dst[:, None]
                if mask2.any():
                    pool_sorted   = np.sort(self.dst_pool)
                    idx           = np.searchsorted(pool_sorted, neg[mask2])
                    idx           = (idx + 1) % pool_sorted.size
                    neg[mask2]    = pool_sorted[idx]
        else:
            neg  = self.rng.integers(0, self.num_nodes, size=(B, K), dtype=np.int64)
            mask = neg == dst[:, None]
            neg[mask] = (neg[mask] + 1) % self.num_nodes

        cand_src_list, cand_dst_list, labels_list, sizes = [], [], [], []
        for i in range(B):
            u = int(src[i])
            cand_src_list.append(np.full(1 + K, u, dtype = np.int64))
            cand_dst_list.append(np.concatenate([[dst[i]], neg[i]]))
            lbl = np.zeros(1 + K, dtype = np.float32)
            lbl[0] = 1.0
            labels_list.append(lbl)
            sizes.append(1 + K)

        return (
                    np.concatenate(cand_src_list),
                    np.concatenate(cand_dst_list),
                    np.concatenate(labels_list),
                    np.array(sizes, dtype = np.int32),
               )


# ---------------------------------------------------------------------------
# Eval-time wrapper (pre-computed negatives)
# ---------------------------------------------------------------------------

class OfficialNegativeSampler:
    """Wraps pre-computed official eval negatives for MRR / AP scoring."""

    def __init__(self, neg_dst: np.ndarray):
        self.neg_dst = neg_dst  # object array [E], each entry a 1-D int64 array

    def build_candidates(
                            self,
                            src:    np.ndarray,
                            dst:    np.ndarray,
                            offset: int = 0,
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (cand_src, cand_dst, labels, sizes).

        Layout per query i: [positive] + [official negatives].
        sizes[i] = 1 + len(neg_dst[offset + i]).
        """
        B = src.shape[0]
        cand_src_list, cand_dst_list, labels_list, sizes = [], [], [], []
        for i in range(B):
            u   = int(src[i])
            v   = int(dst[i])
            neg = np.asarray(self.neg_dst[offset + i], dtype = np.int64)
            K   = neg.shape[0]
            cand_src_list.append(np.full(1 + K, u, dtype = np.int64))
            cand_dst_list.append(np.concatenate([[v], neg]))
            lbl = np.zeros(1 + K, dtype = np.float32)
            lbl[0] = 1.0
            labels_list.append(lbl)
            sizes.append(1 + K)
        return (
                    np.concatenate(cand_src_list),
                    np.concatenate(cand_dst_list),
                    np.concatenate(labels_list),
                    np.array(sizes, dtype = np.int32),
               )


# ---------------------------------------------------------------------------
# Precompute and cache eval negatives
# ---------------------------------------------------------------------------

def _precompute_split_eval_negatives(
                                        root:         str,
                                        name:         str,
                                        split:        str,
                                        *,
                                        num_neg_e:    int   = 100,
                                        strategy:     str   = "hist_rnd",
                                        hist_ratio:   float = 0.5,
                                        seed:         int   = 2020,
                                        exclude_self: bool  = True,
                                    ) -> Path:
    """
    Compute and save {split} eval negatives to the dataset cache.

    Output: {root}/{name}/cache/{split}_official_negatives.npz
    If num_neg_e == -1: produces ragged negatives (dtype=object array),
    containing every valid negative destination for each positive event.
    """
    assert split in ("val", "test")

    # Imported here to break the circular-import cycle.
    # precompute_negatives=False prevents infinite recursion.
    from gsn.datasets.tgb_loader import load_dataset
    (train_split, val_split, test_split), meta = load_dataset(
        name, root=root, cache=True, precompute_negatives=False
    )

    N = int(meta["num_nodes"])

    train_src = train_split.src
    train_dst = train_split.dst
    train_ts  = train_split.ts

    eval_split = val_split if split == "val" else test_split
    pos_src = eval_split.src
    pos_dst = eval_split.dst
    pos_ts  = eval_split.ts

    dst_cands = np.unique(train_split.dst) if meta.get("bipartite", False) else None

    sampler_K = num_neg_e if (isinstance(num_neg_e, int) and num_neg_e > 0) else 1
    sampler = TGBStyleTrainNegativeSampler(
                                            train_src, train_dst, train_ts,
                                            num_nodes = N,
                                            num_neg_e = sampler_K,
                                            strategy = strategy,
                                            hist_ratio = float(hist_ratio),
                                            seed = int(seed),
                                            exclude_self = bool(exclude_self),
                                            dst_candidates = dst_cands
                                          )
    all_dst = sampler.all_dst.astype(np.int64)

    # ------------------------------------------------------------------
    # "All negatives" helpers for num_neg_e == -1 (ragged)
    # ------------------------------------------------------------------

    def _ceil_int(x: float) -> int:
        return int(np.ceil(x - 1e-12))

    def _all_neg_rnd(invalid: np.ndarray) -> np.ndarray:
        return np.setdiff1d(all_dst, invalid, assume_unique = False).astype(np.int64)

    def _all_neg_hist_rnd(u: int, invalid: np.ndarray, h: float) -> np.ndarray:
        h = float(max(0.0, min(1.0, h)))
        seen = sampler.hist_edge_set_per_node.get(int(u), np.empty(0, dtype=np.int64))

        hist_pool = np.setdiff1d(seen, invalid, assume_unique = False).astype(np.int64)
        rnd_inv   = invalid if seen.size == 0 else np.unique(np.concatenate([invalid, seen]))
        rnd_pool  = np.setdiff1d(all_dst, rnd_inv, assume_unique = False).astype(np.int64)

        H, R = int(hist_pool.size), int(rnd_pool.size)
        if H == 0 and R == 0:
            return np.empty(0, dtype = np.int64)
        if h <= 0.0 or H == 0:
            return rnd_pool
        if h >= 1.0 or R == 0:
            return hist_pool
        rnd_need = _ceil_int(H * (1.0 - h) / h)
        if rnd_need <= R:
            rnd_take = sampler._sample_rnd(rnd_inv, rnd_need)
            return np.concatenate([hist_pool, rnd_take]).astype(np.int64)
        hist_need = _ceil_int(R * h / (1.0 - h))
        if hist_need <= H:
            hist_take = sampler._weighted_choice(hist_pool, hist_need, replace = False)
            return np.concatenate([hist_take, rnd_pool]).astype(np.int64)
        neg  = np.concatenate([hist_pool, rnd_pool]).astype(np.int64)
        base = float(H + R)
        if float(H) / base < h:
            extra = _ceil_int((h * base - H) / (1.0 - h))
            if extra > 0:
                neg = np.concatenate([neg, sampler._weighted_choice(hist_pool, extra, replace = True)])
        else:
            extra = _ceil_int(((1.0 - h) * H - h * R) / h)
            if extra > 0:
                neg = np.concatenate([neg, sampler._weighted_choice(rnd_pool, extra, replace = True)])
        return neg.astype(np.int64)

    # ------------------------------------------------------------------
    # Per-positive sampling loop
    # ------------------------------------------------------------------

    Npos    = int(pos_src.shape[0])
    neg_dst = np.empty(Npos, dtype = object)
    sizes   = np.zeros(Npos, dtype = np.int64)

    def _pos_at_t(u: int, t: int) -> np.ndarray:
        dct = sampler.pos_ts_edge_dict.get(int(t), None)
        if dct is None:
            return np.empty(0, dtype = np.int64)
        return dct.get(int(u), np.empty(0, dtype = np.int64))

    for b in range(Npos):
        u = int(pos_src[b])
        v = int(pos_dst[b])
        t = int(pos_ts[b])

        same_ts = _pos_at_t(u, t)
        invalid = np.unique(np.concatenate([same_ts, np.asarray([v], dtype = np.int64)]))
        if exclude_self:
            invalid = np.unique(np.concatenate([invalid, np.asarray([u], dtype = np.int64)]))

        if num_neg_e == -1:
            arr = (
                    _all_neg_hist_rnd(u, invalid, hist_ratio)
                    if strategy == "hist_rnd"
                    else _all_neg_rnd(invalid)
                  )
            arr = np.setdiff1d(arr, invalid, assume_unique = False).astype(np.int64)
        else:
            arr = sampler.sample_fixed_k(u, invalid).astype(np.int64)
            # Top-up if short
            if arr.size < int(num_neg_e):
                pool = np.setdiff1d(
                                        all_dst,
                                        np.unique(np.concatenate([invalid, arr])),
                                        assume_unique = False,
                                    )
                need = int(num_neg_e) - int(arr.size)
                if pool.size > 0 and need > 0:
                    extra = sampler._weighted_choice(pool, need, replace = (need > pool.size))
                    arr   = np.concatenate([arr, extra]).astype(np.int64)
            # Trim if over
            if arr.size > int(num_neg_e):
                arr = sampler.rng.choice(arr, size = int(num_neg_e), replace = False).astype(np.int64)

        neg_dst[b] = arr
        sizes[b]   = int(arr.size)

    cache_dir = _cache_dir(root, name)
    out_path  = cache_dir / f"{split}_official_negatives.npz"
    np.savez(
                out_path,
                neg_dst = neg_dst,
                sizes = sizes,
                src = pos_src,
                dst = pos_dst,
                ts = pos_ts,
                param_num_neg_e    = np.asarray(int(num_neg_e),    dtype = np.int64),
                param_strategy     = np.asarray(str(strategy)),
                param_hist_ratio   = np.asarray(float(hist_ratio), dtype = np.float32),
                param_seed         = np.asarray(int(seed),         dtype = np.int64),
                param_exclude_self = np.asarray(int(exclude_self), dtype = np.int8),
                param_version      = np.asarray("zenodo_gsn_v1")
            )
    return out_path


def load_official_negatives(
                                root:         str,
                                name:         str,
                                split:        str,
                                *,
                                recompute:    bool  = False,
                                num_neg_e:    int   = 100,
                                strategy:     str   = "hist_rnd",
                                hist_ratio:   float = 0.5,
                                seed:         int   = 2020,
                                exclude_self: bool  = True
                           ):
    """
    Load (or precompute) cached eval negatives for a split.

    Returns an NpzFile with keys: neg_dst, sizes, src, dst, ts.
    neg_dst[i] is a 1-D int64 array of negative destinations for event i.
    The cache is automatically invalidated when sampling parameters change.
    """
    assert split in ("val", "test")
    cache_dir  = _cache_dir(root, name)
    split_path = cache_dir / f"{split}_official_negatives.npz"

    def _npz_scalar(z, key):
        if key not in z:
            return None
        v = z[key]
        try:
            return v.item()
        except Exception:
            try:
                a = np.asarray(v)
                if a.size == 1:
                    return a.reshape(()).item()
            except Exception:
                return None
        return None

    def _stale(z) -> bool:
        if _npz_scalar(z, "param_version") is None:
            return True
        try:
            if int(_npz_scalar(z, "param_num_neg_e")) != int(num_neg_e):
                return True
        except Exception:
            return True
        if str(_npz_scalar(z, "param_strategy")) != str(strategy):
            return True
        try:
            if int(_npz_scalar(z, "param_seed")) != int(seed):
                return True
        except Exception:
            return True
        try:
            if int(_npz_scalar(z, "param_exclude_self")) != int(exclude_self):
                return True
        except Exception:
            return True
        if strategy == "hist_rnd":
            try:
                if abs(float(_npz_scalar(z, "param_hist_ratio")) - float(hist_ratio)) > 1e-6:
                    return True
            except Exception:
                return True
        return False

    def _do_compute():
        _precompute_split_eval_negatives(
                                            root, name, split,
                                            num_neg_e = num_neg_e,
                                            strategy = strategy,
                                            hist_ratio = hist_ratio,
                                            seed = seed,
                                            exclude_self = exclude_self,
                                        )

    if recompute:
        try:
            if split_path.exists():
                split_path.unlink()
        except Exception:
            pass
        _do_compute()
        return np.load(split_path, allow_pickle = True)

    if split_path.exists():
        z = np.load(split_path, allow_pickle = True)
        if _stale(z):
            try:
                z.close()
            except Exception:
                pass
            try:
                split_path.unlink()
            except Exception:
                pass
            _do_compute()
            return np.load(split_path, allow_pickle = True)
        return z

    _do_compute()
    return np.load(split_path, allow_pickle = True)
