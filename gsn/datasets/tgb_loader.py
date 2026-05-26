"""Unified Zenodo/DGB dataset loader for GSN temporal link prediction.

Replaces the old TGB (py-tgb) dependency entirely.
Downloads all datasets from https://zenodo.org/records/7213796.

Usage
-----
    (train, val, test), meta = load_dataset("tgbl-wiki", root="data/")
    (train, val, test), meta = load_tgb("tgbl-wiki", root="data/")  # alias

    # With pre-computed official eval negatives:
    (train, val, test), meta = load_dataset(
        "tgbl-wiki", root="data/",
        precompute_negatives=True, num_neg_e=100,
    )
"""
from __future__ import annotations

import csv
import json
import os
import time
import zipfile

import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

ZENODO_RECORD = "7213796"


def _zenodo_urls(zip_name: str) -> List[str]:
    """Return candidate download URLs in preference order.

    The REST-API content endpoint (/api/records/…/files/…/content) routes
    through Zenodo's application servers rather than their Cloudflare CDN,
    so it is less likely to be blocked on datacenter IPs.
    """
    rid = ZENODO_RECORD
    return [
        f"https://zenodo.org/api/records/{rid}/files/{zip_name}/content",
        f"https://zenodo.org/record/{rid}/files/{zip_name}",
        f"https://zenodo.org/records/{rid}/files/{zip_name}?download=1",
    ]

_REGISTRY: Dict[str, Dict[str, Any]] = {
    "tgbl-wiki": {
        "zip_name":  "wikipedia.zip",
        "csv_names": ["ml_wikipedia.csv", "wikipedia.csv",
                      "ml_Wikipedia.csv", "Wikipedia.csv"],
        "npy_names": ["ml_wikipedia.npy", "ml_Wikipedia.npy"],
        "bipartite": True,
    },
    "tgbl-mooc": {
        "zip_name":  "mooc.zip",
        "csv_names": ["ml_mooc.csv", "mooc.csv",
                      "ml_MOOC.csv", "MOOC.csv"],
        "npy_names": ["ml_mooc.npy", "ml_MOOC.npy"],
        "bipartite": True,
    },
    "tgbl-enron": {
        "zip_name":  "enron.zip",
        "csv_names": ["ml_enron.csv", "enron.csv",
                      "ml_Enron.csv", "Enron.csv"],
        "npy_names": ["ml_enron.npy", "ml_Enron.npy"],
        "bipartite": False,
    },
    "tgbl-uci": {
        "zip_name":  "uci.zip",
        "csv_names": ["ml_uci.csv", "uci.csv",
                      "ml_UCI.csv", "UCI.csv"],
        "npy_names": ["ml_uci.npy", "ml_UCI.npy"],
        "bipartite": False,
    },
    "tgbl-uslegis": {
        "zip_name":  "USLegis.zip",
        "csv_names": ["ml_USLegis.csv", "USLegis.csv",
                      "ml_uslegis.csv", "uslegis.csv"],
        "npy_names": ["ml_USLegis.npy", "ml_uslegis.npy"],
        "bipartite": False,
    },
    "tgbl-canparl": {
        "zip_name":  "CanParl.zip",
        "csv_names": ["ml_CanParl.csv", "CanParl.csv",
                      "ml_canparl.csv", "canparl.csv"],
        "npy_names": ["ml_CanParl.npy", "ml_canparl.npy"],
        "bipartite": False,
    },
    "tgbl-contacts": {
        "zip_name":  "Contacts.zip",
        "csv_names": ["ml_Contacts.csv", "Contacts.csv",
                      "ml_contacts.csv", "contacts.csv"],
        "npy_names": ["ml_Contacts.npy", "ml_contacts.npy"],
        "bipartite": False,
    },
}

SUPPORTED = set(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Public data container
# ---------------------------------------------------------------------------

@dataclass
class TGBSplit:
    """One chronological split of a temporal link-prediction dataset."""

    src:       np.ndarray            # [E] int64
    dst:       np.ndarray            # [E] int64
    ts:        np.ndarray            # [E] int64  (shifted to start at 0)
    edge_feat: Optional[np.ndarray]  # [E, F_e] float32 or None
    num_nodes: int
    neg_dst:   Optional[np.ndarray] = None  # object[E] of 1-D int64 arrays (val/test only)

    def __len__(self) -> int:
        return int(self.src.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.src.shape[0])

    def iter_events(self):
        ef = self.edge_feat
        nd = self.neg_dst
        for i in range(len(self)):
            yield (
                int(self.src[i]),
                int(self.dst[i]),
                int(self.ts[i]),
                (ef[i] if ef is not None else None),
                (nd[i] if nd is not None else None),
            )


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_dir(root: str, name: str) -> Path:
    p = Path(root) / name / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_exists(root: str, name: str) -> bool:
    d = _cache_dir(root, name)
    return (d / "full_data.npz").exists() and (d / "meta.json").exists()


def _save_cache(root: str, name: str,
                full: Dict[str, np.ndarray],
                meta: Dict[str, Any]) -> None:
    """Cache raw data and meta only.  Masks are always recomputed — they are
    cheap (pure numpy) and depend on split_by / split_ratios which can change
    between calls without requiring a re-download."""
    d = _cache_dir(root, name)
    np.savez(d / "full_data.npz", **full)
    with open(d / "meta.json", "w", encoding = "utf-8") as f:
        json.dump(meta, f, indent = 4)


def _load_cache(root: str, name: str) -> Optional[Tuple[Dict, Dict]]:
    """Returns (full, meta).  Masks are not cached; always recomputed."""
    d      = _cache_dir(root, name)
    p_full = d / "full_data.npz"
    p_meta = d / "meta.json"
    if not (p_full.exists() and p_meta.exists()):
        return None
    full = {k: np.asarray(v) for k, v in np.load(p_full, allow_pickle = False).items()}
    with open(p_meta, encoding = "utf-8") as f:
        meta = json.load(f)
    return full, meta


# ---------------------------------------------------------------------------
# Download / extraction utilities
# ---------------------------------------------------------------------------

def _ensure_dir(path) -> None:
    os.makedirs(str(path), exist_ok = True)


def _atomic_write(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    _ensure_dir(os.path.dirname(path))
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer":         f"https://zenodo.org/records/{ZENODO_RECORD}",
    "Connection":      "keep-alive",
}


def _download(
        urls:      "str | List[str]",
        dst_path:  str,
        *,
        retries:   int   = 4,
        backoff_s: float = 2.0,
) -> None:
    """Download to dst_path, trying each URL in *urls* until one succeeds."""
    if isinstance(urls, str):
        urls = [urls]
    _ensure_dir(os.path.dirname(dst_path))
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        return
    tmp_path = dst_path + ".tmp"
    last_err: Optional[Exception] = None

    with requests.Session() as session:
        session.headers.update(_BROWSER_HEADERS)
        # Visit the record page so Zenodo can set its session cookie.
        try:
            session.get(
                f"https://zenodo.org/records/{ZENODO_RECORD}",
                timeout = (10, 30),
            )
        except Exception:
            pass

        for url in urls:
            for attempt in range(retries):
                try:
                    with session.get(
                        url,
                        stream          = True,
                        timeout         = (15, 300),
                        allow_redirects = True,
                    ) as r:
                        r.raise_for_status()
                        with open(tmp_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size = 1 << 20):
                                f.write(chunk)
                    if os.path.getsize(tmp_path) == 0:
                        raise RuntimeError("Empty response body")
                    os.replace(tmp_path, dst_path)
                    return
                except Exception as exc:
                    last_err = exc
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    time.sleep(backoff_s * (attempt + 1))

    tried = "\n  ".join(urls)
    raise RuntimeError(
        f"Download failed after trying {len(urls)} URL(s):\n  {tried}"
    ) from last_err


def _extract_zip(zip_path: str, out_dir: str) -> None:
    _ensure_dir(out_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def _find_first(root_dir: str, names: List[str]) -> Optional[str]:
    want = {n.lower() for n in names}
    for dp, _, fns in os.walk(root_dir):
        for fn in fns:
            if fn.lower() in want:
                return os.path.join(dp, fn)
    return None


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def _parse_csv(
                csv_path: str,
                edge_npy: Optional[str]
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[List[float]]]:
    
    """
    Parse DGB-format CSV. Returns (src_raw, dst_raw, ts_f, y, edge_idx, extra_feat).
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        first  = next(reader)

        has_header = any(
            x.strip().lower() in ("u", "i", "src", "dst", "ts", "timestamp", "idx", "label")
            for x in first
        )
        if has_header:
            header = [h.strip() for h in first]
        else:
            header = ["u", "i", "ts", "label", "idx"]
            f.seek(0)
            reader = csv.reader(f)

        def _col(*cands: str) -> Optional[int]:
            for c in cands:
                if c in header:
                    return header.index(c)
            return None

        iu     = _col("u", "src", "source")
        iv     = _col("i", "dst", "destination")
        its    = _col("ts", "timestamp", "time")
        ilabel = _col("label", "edge_label")
        iidx   = _col("idx", "edge_idx", "edge_idxs")

        if iu is None or iv is None or its is None:
            raise RuntimeError(f"Unrecognized CSV columns in {csv_path}: {header}")

        used_cols = {iu, iv, its}
        if ilabel is not None:
            used_cols.add(ilabel)
        if iidx is not None:
            used_cols.add(iidx)

        src_raw:    List[int]         = []
        dst_raw:    List[int]         = []
        ts_raw:     List[float]       = []
        y_raw:      List[float]       = []
        idx_raw:    List[int]         = []
        extra_feat: List[List[float]] = []

        for row in reader:
            if not row:
                continue
            src_raw.append(int(float(row[iu])))
            dst_raw.append(int(float(row[iv])))
            ts_raw.append(float(row[its]))
            y_raw.append(float(row[ilabel]) if ilabel is not None else 1.0)
            idx_raw.append(int(float(row[iidx])) if iidx is not None else len(idx_raw))
            if edge_npy is None:
                feats = []
                for j in range(len(row)):
                    if j in used_cols:
                        continue
                    try:
                        feats.append(float(row[j]))
                    except Exception:
                        feats.append(0.0)
                extra_feat.append(feats if feats else [0.0])

    return (
                np.asarray(src_raw,  dtype = np.int64),
                np.asarray(dst_raw,  dtype = np.int64),
                np.asarray(ts_raw,   dtype = np.float64),
                np.asarray(y_raw,    dtype = np.float32),
                np.asarray(idx_raw,  dtype = np.int64),
                extra_feat
            )


# ---------------------------------------------------------------------------
# Timestamp-boundary split
# ---------------------------------------------------------------------------

def _split_masks_by_timestamp(
                                    ts: np.ndarray,
                                    split_ratios: Tuple[float, float, float],
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Time-contiguous split that respects timestamp boundaries.

    All events at the same timestamp land in the same split.
    Falls back to edge-index splitting when fewer than 3 unique timestamps exist.
    """
    ts = np.asarray(ts, dtype = np.int64)
    n  = int(ts.shape[0])
    if n == 0:
        z = np.zeros(0, dtype = bool)
        return z, z, z

    r_tr, r_va, r_te = map(float, split_ratios)
    if abs((r_tr + r_va + r_te) - 1.0) > 1e-6:
        raise ValueError(f"split_ratios must sum to 1.0, got {split_ratios}")

    uniq, counts = np.unique(ts, return_counts = True)
    U = int(uniq.size)

    def _edge_index_split():
        order = np.argsort(ts, kind = "stable")
        n_tr  = int(np.floor(r_tr * n))
        n_va  = int(np.floor(r_va * n))
        tr = np.zeros(n, dtype = bool); tr[order[:n_tr]] = True
        va = np.zeros(n, dtype = bool); va[order[n_tr:n_tr + n_va]] = True
        te = np.zeros(n, dtype = bool); te[order[n_tr + n_va:]] = True
        return tr, va, te

    if U < 3:
        return _edge_index_split()

    cum  = np.cumsum(counts)
    i_tr = int(np.searchsorted(cum, r_tr * n,          side = "left"))
    i_va = int(np.searchsorted(cum, (r_tr + r_va) * n, side = "left"))

    i_tr = max(0, min(i_tr, U - 3))
    i_va = max(i_tr + 1, min(i_va, U - 2))

    t_tr_end = int(uniq[i_tr])
    t_va_end = int(uniq[i_va])

    tr = (ts <= t_tr_end)
    va = (ts > t_tr_end) & (ts <= t_va_end)
    te = (ts > t_va_end)

    if not (tr.any() and va.any() and te.any()):
        return _edge_index_split()

    return tr.astype(bool), va.astype(bool), te.astype(bool)


# ---------------------------------------------------------------------------
# Event-count split  (new default)
# ---------------------------------------------------------------------------

def _split_masks_by_event_count(
    ts:           np.ndarray,
    split_ratios: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological split where each partition receives approximately the
    requested *fraction of events* (not of the time range).

    Boundaries are snapped to timestamp edges so that all events sharing a
    timestamp land in the same split.  Among the two candidate timestamps that
    straddle each target count, the one whose cumulative count is nearest to
    the target is chosen (nearest-neighbour rounding), preventing a single
    dense timestamp cluster from skewing the split.

    Falls back to a pure event-index split when fewer than 3 unique timestamps
    exist.
    """
    ts = np.asarray(ts, dtype = np.int64)
    n  = int(ts.shape[0])
    if n == 0:
        z = np.zeros(0, dtype = bool)
        return z, z, z

    r_tr, r_va, r_te = map(float, split_ratios)
    if abs(r_tr + r_va + r_te - 1.0) > 1e-6:
        raise ValueError(f"split_ratios must sum to 1.0, got {split_ratios}")

    uniq, counts = np.unique(ts, return_counts = True)
    U   = int(uniq.size)
    cum = np.cumsum(counts)   # cum[i] = #events with ts <= uniq[i]

    def _event_index_split():
        order = np.argsort(ts, kind = "stable")
        n_tr  = max(1, int(round(r_tr * n)))
        n_va  = max(1, int(round(r_va * n)))
        n_tr  = min(n_tr, n - 2)
        n_va  = min(n_va, n - n_tr - 1)
        tr = np.zeros(n, bool); tr[order[:n_tr]] = True
        va = np.zeros(n, bool); va[order[n_tr : n_tr + n_va]] = True
        te = np.zeros(n, bool); te[order[n_tr + n_va:]] = True
        return tr, va, te

    if U < 3:
        return _event_index_split()

    def _nearest_idx(target: float, lo: int, hi: int) -> int:
        """Index i in [lo, hi] s.t. cum[i] is closest to target."""
        lo, hi = max(0, lo), min(U - 1, hi)
        if lo >= hi:
            return lo
        sub = cum[lo : hi + 1]
        pos = int(np.searchsorted(sub, target, side = "left"))
        # Two boundary candidates: the timestamp just below and the one at pos
        cands = [lo + p for p in (pos - 1, pos) if 0 <= lo + p <= hi]
        return min(cands, key = lambda c: abs(cum[c] - target))

    i_tr = _nearest_idx(r_tr * n,          lo = 0,       hi = U - 3)
    i_va = _nearest_idx((r_tr + r_va) * n, lo = i_tr + 1, hi = U - 2)

    t_tr_end = int(uniq[i_tr])
    t_va_end = int(uniq[i_va])

    tr = (ts <= t_tr_end)
    va = (ts > t_tr_end) & (ts <= t_va_end)
    te = (ts > t_va_end)

    if not (tr.any() and va.any() and te.any()):
        return _event_index_split()

    return tr.astype(bool), va.astype(bool), te.astype(bool)


# ---------------------------------------------------------------------------
# Zenodo DGB loader (generic)
# ---------------------------------------------------------------------------

def _fix_len(x: Optional[np.ndarray], E: int, fill_value: float = 0.0) -> Optional[np.ndarray]:
    if x is None:
        return None
    x = np.asarray(x)
    if x.shape[0] == E:
        return x
    if x.shape[0] > E:
        return x[:E]
    pad = np.full((E - x.shape[0],) + x.shape[1:], fill_value, dtype = x.dtype)
    return np.concatenate([x, pad], axis = 0)


def _load_from_zenodo(name: str, root: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Download and parse one DGB dataset from Zenodo. Returns (full_np, meta).
    """
    info      = _REGISTRY[name]
    zip_name  = info["zip_name"]
    csv_names = info["csv_names"]
    npy_names = info["npy_names"]
    bipartite = info["bipartite"]

    stem = zip_name.replace(".zip", "")

    dgb_dir     = Path(root) / name / "cache" / f"dgb_{stem}"
    zip_path    = str(dgb_dir / zip_name)
    extract_dir = str(dgb_dir / stem)

    print(f"  [zenodo] Downloading {zip_name} ...")
    _download(_zenodo_urls(zip_name), zip_path)

    if not os.path.isdir(extract_dir) or not any(os.scandir(extract_dir)):
        print(f"  [zenodo] Extracting {zip_name} ...")
        _extract_zip(zip_path, extract_dir)

    csv_path = _find_first(extract_dir, csv_names)
    if not csv_path:
        raise RuntimeError(f"Could not find any of {csv_names} inside {extract_dir}")

    edge_npy = _find_first(extract_dir, npy_names)

    print(f"  [zenodo] Parsing {os.path.basename(csv_path)} ...")
    src_raw, dst_raw, ts_f, y, edge_idx, extra_feat = _parse_csv(csv_path, edge_npy)

    # Remap nodes to contiguous [0, num_nodes)
    uniq  = np.sort(np.unique(np.concatenate([src_raw, dst_raw])))
    remap = {int(old): int(new) for new, old in enumerate(uniq)}
    src   = np.fromiter((remap[int(x)] for x in src_raw), dtype = np.int64, count = src_raw.size)
    dst   = np.fromiter((remap[int(x)] for x in dst_raw), dtype = np.int64, count = dst_raw.size)
    num_nodes = int(uniq.size)

    # Shift timestamps to start at 0
    ts_int     = ts_f.astype(np.int64)
    ts_shifted = (ts_int - int(ts_int.min())) if ts_int.size else ts_int
    E = int(src.shape[0])

    # Edge features / weights from NPY (col 0 = weight, 1+ = features) or extra CSV columns
    edge_feat_mat: Optional[np.ndarray] = None
    if edge_npy is not None:
        edge_feat_mat = np.load(edge_npy)
    elif extra_feat:
        edge_feat_mat = np.asarray(extra_feat, dtype = np.float32)

    edge_feat: Optional[np.ndarray] = None
    w:         Optional[np.ndarray] = None

    if edge_feat_mat is not None:
        ef = _fix_len(np.asarray(edge_feat_mat), E, fill_value = 0.0)
        if ef is not None:
            if ef.ndim == 1:
                w = ef.astype(np.float32)
            elif ef.ndim == 2:
                if ef.shape[1] >= 1:
                    w = ef[:, 0].astype(np.float32)
                if ef.shape[1] >= 2:
                    edge_feat = ef[:, 1:].astype(np.float32)

    if w is None:
        w = np.ones(E, dtype = np.float32)
    else:
        w = _fix_len(w, E, fill_value = 1.0)
        # Guard: if every weight is zero the NPY weight column was a dummy
        # placeholder (DGB convention for feature-less datasets).  Fall back
        # to uniform unit weights so downstream code never sees w = 0.
        if np.all(w == 0):
            w = np.ones(E, dtype = np.float32)

    if edge_feat is not None:
        edge_feat = _fix_len(edge_feat, E, fill_value = 0.0)

    full_np: Dict[str, np.ndarray] = {
                                        "sources":      src,
                                        "destinations": dst,
                                        "timestamps":   ts_shifted.astype(np.int64),
                                        "edge_idxs":    edge_idx,
                                        "edge_label":   y,
                                        "w":            w.astype(np.float32),
                                        "idmap_ids":    uniq.astype(np.int64)
                                     }
    if edge_feat is not None:
        full_np["edge_feat"] = edge_feat.astype(np.float32)
    if bipartite:
        full_np["src_candidates"] = np.unique(src).astype(np.int64)
        full_np["dst_candidates"] = np.unique(dst).astype(np.int64)

    meta: Dict[str, Any] = {
                                "name":          name,
                                "source":        f"zenodo_dgb_{ZENODO_RECORD}",
                                "zenodo_url":    _zenodo_urls(zip_name)[0],
                                "num_nodes":     num_nodes,
                                "num_edges":     E,
                                "bipartite":     bipartite,
                                "eval_metric":   "MRR",
                                "has_edge_feat": bool(edge_feat is not None),
                                "edge_feat_dim": int(edge_feat.shape[1]) if edge_feat is not None else 0
                            }
    return full_np, meta


# ---------------------------------------------------------------------------
# Split materializer
# ---------------------------------------------------------------------------

def _make_split(
    full:      Dict[str, np.ndarray],
    mask:      np.ndarray,
    num_nodes: int,
) -> TGBSplit:
    m  = mask.astype(bool)
    ef = full["edge_feat"][m].astype(np.float32) if "edge_feat" in full else None
    return TGBSplit(
                        src = full["sources"][m].astype(np.int64),
                        dst = full["destinations"][m].astype(np.int64),
                        ts = full["timestamps"][m].astype(np.int64),
                        edge_feat = ef,
                        num_nodes = int(num_nodes),
                        neg_dst = None
                    )


def merge_splits(a: TGBSplit, b: TGBSplit) -> TGBSplit:
    """Concatenate two chronologically-contiguous splits into one.

    Intended for the "train on train+val" regime (a.k.a. retrain-with-frozen-
    hyperparameters): after all hyperparameters have been selected using the
    train→val signal, build a merged training set by appending val to train,
    then retrain from scratch and evaluate ONCE on test.

    Requirements
    ------------
    * ``a`` and ``b`` must share ``num_nodes``.
    * ``b`` must be chronologically AFTER ``a`` (``a.ts[-1] <= b.ts[0]``).
    * ``edge_feat`` presence must agree (both None, or both present with same
      feature dimension).

    Notes
    -----
    * ``neg_dst`` is dropped on the result: pre-computed negatives belong to
      a held-out evaluation split, which ``b`` no longer is.
    """
    if a.num_nodes != b.num_nodes:
        raise ValueError(
            f"merge_splits: num_nodes mismatch (a={a.num_nodes}, b={b.num_nodes})"
        )
    if len(a) > 0 and len(b) > 0 and int(a.ts[-1]) > int(b.ts[0]):
        raise ValueError(
            f"merge_splits expects chronologically-ordered inputs; got "
            f"a.ts[-1]={int(a.ts[-1])} > b.ts[0]={int(b.ts[0])}"
        )
    if (a.edge_feat is None) != (b.edge_feat is None):
        raise ValueError(
            "merge_splits: edge_feat presence mismatch between splits"
        )
    if a.edge_feat is not None and b.edge_feat is not None:
        if a.edge_feat.shape[1] != b.edge_feat.shape[1]:
            raise ValueError(
                f"merge_splits: edge_feat dim mismatch "
                f"({a.edge_feat.shape[1]} vs {b.edge_feat.shape[1]})"
            )
        ef = np.concatenate([a.edge_feat, b.edge_feat], axis = 0).astype(np.float32)
    else:
        ef = None

    return TGBSplit(
                        src       = np.concatenate([a.src, b.src]).astype(np.int64),
                        dst       = np.concatenate([a.dst, b.dst]).astype(np.int64),
                        ts        = np.concatenate([a.ts,  b.ts ]).astype(np.int64),
                        edge_feat = ef,
                        num_nodes = int(a.num_nodes),
                        neg_dst   = None,
                    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(
                    name:                 str,
                    root:                 str   = "data",
                    split_ratios:         Tuple[float, float, float] = (0.7, 0.15, 0.15),
                    cache:                bool  = True,
                    seed:                 int   = 2020,
                    force_rebuild:        bool  = False,
                    precompute_negatives: bool  = False,
                    num_neg_e:            int   = 100,
                    neg_strategy:         str   = "hist_rnd",
                    neg_hist_ratio:       float = 0.5,
                    split_by:             str   = "events",
                ) -> Tuple[Tuple[TGBSplit, TGBSplit, TGBSplit], Dict[str, Any]]:

    """
    Load a temporal link-prediction dataset from Zenodo (DGB format).

    Parameters
    ----------
    name          : dataset name — one of tgbl-wiki, tgbl-mooc, tgbl-enron,
                    tgbl-uci, tgbl-uslegis, tgbl-canparl, tgbl-contacts
    root          : local directory for storing raw ZIPs and cache
    split_ratios  : (train, val, test) fractions; must sum to 1.0
    cache         : save/load numpy cache under {root}/{name}/cache/
    seed          : RNG seed
    force_rebuild : ignore existing cache and re-download/re-parse
    precompute_negatives : compute official eval negatives and attach
                           to val.neg_dst / test.neg_dst
    num_neg_e     : negatives per positive for eval (−1 = ragged "all")
    neg_strategy  : "hist_rnd" (default) or "rnd"
    neg_hist_ratio : fraction drawn from historical neighbours (hist_rnd)
    split_by      : how to place chronological split boundaries.
                    "events" (default) — each split receives approximately the
                      requested fraction of *events*, snapped to the nearest
                      timestamp boundary.  Val and test will have similar event
                      counts even when event density varies over time.
                    "time" — legacy behaviour: boundaries are placed at the
                      first timestamp whose cumulative event count first reaches
                      the target fraction, always rounding up at timestamp
                      clusters.  Can produce unequal val/test sizes on datasets
                      with bursty event distributions.

    Returns
    -------
    (train, val, test), meta
    """
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}. Supported: {sorted(SUPPORTED)}")
    if split_by not in ("events", "time"):
        raise ValueError(f"split_by must be 'events' or 'time', got {split_by!r}")

    if cache and not force_rebuild and _cache_exists(root, name):
        cached = _load_cache(root, name)
        full, meta = cached
    else:
        full, meta = _load_from_zenodo(name, root)
        ts = full["timestamps"]
        uniq_ts, counts = np.unique(ts, return_counts = True)
        meta.update({
                        "seed":              int(seed),
                        "max_dt":            int(np.diff(uniq_ts).max()) if uniq_ts.size > 1 else 0,
                        "events_per_ts_min": int(counts.min()),
                        "events_per_ts_max": int(counts.max()),
                        "events_per_ts_avg": float(counts.mean()),
                        "events_per_ts_std": float(counts.std()),
                    })
        if cache:
            _save_cache(root, name, full, meta)

    # Masks are always recomputed — they are cheap and depend on split_by /
    # split_ratios which can differ between calls without needing a re-download.
    ts = full["timestamps"]
    if split_by == "events":
        train_mask, val_mask, test_mask = _split_masks_by_event_count(ts, split_ratios)
    else:
        train_mask, val_mask, test_mask = _split_masks_by_timestamp(ts, split_ratios)

    meta["split_ratios"] = list(map(float, split_ratios))
    meta["split_by"]     = split_by

    N     = int(meta["num_nodes"])
    train = _make_split(full, train_mask, N)
    val   = _make_split(full, val_mask,   N)
    test  = _make_split(full, test_mask,  N)

    if precompute_negatives:
        from gsn.datasets.negative_sampling import load_official_negatives
        for split_name, split in (("val", val), ("test", test)):
            z = load_official_negatives(
                                            root, name, split_name,
                                            num_neg_e = num_neg_e,
                                            strategy = neg_strategy,
                                            hist_ratio = neg_hist_ratio,
                                            seed = seed,
                                        )
            split.neg_dst = z["neg_dst"]

    return (train, val, test), meta


def load_tgb(
                name:  str,
                root:  str  = "data",
                cache: bool = True,
            ) -> Tuple[Tuple[TGBSplit, TGBSplit, TGBSplit], Dict[str, Any]]:
    """
    Backward-compatible alias for load_dataset.
    """
    return load_dataset(name, root = root, cache = cache)
