# src/geo.py
from __future__ import annotations
from pathlib import Path
import hashlib
import os
import tempfile

import joblib
import navis
import numpy as np
import pandas as pd

# ---------- utils ----------

def ensure_dir(p: str | Path) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def _atomic_dump(obj, path: Path):
    """Write to a temp file, then rename, so an interrupted run never leaves a truncated cache."""
    path = Path(path)
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        joblib.dump(obj, tmp_path)
        os.replace(tmp_path, path)  # atomic on POSIX
    finally:
        if tmp_path.exists():
            try: tmp_path.unlink()
            except Exception: pass

def _safe_load_or_none(path: Path):
    """Return the cached object, or None if the file is unreadable."""
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Cache unreadable ({e.__class__.__name__}): {path}. Deleting and recomputing.")
        try: Path(path).unlink()
        except FileNotFoundError:
            pass
        return None

# ---------- skeleton identity ----------

def skeleton_fingerprint(neuron_skel) -> str:
    """
    Short hash of the skeleton's topology and geometry.

    The caches below are keyed on this, not just on the neuron ID. Before, a
    geodesic matrix computed on an older version of the skeleton (e.g. before
    healing) was silently reused for the new one.
    """
    nodes = neuron_skel.nodes.sort_values("node_id")
    h = hashlib.sha1()
    h.update(nodes[["node_id", "parent_id"]].to_numpy(dtype=np.int64).tobytes())
    h.update(np.round(nodes[["x", "y", "z"]].to_numpy(dtype=np.float64), 4).tobytes())
    return h.hexdigest()[:10]

def _matrix_fingerprint(mat) -> str:
    """Fallback identity for a distance matrix when no skeleton fingerprint is passed."""
    arr = mat.values if isinstance(mat, pd.DataFrame) else np.asarray(mat)
    step = max(1, arr.shape[0] // 64)
    h = hashlib.sha1()
    h.update(str(arr.shape).encode())
    if isinstance(mat, pd.DataFrame):
        h.update(np.asarray(mat.index).tobytes())
    h.update(np.ascontiguousarray(arr[::step, ::step]).tobytes())
    return h.hexdigest()[:10]

def count_unreachable_pairs(geodesic_mat) -> int:
    """Number of node pairs with infinite geodesic distance (0 for a connected skeleton)."""
    arr = geodesic_mat.values if isinstance(geodesic_mat, pd.DataFrame) else np.asarray(geodesic_mat)
    return int(np.isinf(arr).sum())

# ---------- cache paths ----------

def geodesic_cache_path(interim_dir: str | Path, neuron_id: str, fingerprint: str) -> Path:
    return Path(interim_dir) / f"geodesic_mat_{neuron_id}_{fingerprint}.pkl"

def kernel_cache_path(interim_dir: str | Path, neuron_id: str, sigma_um: float, fingerprint: str) -> Path:
    return Path(interim_dir) / f"node_weight_mat_{neuron_id}_{fingerprint}_sigma{sigma_um}.pkl"

def _warn_about_legacy_cache(interim_dir: Path, neuron_id: str) -> None:
    """Old cache files had no fingerprint in the name. They are never loaded any more."""
    legacy = [interim_dir / f"geodesic_mat_{neuron_id}.pkl",
              *interim_dir.glob(f"node_weight_mat_{neuron_id}_sigma:*.pkl")]
    legacy = [p for p in legacy if p.exists()]
    if legacy:
        print("Ignoring cache files from before the fingerprint change (safe to delete):")
        for p in legacy:
            print("  ", p)

# ---------- public API ----------

def compute_or_load_geodesic(
    neuron_skel,
    interim_dir: str | Path,
    neuron_id: str,
    force_recompute: bool = False,
):
    """
    Geodesic (along-cable) distance between all pairs of skeleton nodes, as a
    DataFrame indexed by node_id. Cached per neuron *and* skeleton fingerprint.
    """
    interim_dir = ensure_dir(interim_dir)
    fingerprint = skeleton_fingerprint(neuron_skel)
    pkl = geodesic_cache_path(interim_dir, neuron_id, fingerprint)
    _warn_about_legacy_cache(interim_dir, neuron_id)

    g = None
    if (not force_recompute) and pkl.exists():
        print("Found geodesic cache:", pkl, "→ loading")
        g = _safe_load_or_none(pkl)
        # The fingerprint should guarantee this, but a mismatch here would shift
        # every density value, so check anyway.
        if g is not None and isinstance(g, pd.DataFrame):
            if set(g.index) != set(neuron_skel.nodes["node_id"]):
                print("Cached matrix does not match the skeleton's node IDs → recomputing")
                g = None

    if g is None:
        print("Computing geodesic matrix …")
        g = navis.geodesic_matrix(neuron_skel, weight="weight")
        print("Geodesic matrix shape:", g.shape)
        _atomic_dump(g, pkl)
        print("Saved:", pkl)

    n_inf = count_unreachable_pairs(g)
    if n_inf:
        n_trees = getattr(neuron_skel, "n_trees", "?")
        print(
            f"WARNING: {n_inf} node pairs are unreachable (skeleton has {n_trees} "
            "disconnected pieces). Synapses on different pieces cannot share a "
            "cluster and their distances are infinite. Check the healing step."
        )
    return g


def compute_or_load_node_kernel(
    geodesic_mat,
    interim_dir: str | Path,
    neuron_id: str,
    sigma_um: float,
    force_recompute: bool = False,
    fingerprint: str | None = None,
):
    """
    Gaussian kernel on geodesic distance:  W = exp(-d² / 2σ²),  d and σ in µm.
    W_normalized divides by σ·sqrt(2π), so W_normalized @ counts is a density
    in synapses per µm of cable.

    Returns (W, W_normalized). Pass `fingerprint=skeleton_fingerprint(neuron)`
    so the cache is tied to the skeleton; without it the fingerprint is derived
    from the distance matrix itself.
    """
    interim_dir = ensure_dir(interim_dir)
    fingerprint = fingerprint or _matrix_fingerprint(geodesic_mat)
    pkl = kernel_cache_path(interim_dir, neuron_id, sigma_um, fingerprint)

    W_nodes = None
    if (not force_recompute) and pkl.exists():
        print(f"Found kernel cache: {pkl} → loading")
        W_nodes = _safe_load_or_none(pkl)
        if W_nodes is not None and getattr(W_nodes, "shape", None) != geodesic_mat.shape:
            print("Cached kernel has the wrong shape → recomputing")
            W_nodes = None

    if W_nodes is None:
        print("Computing kernel weight matrix …")
        W_nodes = np.exp(-geodesic_mat**2 / (2.0 * sigma_um**2))
        _atomic_dump(W_nodes, pkl)
        print("Saved:", pkl)

    W_nodes_normalized = W_nodes / (sigma_um * np.sqrt(2.0 * np.pi))
    return W_nodes, W_nodes_normalized
