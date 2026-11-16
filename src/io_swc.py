# src/io_swc.py
from pathlib import Path
import navis

def _project_root() -> Path:
    here = Path.cwd()
    for p in (here, *here.parents):
        if (p / "src").exists() and (p / "config").exists():
            return p
    return Path.cwd()

def build_swc_path(swc_dir: str | Path, prefix: str, neuron_id: str) -> Path:
    swc_dir = Path(swc_dir)
    if not swc_dir.is_absolute():
        swc_dir = _project_root() / swc_dir
    swc_dir.mkdir(parents=True, exist_ok=True)

    p = swc_dir / f"{prefix}{neuron_id}.swc"
    if not p.exists():
        raise FileNotFoundError(f"SWC not found: {p}")
    return p


def load_and_heal_swc(swc_path: str | Path):
    """
    Read an SWC file and reconnect its fragments into a single tree.

    navis may return a NeuronList; in that case the first neuron is used.
    Warns if the skeleton is still in several pieces after healing, because
    geodesic distances between pieces are infinite.
    """
    n = navis.read_swc(str(swc_path))
    if hasattr(n, "__iter__") and not hasattr(n, "nodes"):
        n = n[0]
    n_before = n.n_trees
    n = navis.heal_skeleton(n)
    if n.n_trees > 1:
        print(f"WARNING: skeleton still has {n.n_trees} disconnected pieces after healing "
              f"(was {n_before}).")
    elif n_before > 1:
        print(f"Healed skeleton: {n_before} fragments → 1 tree")
    return n
