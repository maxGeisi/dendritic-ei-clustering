"""
A small synthetic neuron for testing the pipeline without MICrONS data.

Y-shaped dendrite sampled every 0.5 µm: a 60 µm trunk along x that splits into
two 40 µm branches. Node IDs are deliberately shuffled, so that they do not
follow the cable. One branch is cut in the SWC file to exercise the healing step.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STEP = 0.5  # µm between neighbouring nodes


def _build_tree():
    coords, parents = [(0.0, 0.0, 0.0)], [-1]
    for k in range(1, 121):                      # trunk
        coords.append((k * STEP, 0.0, 0.0)); parents.append(k - 1)
    fork = len(coords) - 1
    for sign in (+1, -1):                        # two branches
        prev = fork
        for k in range(1, 81):
            d = k * STEP / np.sqrt(2)
            coords.append((60.0 + d, sign * d, 0.0)); parents.append(prev)
            prev = len(coords) - 1
    return np.array(coords), np.array(parents)


@pytest.fixture(scope="session")
def swc_path(tmp_path_factory):
    coords, parents = _build_tree()
    n = len(coords)
    rng = np.random.default_rng(0)
    new_id = rng.permutation(n) + 1              # position -> shuffled node ID
    cut_child = 121 + 40                         # break branch 1 halfway
    rows = []
    for pos in range(n):
        parent = -1 if parents[pos] < 0 or pos == cut_child else new_id[parents[pos]]
        rows.append((new_id[pos], 0, *coords[pos], 0.5, parent))
    rows.sort()
    path = tmp_path_factory.mktemp("swc") / "toy.swc"
    with open(path, "w") as f:
        for r in rows:
            f.write("%d %d %.4f %.4f %.4f %.2f %d\n" % r)
    return path


@pytest.fixture(scope="session")
def neuron(swc_path):
    from src.io_swc import load_and_heal_swc
    return load_and_heal_swc(swc_path)


@pytest.fixture(scope="session")
def synapses():
    """E synapses: a tight group of 5 at x≈20 on the trunk, 4 at x≈45, and 3 isolated ones."""
    xs_e = [19.6, 19.8, 20.0, 20.2, 20.4, 44.7, 44.9, 45.1, 45.3, 5.0, 32.0, 57.0]
    xs_i = [20.1, 45.0, 10.0]
    rows = []
    for k, x in enumerate(xs_e):
        rows.append(dict(id=k, x=x, y=0.05, z=0.0, syn_size=100, pre_identity=1000 + k,
                         pre_type="Unknown", isOnSpine=0))
    for k, x in enumerate(xs_i):
        rows.append(dict(id=100 + k, x=x, y=-0.05, z=0.0, syn_size=100, pre_identity=2000 + k,
                         pre_type="Unknown", isOnSpine=1))
    return pd.DataFrame(rows)
