# src/density.py
from __future__ import annotations
import pandas as pd

def node_density_from_counts(W_nodes_normalized: pd.DataFrame,
                             node_counts: pd.Series) -> pd.Series:
    """
    Kernel density estimate along the dendrite, in synapses per µm:
        density(i) = sum_j W_norm[i, j] * counts[j]
    Both W_norm and node_counts are indexed by node_id.
    """
    counts = node_counts.reindex(W_nodes_normalized.index, fill_value=0)
    dens = W_nodes_normalized.dot(counts)
    return dens

def print_density_statistics(node_density: pd.Series) -> None:
    print(f"Maximum density: {node_density.max():.3f}")
    print(f"Average density: {node_density.mean():.3f}") 
    print(f"Minimum density: {node_density.min():.3f}")

def attach_density_to_nodes(neuron_skel, node_density: pd.Series) -> pd.DataFrame:
    """
    Return a copy of neuron_skel.nodes with a 'synapse_density' column.

    node_density is indexed by node_id (it comes from the geodesic matrix), while
    the nodes table has a plain 0..N-1 row index. The two must be joined on
    node_id: joining on the row index assigns every node the density of the
    node with the next-lower ID.
    """
    nodes = neuron_skel.nodes.copy()
    nodes["node_id"] = nodes["node_id"].astype(int)
    nodes["synapse_density"] = nodes["node_id"].map(node_density)
    n_missing = int(nodes["synapse_density"].isna().sum())
    if n_missing:
        print(f"WARNING: {n_missing} nodes have no density value; filling with the mean.")
        nodes["synapse_density"] = nodes["synapse_density"].fillna(nodes["synapse_density"].mean())
    return nodes
