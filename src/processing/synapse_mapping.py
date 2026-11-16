# src/processing/synapse_mapping.py
"""
Synapse-to-node mapping functions for dendric clustering analysis.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from typing import Tuple, Dict, Any

def map_synapses_to_nodes(
    neuron_skel,
    syn_df: pd.DataFrame,
    coord_cols: Tuple[str, str, str] = ("Epos3DX", "Epos3DY", "Epos3DZ"),
    method: str = "brute_force"
) -> pd.DataFrame:
    """
    Map synapses to their closest skeleton nodes.
    
    Args:
        neuron_skel: Navis neuron skeleton object
        syn_df: DataFrame with synapse coordinates
        coord_cols: Tuple of coordinate column names (x, y, z)
        method: "brute_force" or "kdtree" for nearest neighbor search
        
    Returns:
        DataFrame with added 'closest_node_id' and 'distance_to_node' columns
    """
    syn_df = syn_df.copy()
    
    # Get skeleton coordinates
    skeleton_nodes = neuron_skel.nodes
    skeleton_coords = skeleton_nodes[['x', 'y', 'z']].values
    
    # Get synapse coordinates
    synapse_coords = syn_df[list(coord_cols)].values
    
    node_ids = skeleton_nodes['node_id'].to_numpy()

    if method == "kdtree":
        tree = cKDTree(skeleton_coords)
        distances, indices = tree.query(synapse_coords)
    else:
        # Brute force (original implementation). Same result as the KD-tree, O(N*M).
        indices = np.empty(len(synapse_coords), dtype=int)
        distances = np.empty(len(synapse_coords), dtype=float)
        for k, syn_xyz in enumerate(synapse_coords):
            dists = np.sqrt(((skeleton_coords - syn_xyz) ** 2).sum(axis=1))
            indices[k] = np.argmin(dists)
            distances[k] = dists[indices[k]]

    # `indices` are row positions in the nodes table, not node IDs.
    closest_node_ids = node_ids[indices]

    syn_df['closest_node_id'] = closest_node_ids
    syn_df['distance_to_node'] = distances
    
    return syn_df

def compute_synapse_node_statistics(syn_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute statistics about synapse-to-node mapping.
    
    Args:
        syn_df: DataFrame with 'closest_node_id' column
        
    Returns:
        Dictionary with mapping statistics
    """
    node_counts = syn_df['closest_node_id'].value_counts()
    
    stats = {
        'n_unique_nodes_with_synapses': len(node_counts),
        'max_synapses_per_node': node_counts.max(),
        'max_synapses_node_id': node_counts.idxmax(),
        'avg_synapses_per_node': node_counts.mean(),
        'median_synapses_per_node': node_counts.median(),
        'total_synapses': len(syn_df)
    }
    
    return stats

def create_node_counts_series(
    syn_df: pd.DataFrame, 
    neuron_skel,
    node_id_col: str = 'closest_node_id'
) -> pd.Series:
    """
    Create a Series of synapse counts per node, aligned with neuron skeleton.
    
    Args:
        syn_df: DataFrame with synapse data
        neuron_skel: Navis neuron skeleton object
        node_id_col: Column name containing node IDs
        
    Returns:
        Series indexed by node_id with synapse counts (0 for nodes without synapses)
    """
    node_counts = syn_df[node_id_col].value_counts()
    # Index by node_id (not by the row index of the nodes table), zeros elsewhere
    node_counts = node_counts.reindex(neuron_skel.nodes['node_id'], fill_value=0)
    node_counts.index.name = 'node_id'
    return node_counts
