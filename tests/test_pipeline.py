import numpy as np
import pandas as pd

from src.clustering import get_local_maximum, group_by_peak, add_synapse_counts
from src.density import node_density_from_counts, attach_density_to_nodes
from src.geo import (compute_or_load_geodesic, compute_or_load_node_kernel,
                     skeleton_fingerprint, geodesic_cache_path)
from src.prep import split_e_i_by_isOnSpine, ei_label_report
from src.processing.synapse_mapping import map_synapses_to_nodes, create_node_counts_series

SIGMA = 1.0
STEP = 0.5


def _density(neuron, synapses, tmp_path):
    syn_e, _ = split_e_i_by_isOnSpine(synapses)
    syn_e = map_synapses_to_nodes(neuron, syn_e, method="kdtree")
    counts = create_node_counts_series(syn_e, neuron)
    geo = compute_or_load_geodesic(neuron, tmp_path, "toy")
    _, W_norm = compute_or_load_node_kernel(geo, tmp_path, "toy", SIGMA,
                                            fingerprint=skeleton_fingerprint(neuron))
    dens = node_density_from_counts(W_norm, counts)
    nodes = add_synapse_counts(attach_density_to_nodes(neuron, dens), counts)
    return syn_e, counts, geo, nodes


def test_healing_gives_one_connected_tree(neuron, tmp_path):
    assert neuron.n_trees == 1
    geo = compute_or_load_geodesic(neuron, tmp_path, "toy")
    assert not np.isinf(geo.values).any()


def test_both_mapping_methods_return_node_ids(neuron, synapses):
    syn_e, _ = split_e_i_by_isOnSpine(synapses)
    brute = map_synapses_to_nodes(neuron, syn_e, method="brute_force")
    tree = map_synapses_to_nodes(neuron, syn_e, method="kdtree")
    assert (brute["closest_node_id"].values == tree["closest_node_id"].values).all()
    # the mapped node really is the one next to the synapse
    by_id = neuron.nodes.set_index("node_id")
    assert np.allclose(by_id.loc[brute["closest_node_id"], "x"].values, syn_e["Epos3DX"].values, atol=STEP)


def test_no_synapse_is_lost_in_the_counts(neuron, synapses):
    syn_e, _ = split_e_i_by_isOnSpine(synapses)
    syn_e = map_synapses_to_nodes(neuron, syn_e)
    counts = create_node_counts_series(syn_e, neuron)
    assert counts.sum() == len(syn_e)
    assert set(counts.index) == set(neuron.nodes["node_id"])


def test_density_peaks_where_the_synapses_are(neuron, synapses, tmp_path):
    _, _, _, nodes = _density(neuron, synapses, tmp_path)
    peak = nodes.loc[nodes["synapse_density"].idxmax()]
    assert abs(peak["x"] - 20.0) <= STEP and abs(peak["y"]) < 1e-6


def test_density_integrates_to_synapse_count(neuron, synapses, tmp_path):
    # density is in synapses/µm, so density * node spacing summed over the cable
    # must give back the number of synapses (all of them sit far from the fork).
    syn_e, _, _, nodes = _density(neuron, synapses, tmp_path)
    assert np.isclose((nodes["synapse_density"] * STEP).sum(), len(syn_e), rtol=0.02)


def test_gradient_ascent_recovers_the_planted_clusters(neuron, synapses, tmp_path):
    syn_e, _, _, nodes = _density(neuron, synapses, tmp_path)
    start = nodes.loc[nodes["synapse_count"] > 0, "node_id"].tolist()
    peaks = get_local_maximum(neuron, nodes, start)
    cluster_of = syn_e["closest_node_id"].map(peaks)
    x = syn_e["Epos3DX"]
    group_a, group_b = cluster_of[(x > 19) & (x < 21)], cluster_of[(x > 44) & (x < 46)]
    assert group_a.nunique() == 1 and group_b.nunique() == 1
    assert group_a.iloc[0] != group_b.iloc[0]
    sizes = sorted(len(v) for v in group_by_peak(peaks).values())
    assert len(sizes) == 5      # two groups + three isolated synapses


def test_cache_is_tied_to_the_skeleton(neuron, tmp_path):
    fp = skeleton_fingerprint(neuron)
    compute_or_load_geodesic(neuron, tmp_path, "toy")
    assert geodesic_cache_path(tmp_path, "toy", fp).exists()

    moved = neuron.copy()
    moved.nodes.loc[0, "x"] += 1.0
    assert skeleton_fingerprint(moved) != fp

    # a cache file from before the fingerprint change must not be picked up
    pd.DataFrame([[0.0]]).to_pickle(tmp_path / "geodesic_mat_toy.pkl")
    geo = compute_or_load_geodesic(neuron, tmp_path, "toy")
    assert geo.shape[0] == neuron.n_nodes


def test_known_presynaptic_type_overrides_spine_flag(synapses):
    df = synapses.copy()
    df.loc[0, ["pre_type", "pre_identity"]] = ["DTC", 9999]   # interneuron axon, flagged as E
    df.loc[12, "pre_identity"] = 9999                         # same axon, flagged as I

    e_thesis, i_thesis = split_e_i_by_isOnSpine(df, pre_type_override=False)
    e_fixed, i_fixed = split_e_i_by_isOnSpine(df)
    assert 0 in e_thesis["id"].values and 0 in i_fixed["id"].values
    assert len(e_fixed) == len(e_thesis) - 1 and len(i_fixed) == len(i_thesis) + 1

    report = ei_label_report(df)
    assert report["n_relabelled_by_pre_type"] == 1
    assert report["n_axons_with_mixed_flag_labels"] == 1
