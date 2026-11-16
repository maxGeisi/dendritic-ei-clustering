# Dendritic E/I clustering on MICrONS neurons

Code from my bachelor's thesis in the Computational Neuroscience lab of Julijana Gjorgjieva (TUM). The question: when excitatory synapses cluster on a dendrite, do inhibitory synapses sit next to those clusters, and does that differ from what you'd expect by chance? This is one angle on E–I co-tuning at the scale of single branches.

The input is a reconstructed neuron from the [MICrONS](https://www.microns-explorer.org/) cubic-millimetre dataset: a skeleton (SWC) plus a table of its input synapses.

## How it works

**1. Synapses onto the skeleton.** Each synapse is assigned to its nearest skeleton node (Euclidean distance).

**2. Density along the dendrite.** Distances between nodes are measured along the cable (geodesic), not through space. Synapse density at node *i* is a Gaussian kernel estimate over all nodes *j*:

```
ρ(i) = Σ_j  c_j · exp(−d_ij² / 2σ²) / (σ·√(2π))
```

where `c_j` is the number of synapses on node *j*, `d_ij` the geodesic distance in µm, and σ = 1 µm by default. Units are synapses per µm of cable.

**3. Clusters by gradient ascent.** From every node that carries a synapse, step to the neighbour with the highest density until no neighbour is higher. All synapses that end on the same local maximum form one cluster. Clusters below the neuron's mean synapse density are dropped.

**4. Inhibition relative to E clusters.** Inhibitory synapses are mapped to E clusters either by geodesic distance (with a cutoff of 1/mean density) or by following the E-density gradient. The notebook then compares clusters with and without nearby inhibition (cable length, synapse count, size), using Mann–Whitney U tests.

## Running it

```bash
conda create -n dendritic python=3.11 && conda activate dendritic
pip install -r requirements.txt
pytest                                   # runs on a small synthetic neuron, no data needed
jupyter notebook main/thesis_script_main.ipynb
```

Data is not included. Put the files here and set the neuron ID in `config/default.yaml`:

```
data/microns/extracted_swc/SWC_Skel/{swc_prefix}{neuron_id}.swc     skeleton, coordinates in µm
data/microns/extracted_swc/syn/{syn_prefix}{neuron_id}.xlsx         synapses: id, x, y, z (µm), syn_size,
                                                                    isOnSpine, pre_identity, pre_type
```

The first run computes the full node-by-node geodesic matrix, which takes 10–30 minutes and several GB of RAM for a ~30k-node neuron. It is cached under `data/microns/interim_results/`, keyed on a hash of the skeleton, so a changed skeleton never picks up an old matrix.

## E/I labels

The synapse table has no ground-truth E/I label. The thesis uses the `isOnSpine` flag of the export as a proxy (flag 0 = excitatory population, flag 1 = inhibitory). That proxy is imperfect: the same presynaptic axon sometimes ends up with both labels. So by default a known presynaptic cell type (`pre_type`, e.g. an interneuron class) now overrides the flag. The notebook prints how many synapses this relabels and how many axons carry mixed labels. Set `ei_labels.pre_type_override: false` to get the original thesis rule.

## Layout

```
config/default.yaml   neuron ID, paths, σ, E/I labelling, mapping method
src/                  loading, density, clustering, E/I mapping, plots
main/                 the analysis notebook
tests/                pipeline tests on a synthetic Y-shaped dendrite
```

## Limitations

- Most presynaptic partners have no cell-type label, so for them E/I identity still rests on the spine flag.
- No shuffled-position control yet, so cluster statistics are not compared against a null model. Many clusters contain only two synapses.
- The dense geodesic matrix limits this to one neuron at a time.
- Numbers differ slightly from the thesis PDF: after submission I fixed a node-ID misalignment in the density step and added the `pre_type` override.

## Contact

Maximilian Geisenheyner
