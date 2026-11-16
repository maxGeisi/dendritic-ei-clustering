from pathlib import Path
import yaml
from dataclasses import dataclass, field

from .prep import DEFAULT_EXCITATORY_PRE_TYPES, DEFAULT_INHIBITORY_PRE_TYPES

@dataclass
class VoxelNM:
    dx: float
    dy: float
    dz: float

@dataclass
class Paths:
    base_data: str
    swc_dir: str
    syn_dir: str
    interim_results_dir: str   

@dataclass
class InputSpec:
    neuron_id: str
    swc_prefix: str
    syn_prefix: str

@dataclass
class Analysis:
    sigma_um: float
    distance_mapping: bool

@dataclass
class EILabels:
    """How synapses are labelled E or I. See prep.assign_e_i_labels."""
    excitatory_flag: int = 0
    pre_type_override: bool = True
    inhibitory_pre_types: list = field(default_factory=lambda: list(DEFAULT_INHIBITORY_PRE_TYPES))
    excitatory_pre_types: list = field(default_factory=lambda: list(DEFAULT_EXCITATORY_PRE_TYPES))

@dataclass
class Output:
    figures_dir: str

@dataclass
class Config:
    dataset_id: str
    paths: Paths
    input: InputSpec
    voxel_nm: VoxelNM
    analysis: Analysis
    output: Output
    ei_labels: EILabels = field(default_factory=EILabels)

def load_config(path: str | Path = "config/default.yaml") -> Config:
    with open(path, "r") as f:
        y = yaml.safe_load(f)

    return Config(
        dataset_id=y["dataset_id"],
        paths=Paths(**y["paths"]),
        input=InputSpec(**y["input"]),
        voxel_nm=VoxelNM(**y["voxel_nm"]),
        analysis=Analysis(**y["analysis"]),
        output=Output(**y["output"]),
        ei_labels=EILabels(**y.get("ei_labels", {})),
    )
