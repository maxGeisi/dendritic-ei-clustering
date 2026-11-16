# src/prep.py
from __future__ import annotations
import pandas as pd

# ---------- RAW-layer ops (work on original columns) ----------

def dedupe_raw_xyz(df: pd.DataFrame, x="x", y="y", z="z") -> pd.DataFrame:
    """Exact duplicate removal by raw x,y,z columns (original behavior)."""
    return df.drop_duplicates(subset=[x, y, z], keep="first").reset_index(drop=True)

def add_synapse_volume_nm3_to_um3(df: pd.DataFrame,
                                  syn_size_col: str = "syn_size",
                                  out_col: str = "SynapseVolume",
                                  dx_nm: float = 10, dy_nm: float = 10, dz_nm: float = 40) -> pd.DataFrame:
    """Compute volume in µm³ from syn_size * (dx*dy*dz) nm³."""
    df = df.copy()
    if syn_size_col in df.columns:
        voxel_vol_um3 = (dx_nm * dy_nm * dz_nm) / 1e9
        df[out_col] = df[syn_size_col] * voxel_vol_um3
    return df

# Presynaptic cell-type labels used in the MICrONS column tables. Only these are
# trusted to override the spine flag; everything else ("Unknown", NaN) is not.
DEFAULT_INHIBITORY_PRE_TYPES = ("PTC", "DTC", "STC", "ITC", "BC", "MC", "BPC", "NGC")
DEFAULT_EXCITATORY_PRE_TYPES = ("23P", "4P", "5P-IT", "5P-ET", "5P-NP", "6P-IT", "6P-CT")

def assign_e_i_labels(df: pd.DataFrame,
                      excitatory_flag: int = 0,
                      pre_type_override: bool = True,
                      inhibitory_pre_types=DEFAULT_INHIBITORY_PRE_TYPES,
                      excitatory_pre_types=DEFAULT_EXCITATORY_PRE_TYPES) -> pd.Series:
    """
    Label every synapse 'E' or 'I'.

    Base rule (as in the thesis): the `isOnSpine` flag. In the synapse export
    used here, flag == 0 marks the excitatory population and flag == 1 the
    inhibitory one. Which flag value means "excitatory" is a property of the
    export, so it is a parameter (`excitatory_flag`) and not hard-coded.

    The flag is only a proxy. Where the presynaptic cell has a known type
    (`pre_type`), that type wins if `pre_type_override` is set: an axon from an
    interneuron is inhibitory wherever it lands.
    """
    if "isOnSpine" not in df.columns:
        raise KeyError("Expected 'isOnSpine' column in synapse table.")

    label = pd.Series("I", index=df.index, dtype=object)
    label[df["isOnSpine"] == excitatory_flag] = "E"

    if pre_type_override and "pre_type" in df.columns:
        pre = df["pre_type"].astype(str)
        label[pre.isin(inhibitory_pre_types)] = "I"
        label[pre.isin(excitatory_pre_types)] = "E"
    return label


def ei_label_report(df: pd.DataFrame, excitatory_flag: int = 0, **label_kwargs) -> dict:
    """
    How far can the spine-flag labels be trusted?

    A single axon is either excitatory or inhibitory, so a presynaptic partner
    whose synapses carry both labels points to mislabelled synapses. Reports
    that count, plus how many synapses the pre_type override relabels.
    """
    by_flag = assign_e_i_labels(df, excitatory_flag, pre_type_override=False)
    final = assign_e_i_labels(df, excitatory_flag, **label_kwargs)
    report = {
        "n_synapses": int(len(df)),
        "n_E": int((final == "E").sum()),
        "n_I": int((final == "I").sum()),
        "n_relabelled_by_pre_type": int((by_flag != final).sum()),
    }
    if "pre_identity" in df.columns:
        per_axon = by_flag.groupby(df["pre_identity"]).agg(["nunique", "size"])
        multi = per_axon[per_axon["size"] >= 2]
        report["n_axons_with_2plus_synapses"] = int(len(multi))
        report["n_axons_with_mixed_flag_labels"] = int((multi["nunique"] > 1).sum())
    return report


def split_e_i_by_isOnSpine(df: pd.DataFrame,
                           excitatory_flag: int = 0,
                           pre_type_override: bool = True,
                           inhibitory_pre_types=DEFAULT_INHIBITORY_PRE_TYPES,
                           excitatory_pre_types=DEFAULT_EXCITATORY_PRE_TYPES,
                           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the synapse table into (excitatory, inhibitory) using assign_e_i_labels.
    Coordinates are renamed to Epos3D* / Ipos3D*, as the rest of the pipeline expects.

    With pre_type_override=False this is exactly the rule used in the thesis.
    """
    label = assign_e_i_labels(df, excitatory_flag, pre_type_override,
                              inhibitory_pre_types, excitatory_pre_types)
    syn_exec_df = df.loc[label == "E"].rename(
        columns={"x": "Epos3DX", "y": "Epos3DY", "z": "Epos3DZ"}).copy()
    syn_inh_df = df.loc[label == "I"].rename(
        columns={"x": "Ipos3DX", "y": "Ipos3DY", "z": "Ipos3DZ"}).copy()
    return syn_exec_df, syn_inh_df

# ---------- NORMALIZED-layer ops (standard schema for new pipeline) ----------

def normalize_position_columns(df: pd.DataFrame, x="x", y="y", z="z", unit="nm") -> pd.DataFrame:
    """Create x_um,y_um,z_um in micrometers from raw x,y,z."""
    df = df.copy()
    if unit.lower() == "nm":
        df["x_um"] = df[x] / 1000.0
        df["y_um"] = df[y] / 1000.0
        df["z_um"] = df[z] / 1000.0
    elif unit.lower() == "um":
        df["x_um"], df["y_um"], df["z_um"] = df[x], df[y], df[z]
    else:
        raise ValueError(f"Unknown unit: {unit}")
    return df

def compute_volume_um3(df: pd.DataFrame, dx_nm: float, dy_nm: float, dz_nm: float) -> pd.DataFrame:
    """Add volume_um3 from syn_size if present."""
    if "syn_size" not in df.columns:
        return df
    df = df.copy()
    voxel_vol_um3 = (dx_nm * dy_nm * dz_nm) / 1e9
    df["volume_um3"] = df["syn_size"] * voxel_vol_um3
    return df

def normalized_from_raw(df_raw: pd.DataFrame,
                        pos_unit: str = "um",
                        **label_kwargs) -> pd.DataFrame:
    """
    Normalized schema: x_um, y_um, z_um, 0/1 'is_on_spine' and type {'E','I'}.
    `pos_unit` is the unit of the raw x, y, z columns. The MICrONS export used
    in the thesis is already in micrometers.
    """
    df_norm = normalize_position_columns(df_raw, x="x", y="y", z="z", unit=pos_unit)
    df_norm["is_on_spine"] = df_raw["isOnSpine"].astype(int)
    df_norm["type"] = assign_e_i_labels(df_raw, **label_kwargs)

    keep = [c for c in ["x_um","y_um","z_um","syn_size","volume_um3","is_on_spine","type"] if c in df_norm.columns]
    return df_norm[keep]

def drop_duplicate_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Exact dedupe on x_um,y_um,z_um (normalized layer)."""
    cols = [c for c in ["x_um","y_um","z_um"] if c in df.columns]
    if not cols:
        return df
    return df.drop_duplicates(subset=cols, keep="first").reset_index(drop=True)
