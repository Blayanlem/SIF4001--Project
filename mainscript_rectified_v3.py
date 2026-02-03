from __future__ import annotations

import math
import sys
import argparse
from pathlib import Path
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import  scatter_mean
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import radius_graph
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.util.jit import compile_mode
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from torch_geometric.loader import DataLoader

from improved_loss_functions_rectified_v2 import (
    step_penalty_loss,
    huber_step_penalty_loss,
    adaptive_loss_with_step_penalty,
    log_delta_huber_loss,
)

# ---------------------------
def print_banner(title: str):
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}")

def print_kv(key: str, val: str, width: int = 24):
    print(f"{key:<{width}}: {val}")

@torch.no_grad()
def report_and_export_split(loader, model, device, plots_dir, job_id, split_name: str, floor: float = 1e-3):
    if loader is None:
        print(f"{split_name.upper():<5} -> loader is None")
        return None

    # reuse your existing rel_mae_report_with_floor
    rel_mean_floor, rel_median_floor, rel_mean_masked, abs_mae, log_mae, preds, targets = \
        rel_mae_report_with_floor(loader, model, device, floor=floor, debug=False)

    if math.isnan(abs_mae):
        print(f"{split_name.upper():<5} -> metrics: N/A")
        return None

    # Export CSV once (best checkpoint context)
    save_predictions_to_csv(preds, targets, plots_dir, job_id, split_name=split_name)

    # Pretty print
    print(f"{split_name.upper():<5} MAE(Vref)              : {abs_mae:.6f}")
    print(f"{split_name.upper():<5} logMAE(log|V|+eps)     : {log_mae:.4f}")
    print(f"{split_name.upper():<5} RelMAE% mean(floor)     : {rel_mean_floor:.2f}%")
    print(f"{split_name.upper():<5} RelMAE% median(floor)   : {rel_median_floor:.2f}%")
    print(f"{split_name.upper():<5} RelMAE% mean(|t|>=floor): {rel_mean_masked:.2f}%")

    # Optional: show a tiny sample (don’t spam SLURM)
    try:
        import pandas as pd
        df = pd.DataFrame({"vref_true": targets.numpy(), "vref_pred": preds.numpy()})
        print(f"{split_name.upper():<5} sample (first 5):")
        print(df.head(5).to_string(index=False))
    except Exception:
        pass

    return abs_mae
# ---------------------------


torch.set_default_dtype(torch.float32)
DEFAULT_TORCH_DTYPE = torch.get_default_dtype()
# ----------------------------
# Target transform for log-normal coupling magnitudes
# ----------------------------
USE_LOG_TARGET = True               # train ? in log-space if True
LOG_EPS = 1e-5                      # dataset min ~1e-5; keeps log stable
LOG_CLAMP_MIN = 0.0                 # clamp baseline before abs/log (magnitude target)

def _v0_to_log(v0: torch.Tensor) -> torch.Tensor:
    """Convert baseline V0 (may be signed/noisy) to log-space magnitude."""
    v0_mag = torch.clamp(torch.abs(v0), min=LOG_CLAMP_MIN) + LOG_EPS
    return torch.log(v0_mag)

def _vref_to_log(vref: torch.Tensor) -> torch.Tensor:
    vref_mag = torch.clamp(vref, min=0.0) + LOG_EPS
    return torch.log(vref_mag)

def _log_to_vref_train(logv: torch.Tensor) -> torch.Tensor:
    """Invert log(|V|+eps) to a positive magnitude for TRAINING.

    Important: do NOT subtract LOG_EPS or clamp here; that would create a dead zone
    (zero gradient) whenever exp(logv) < LOG_EPS.
    """
    return torch.exp(logv)

def _log_to_vref_report(logv: torch.Tensor) -> torch.Tensor:
    """Invert log(|V|+eps) to a physical |V| magnitude for REPORTING."""
    return torch.clamp(torch.exp(logv) - LOG_EPS, min=0.0)

# Backwards-compatible default: reporting version.
_log_to_vref = _log_to_vref_report



def _log_deltas_to_linear_deltas(
    pred_delta_log: torch.Tensor,
    target_delta_log: torch.Tensor,
    v0_raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert ?log|V| targets/preds to linear-space ?V for legacy losses.

    Returns:
      pred_delta_lin, target_delta_lin, v0_mag

    Notes:
      - Targets are |V| magnitudes, so we use a magnitude baseline.
      - v0_mag is consistent with _v0_to_log/_log_to_vref (abs + eps in log, then exp - eps).
      - This keeps gradients flowing through exp() when optimizing in log-space.
    """
    logv0 = _v0_to_log(v0_raw)
    v0_mag = _log_to_vref_train(logv0)
    pred_vref = _log_to_vref_train(logv0 + pred_delta_log)
    target_vref = _log_to_vref_train(logv0 + target_delta_log)

    pred_delta_lin = pred_vref - v0_mag
    target_delta_lin = target_vref - v0_mag
    return pred_delta_lin, target_delta_lin, v0_mag


def _call_loss_with_optional_log_transform(
    loss_fn,
    pred_delta: torch.Tensor,
    target_delta: torch.Tensor,
    R_internal_batch: torch.Tensor,
    V0_batch: torch.Tensor,
    graph_cutoff_for_weighting: float,
    **kwargs,
) -> torch.Tensor:
    """Call a loss function with an explicit log-target mode flag.

    This removes the previous "convert ?log|V| -> ?V" adapter to avoid accidental
    double-application of the log transform.

    Contract:
      - If USE_LOG_TARGET=True, pred_delta/target_delta are ?log|V| (dimensionless).
        The loss_fn must reconstruct Vref internally from (V0_batch, ?log|V|).
      - If USE_LOG_TARGET=False, pred_delta/target_delta are linear ?V.
    """
    # Pass mode explicitly so the loss cannot silently flip due to environment variables.
    return loss_fn(
        pred_delta,
        target_delta,
        R_internal_batch,
        V0_batch,
        graph_cutoff_for_weighting,
        use_log_target=USE_LOG_TARGET,
        log_eps=LOG_EPS,
        log_clamp_min=LOG_CLAMP_MIN,
        **kwargs,
    )

# ----------------------------
# Helpers for robust scalar globals (data.u) handling
# ----------------------------
def _infer_num_graphs(data_obj: Data) -> int:
    """Infer number of graphs in a (batched) PyG Data/Batch robustly."""
    # Prefer PyG's num_graphs when sane
    ng = getattr(data_obj, "num_graphs", None)
    if isinstance(ng, int) and ng > 0:
        return ng
    # Fall back to batch vector (atoms) if present
    b = getattr(data_obj, "batch", None)
    if b is not None and torch.is_tensor(b) and b.numel() > 0:
        return int(b.max().item()) + 1
    return 0

def _ensure_u_2d(data_obj: Data, num_graphs: int, u_dim: int, device: torch.device) -> torch.Tensor:
    """Ensure data_obj.u is shaped [num_graphs, u_dim]. Works for u being [num_graphs*u_dim] or [u_dim]."""
    u = getattr(data_obj, "u", None)
    if u is None:
        return torch.empty((num_graphs, u_dim), device=device, dtype=DEFAULT_TORCH_DTYPE)
    if not torch.is_tensor(u):
        u = torch.as_tensor(u, device=device, dtype=DEFAULT_TORCH_DTYPE)
    else:
        u = u.to(device=device, dtype=DEFAULT_TORCH_DTYPE)

    if u.numel() == 0:
        return torch.empty((num_graphs, u_dim), device=device, dtype=DEFAULT_TORCH_DTYPE)

    if u.dim() == 2:
        # already [N, K]
        return u

    if u.dim() != 1:
        raise RuntimeError(f"Unexpected u.dim()={u.dim()} for u with shape {tuple(u.shape)}")

    # u is 1D. Try to reshape.
    if num_graphs > 0 and u.numel() == num_graphs * u_dim:
        return u.reshape(num_graphs, u_dim)
    if u.numel() == u_dim:
        # single graph
        return u.unsqueeze(0)
    if num_graphs == 0:
        # best-effort: infer N from length
        if u.numel() % u_dim == 0:
            return u.reshape(u.numel() // u_dim, u_dim)
    raise RuntimeError(
        f"Cannot reshape u of shape {tuple(u.shape)} into [num_graphs={num_graphs}, u_dim={u_dim}]."
    )

def _extract_v0_from_u(u_2d: torch.Tensor) -> torch.Tensor:
    """Extract V0 (baseline coupling) from u (shape [N, u_dim]) using SCALAR_GLOBAL_FEATURES_ORDER."""
    if u_2d.numel() == 0:
        return torch.empty((0,), device=u_2d.device, dtype=u_2d.dtype)
    if u_2d.dim() != 2:
        raise RuntimeError(f"Expected u_2d to be 2D, got {u_2d.dim()}D")
    return u_2d[:, V0_SCALAR_GLOBAL_INDEX].contiguous()

if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision("high")

PHYSICAL_UNIT_ANGSTROM = 1.0
INT_ANGSTROM_SCALE = 0.1

NM_TO_PHYSICAL_ANGSTROM = 10.0
DEBYE_TO_E_PHYSICAL_ANGSTROM = 0.20819434

SCALAR_GLOBAL_FEATURES_ORDER = ["R", "theta", "V0", "PR"]
V0_SCALAR_GLOBAL_INDEX = SCALAR_GLOBAL_FEATURES_ORDER.index("V0")
SCALAR_GLOBAL_IRREPS_STR = f"{len(SCALAR_GLOBAL_FEATURES_ORDER)}x0e"
SCALAR_GLOBAL_IRREPS_DIM = o3.Irreps(SCALAR_GLOBAL_IRREPS_STR).dim

FIELD_NODE_IRREPS = o3.Irreps("1x1o + 1x2e")
FIELD_NODE_IRREPS_DIM = FIELD_NODE_IRREPS.dim

IRREPS_EDGE_SH_PRECOMPUTE = o3.Irreps("4x0e+4x1o+4x2e+4x3o")

RUN_FROM_IDE = False

DEFAULT_ROOT_DIR = Path("/scr/user/blayanlem/FYP/combine")

DEFAULT_EPOCHS = 1500
DEFAULT_BATCH = 5
DEFAULT_PHYSICAL_CUTOFF = 12.0
DEFAULT_LR = 1e-3
DEFAULT_NUM_RBF = 50
DEFAULT_NUM_CONV_LAYERS = 6
PHYSICAL_JITTER_STRENGTH = 0.02
MAPE_EPSILON = 1e-7
BOOST_EXP  = 2
BOOST_BASE = 1.0

PT = {}
for Z, sym in enumerate(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca".split(), 1):
    PT[sym] = Z

@torch.jit.script
def random_rotation_matrix(device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    q = torch.randn(4, device=device, dtype=dtype)
    norm = q.norm() + 1e-8
    q = q / norm
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    R00 = 1 - 2*(y*y + z*z); R01 = 2*(x*y - w*z);   R02 = 2*(x*z + w*y)
    R10 = 2*(x*y + w*z);   R11 = 1 - 2*(x*x + z*z); R12 = 2*(y*z - w*x)
    R20 = 2*(x*z - w*y);   R21 = 2*(y*z + w*x);   R22 = 1 - 2*(x*x + y*y)
    
    row0 = torch.stack([R00, R01, R02]); row1 = torch.stack([R10, R11, R12]); row2 = torch.stack([R20, R21, R22])
    R_candidate = torch.stack([row0, row1, row2])
    if torch.linalg.det(R_candidate) < 0: R_candidate[:, 2] *= -1
    return R_candidate

@torch.no_grad()
def transform_by_matrix(
    irreps: o3.Irreps,
    feats: torch.Tensor,
    rotation: torch.Tensor,
    *,
    check: bool = False, 
) -> torch.Tensor:
    if rotation.shape != (3, 3):
        raise ValueError("`rotation` must have shape (3, 3)")
    
    original_shape = feats.shape
    if feats.ndim == 0 or feats.shape[-1] != irreps.dim:
        raise ValueError(f"Last dimension of feats ({feats.shape[-1]}) must match irreps.dim ({irreps.dim})")

    if feats.ndim > 1:
        feats_matrix = feats.reshape(-1, irreps.dim)
    else:
        feats_matrix = feats.unsqueeze(0)

    D = irreps.D_from_matrix(rotation)
    if check and not torch.allclose(D @ D.T, torch.eye(D.shape[0], dtype=D.dtype, device=D.device), atol=1e-4):
        print("Warning: D @ D.T is not close to identity in transform_by_matrix.")
    
    transformed_feats_matrix = torch.matmul(feats_matrix, D.T)
    
    if feats.ndim > 1:
        return transformed_feats_matrix.reshape(*original_shape[:-1], irreps.dim)
    else:
        return transformed_feats_matrix.squeeze(0)

def decompose_quadrupole_to_real_spherical(q_vec: torch.Tensor) -> torch.Tensor:
    if q_vec.ndim < 1 or q_vec.shape[-1] != 6:
        raise ValueError(f"Expected last dimension = 6, got {q_vec.shape}")
    Qxx, Qxy, Qxz, Qyy, Qyz, Qzz = q_vec[...,0], q_vec[...,1], q_vec[...,2], q_vec[...,3], q_vec[...,4], q_vec[...,5]
    tr = (Qxx + Qyy + Qzz) / 3.0
    Qxx_t, Qyy_t, Qzz_t = Qxx - tr, Qyy - tr, Qzz - tr
    c0 = (2 * Qzz_t - Qxx_t - Qyy_t) / math.sqrt(6.0)
    c1, c2 = math.sqrt(2.0) * Qxz, math.sqrt(2.0) * Qyz
    c3, c4 = (Qxx_t - Qyy_t) / math.sqrt(2.0), math.sqrt(2.0) * Qxy
    return torch.stack([c0, c1, c2, c3, c4], dim=-1).to(q_vec.device)

def read_pdb(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    elems = []
    xyz = []
    if not path.exists(): raise FileNotFoundError(f"PDB missing: {path}")
    with path.open() as fh:
        for line_idx, line in enumerate(fh):
            if line.startswith(("ATOM  ", "HETATM")):
                atom_name_field = line[12:16].strip()
                elem_symbol = ""
                if len(line) >= 78:
                    elem_field_std = line[76:78].strip()
                    if elem_field_std:
                        cap_elem_field_std = elem_field_std[0].upper() + (elem_field_std[1:].lower() if len(elem_field_std)>1 else "")
                        if cap_elem_field_std in PT: elem_symbol = cap_elem_field_std
                if (not elem_symbol or elem_symbol not in PT) and atom_name_field:
                    if len(atom_name_field) >= 2 and atom_name_field[0].isalpha():
                        potential_sym_2_cand = atom_name_field[:2].capitalize() if atom_name_field[1].islower() else atom_name_field[0].capitalize() + atom_name_field[1].lower()
                        if potential_sym_2_cand in PT: elem_symbol = potential_sym_2_cand
                    if not elem_symbol and atom_name_field[0].isalpha():
                        potential_sym_1 = atom_name_field[0].capitalize()
                        if potential_sym_1 in PT: elem_symbol = potential_sym_1
                if elem_symbol and elem_symbol in PT:
                    elems.append(PT[elem_symbol])
                    try: xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    except ValueError: raise ValueError(f"Could not parse coordinates for atom {elem_symbol} in {path} at line {line_idx+1}: {line.strip()}")
    if not elems: raise ValueError(f"No known atoms found or parsed in PDB: {path}")
    return torch.tensor(elems, dtype=torch.long), torch.tensor(xyz, dtype=DEFAULT_TORCH_DTYPE)

class RadialBesselBasisLayer(torch.nn.Module):
    def __init__(self, num_rbf: int, cutoff: float, learnable_freqs: bool = False, device=None, dtype=None):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        freq_dtype = torch.float64
        final_dtype = dtype if dtype is not None else DEFAULT_TORCH_DTYPE
        frequencies = torch.arange(1, num_rbf + 1, device=device, dtype=freq_dtype) * (math.pi / self.cutoff)
        if learnable_freqs: self.frequencies = nn.Parameter(frequencies.to(dtype=final_dtype))
        else: self.register_buffer('frequencies', frequencies)

    def _cosine_cutoff(self, distances: torch.Tensor) -> torch.Tensor:
        mask = (distances <= self.cutoff).float()
        return mask * (0.5 * (torch.cos(math.pi * distances / self.cutoff) + 1.0))

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        if distances.numel() == 0: return torch.empty((0, self.num_rbf), device=distances.device, dtype=distances.dtype)
        distances_shaped = distances[:, None] if distances.ndim == 1 else distances
        current_frequencies = self.frequencies.to(device=distances_shaped.device, dtype=distances_shaped.dtype)
        freq_dist = distances_shaped * current_frequencies.unsqueeze(0)
        rbf_values = torch.where(torch.abs(freq_dist) < 1e-6, torch.ones_like(freq_dist), torch.sin(freq_dist) / freq_dist)
        cutoff_factor = self._cosine_cutoff(distances_shaped)
        return cutoff_factor * rbf_values


class CachedDimerDataset(Dataset):
    def __init__(self, cache_file: str | Path, augment_with_rotations: bool = False, physical_jitter_strength: float = 0.0):
        super().__init__(root=None, transform=None, pre_transform=None, pre_filter=None)
        self.augment_with_rotations = augment_with_rotations
        self.physical_jitter_strength = physical_jitter_strength
        self.internal_jitter_strength = physical_jitter_strength * INT_ANGSTROM_SCALE
        
        loaded_data = torch.load(cache_file, weights_only=False)
        if isinstance(loaded_data, dict) and 'graphs' in loaded_data and 'metadata' in loaded_data:
            self.graphs: list[Data] = loaded_data['graphs']
            self.metadata: dict[str, Any] = loaded_data['metadata']
        elif isinstance(loaded_data, list):
            self.graphs: list[Data] = loaded_data
            self.metadata = {}
            print("Warning: Loaded legacy cache file without metadata. Consider re-generating the cache.")
        else:
            raise TypeError(f"Unknown cache file format for {cache_file}")

        if not isinstance(self.graphs, list):
            self.graphs = [self.graphs]
        
        self.internal_cutoff_from_cache = self.metadata.get('internal_cutoff_val', DEFAULT_PHYSICAL_CUTOFF * INT_ANGSTROM_SCALE)


    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        data = self.graphs[idx].clone()
        if self.augment_with_rotations or self.internal_jitter_strength > 0:
            pos_to_augment = data.pos_atoms.clone()
            rotation_mat = torch.eye(3, dtype=DEFAULT_TORCH_DTYPE, device=pos_to_augment.device)
            
            if self.internal_jitter_strength > 0:
                 pos_to_augment += torch.randn_like(pos_to_augment) * self.internal_jitter_strength
            
            if self.augment_with_rotations:
                rotation_mat = random_rotation_matrix(device=pos_to_augment.device, dtype=DEFAULT_TORCH_DTYPE)
                pos_to_augment = pos_to_augment @ rotation_mat.T
            
            data.pos_atoms = pos_to_augment

            if hasattr(data, 'field_node_features'):
                dip_part_dim = o3.Irreps("1x1o").dim
                dip_part = data.field_node_features[:dip_part_dim]
                quad_part = data.field_node_features[dip_part_dim:]
                data.field_node_features = torch.cat([
                    transform_by_matrix(o3.Irreps("1x1o"), dip_part, rotation_mat, check=False),
                    transform_by_matrix(o3.Irreps("1x2e"), quad_part, rotation_mat, check=False)
                ])
            
            if self.internal_jitter_strength > 0: 
                data.edge_index_atoms = radius_graph(data.pos_atoms, r=self.internal_cutoff_from_cache, loop=False, flow='source_to_target', max_num_neighbors=1000)
                if data.edge_index_atoms.dtype != torch.long: data.edge_index_atoms = data.edge_index_atoms.long()

                if data.edge_index_atoms.numel() > 0:
                    rel_pos_aa = data.pos_atoms[data.edge_index_atoms[1]] - data.pos_atoms[data.edge_index_atoms[0]]
                    data.edge_attr_sh = o3.spherical_harmonics(
                        IRREPS_EDGE_SH_PRECOMPUTE, rel_pos_aa, normalize=True, normalization='component'
                    )
                    data.edge_attr_atoms = rel_pos_aa.norm(dim=1, keepdim=True)
                else:
                    data.edge_attr_sh = torch.empty((0, IRREPS_EDGE_SH_PRECOMPUTE.dim), dtype=DEFAULT_TORCH_DTYPE, device=data.pos_atoms.device)
                    data.edge_attr_atoms = torch.empty((0,1), dtype=DEFAULT_TORCH_DTYPE, device=data.pos_atoms.device)
            elif self.augment_with_rotations and hasattr(data, 'edge_attr_sh') and data.edge_attr_sh.numel() > 0 :
                data.edge_attr_sh = transform_by_matrix(IRREPS_EDGE_SH_PRECOMPUTE, data.edge_attr_sh, rotation_mat, check=False)
                if hasattr(data,'edge_index_atoms') and data.edge_index_atoms.numel() > 0 and (not hasattr(data, 'edge_attr_atoms') or data.edge_attr_atoms is None):
                      data.edge_attr_atoms = (data.pos_atoms[data.edge_index_atoms[1]] - data.pos_atoms[data.edge_index_atoms[0]]).norm(dim=1, keepdim=True)

        return data

@compile_mode("script")
class PointsConvolutionIntegrated(torch.nn.Module):
    def __init__(self, irreps_node_input, irreps_node_attr, irreps_edge_sh, irreps_node_output,
                 fc_hidden_dims, num_rbf: int, rbf_cutoff: float, learnable_rbf_freqs: bool = False):
        super().__init__()
        self.irreps_node_input=o3.Irreps(irreps_node_input)
        self.irreps_node_attr=o3.Irreps(irreps_node_attr)
        self.irreps_edge_sh=o3.Irreps(irreps_edge_sh)
        self.irreps_node_output=o3.Irreps(irreps_node_output)
        self.rbf_basis=RadialBesselBasisLayer(num_rbf,rbf_cutoff,learnable_rbf_freqs, dtype=DEFAULT_TORCH_DTYPE)
        fc_input_dim=num_rbf
        self.sc=FullyConnectedTensorProduct(self.irreps_node_input,self.irreps_node_attr,self.irreps_node_output)
        self.lin1=FullyConnectedTensorProduct(self.irreps_node_input,self.irreps_node_attr,self.irreps_node_input)
        
        tp_output_channels_list_for_tp = []
        instructions_for_tp = []
        for i_in1, (mul_1, ir_1) in enumerate(self.irreps_node_input):
            for i_in2, (mul_2, ir_2) in enumerate(self.irreps_edge_sh):
                for ir_out_candidate in ir_1 * ir_2:
                    if ir_out_candidate in self.irreps_node_output or ir_out_candidate.l == 0:
                        i_out = len(tp_output_channels_list_for_tp)
                        tp_output_channels_list_for_tp.append( (mul_1, ir_out_candidate) )
                        instructions_for_tp.append( (i_in1, i_in2, i_out, "uvu", True) )

        if not instructions_for_tp:
            raise ValueError(f"No valid paths for TP from {self.irreps_node_input} x {self.irreps_edge_sh} to {self.irreps_node_output} or scalar.")
        
        irreps_tp_direct_output = o3.Irreps(tp_output_channels_list_for_tp)

        self.tp = o3.TensorProduct(
            self.irreps_node_input, self.irreps_edge_sh, irreps_tp_direct_output, instructions_for_tp,
            internal_weights=False, shared_weights=False
        )
        
        fc_neurons_full_list = [fc_input_dim] + fc_hidden_dims + [self.tp.weight_numel]
        self.fc = FullyConnectedNet(fc_neurons_full_list, F.silu)
        self.lin2 = FullyConnectedTensorProduct(self.tp.irreps_out.simplify(), self.irreps_node_attr, self.irreps_node_output)
        self.alpha = FullyConnectedTensorProduct(self.tp.irreps_out.simplify(), self.irreps_node_attr, "0e")
        with torch.no_grad(): self.alpha.weight.zero_()
        if not (self.alpha.irreps_out.lmax==0 and self.alpha.irreps_out.dim==1): raise AssertionError(f"Alpha FCTP output is not scalar. Got: {self.alpha.irreps_out}")

    def forward(self, node_input, node_attr, edge_sh_attr,
                edge_scalar_distances, batch_info_for_scatter) -> torch.Tensor:
        edge_src = batch_info_for_scatter['edge_src']
        edge_dst = batch_info_for_scatter['edge_dst']
        
        expanded_edge_scalars = self.rbf_basis(edge_scalar_distances)
        weight = self.fc(expanded_edge_scalars)
        
        node_self_connection = self.sc(node_input, node_attr)
        node_features_after_lin1 = self.lin1(node_input, node_attr)

        if edge_src.numel() > 0:
            if node_features_after_lin1.shape[0] == 0: raise ValueError("edge_src non-empty, node_features_after_lin1 0 nodes.")
            gathered_node_features = node_features_after_lin1[edge_src]
            
            edge_message_features = self.tp(gathered_node_features, edge_sh_attr, weight)
            aggregated_node_features = scatter_mean(edge_message_features, edge_dst, dim=0, dim_size=node_input.shape[0])
        else:
            aggregated_node_features = torch.zeros((node_input.shape[0], self.tp.irreps_out.dim),
                                                   device=node_input.device, dtype=node_input.dtype)
        
        node_conv_out_before_alpha = self.lin2(aggregated_node_features, node_attr)
        alpha_scalars = torch.tanh(self.alpha(aggregated_node_features, node_attr))
        m = self.sc.output_mask
        alpha_gate = (1 - m) + alpha_scalars * m
        return node_self_connection + alpha_gate * node_conv_out_before_alpha

MAX_OUTPUT_SCALE = 1.00

class DeltaCoupling(nn.Module):
    SCALAR_GLOBAL_IRREPS_STR_CLS = "4x0e"
    FIELD_NODE_IRREPS_STR_CLS = "1x1o + 1x2e"

    def __init__(self, max_Z, num_rbf=DEFAULT_NUM_RBF, internal_rbf_cutoff=DEFAULT_PHYSICAL_CUTOFF*INT_ANGSTROM_SCALE,
                 learnable_rbf_freqs=False, num_conv_layers=DEFAULT_NUM_CONV_LAYERS, output_scale=MAX_OUTPUT_SCALE, dropout=0.05, residual_mix=0.8, use_softplus_output=False):
        super().__init__()

        self.num_conv_layers = num_conv_layers
        self.output_scale = output_scale
        self.residual_mix = float(residual_mix)
        self.node_features_irreps = o3.Irreps("8x0e + 1x1o + 1x2e")
        self.atomic_scalar_embed_dim = o3.Irreps("8x0e").dim
        self.embed_atomic_scalar = nn.Embedding(max_Z+1, self.atomic_scalar_embed_dim)

        # Normalize embeddings
        nn.init.xavier_uniform_(self.embed_atomic_scalar.weight)

        self.field_node_irreps = o3.Irreps(DeltaCoupling.FIELD_NODE_IRREPS_STR_CLS)
        self.field_node_dim = self.field_node_irreps.dim

        self.slice_for_atomic_scalar = slice(0, self.atomic_scalar_embed_dim)
        self.slice_for_field_1o = slice(self.atomic_scalar_embed_dim, self.atomic_scalar_embed_dim + o3.Irreps("1x1o").dim)
        self.slice_for_field_2e = slice(self.slice_for_field_1o.stop, self.slice_for_field_1o.stop + o3.Irreps("1x2e").dim)

        self.use_softplus_output = use_softplus_output

        # Node attribute embedding
        self.node_attr_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.LayerNorm(16),
            nn.SiLU(),
            nn.Linear(16, 16),
            nn.SiLU()
        )
        self.embedded_node_attr_irreps = o3.Irreps("16x0e")

        # Scalar globals
        self.scalar_global_irreps = o3.Irreps(DeltaCoupling.SCALAR_GLOBAL_IRREPS_STR_CLS)

        # Placeholder for edge irreps
        self.irreps_edge_sh_for_conv = o3.Irreps("3x0e + 1x1o")  # adjust as needed

        # Convolutions
        self.convs = nn.ModuleList()
        for _ in range(num_conv_layers):
            conv = PointsConvolutionIntegrated(
                irreps_node_input=self.node_features_irreps,
                irreps_node_attr=self.embedded_node_attr_irreps,
                irreps_edge_sh=self.irreps_edge_sh_for_conv,
                irreps_node_output=self.node_features_irreps,
                fc_hidden_dims=[32,32],
                num_rbf=num_rbf,
                rbf_cutoff=internal_rbf_cutoff,
                learnable_rbf_freqs=learnable_rbf_freqs
            )
            self.convs.append(conv)

        # MLP for ?V prediction
        mlp_input_dim = self.node_features_irreps.dim + self.scalar_global_irreps.dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )
        self._init_weights()

    def _init_weights(self):
        # Xavier init for SiLU
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _build_atom_field_edges(self, ptr, pos_atoms, pos_field_nodes, device, dtype):
        num_graphs = len(ptr)-1
        if num_graphs==0 or pos_atoms.numel()==0:
            return (torch.empty((2,0),dtype=torch.long,device=device),
                    torch.empty((0,self.irreps_edge_sh_for_conv.dim),dtype=dtype,device=device),
                    torch.empty((0,1),dtype=dtype,device=device))

        atoms_per_graph = ptr[1:] - ptr[:-1]
        graph_indices_for_atoms = torch.repeat_interleave(torch.arange(num_graphs, device=device), atoms_per_graph)

        atom_indices_global = torch.arange(ptr[-1], device=device)
        field_node_indices_global_repeated = ptr[-1] + graph_indices_for_atoms

        src_af = atom_indices_global
        dst_af = field_node_indices_global_repeated
        src_fa = field_node_indices_global_repeated
        dst_fa = atom_indices_global

        src_combined = torch.cat([src_af, src_fa])
        dst_combined = torch.cat([dst_af, dst_fa])

        rel_pos_af = pos_field_nodes[graph_indices_for_atoms] - pos_atoms[atom_indices_global]
        rel_pos_full = torch.cat([rel_pos_af, -rel_pos_af])

        sh = o3.spherical_harmonics(self.irreps_edge_sh_for_conv, rel_pos_full, normalize=True, normalization='component')
        dist = rel_pos_full.norm(dim=1, keepdim=True)

        return torch.stack([src_combined, dst_combined]), sh, dist

    def forward(self, data):
        Z_atoms, pos_atoms, atomic_node_attr_input = data.z_atoms, data.pos_atoms, data.atomic_node_attr
        num_graphs = data.num_graphs if hasattr(data, 'num_graphs') else 0
        device = pos_atoms.device

        # Ensure scalar globals u is [num_graphs, SCALAR_GLOBAL_IRREPS_DIM]
        scalar_globals_u_in = _ensure_u_2d(data, _infer_num_graphs(data), SCALAR_GLOBAL_IRREPS_DIM, device)
        data.u = scalar_globals_u_in
        field_node_features_batch_in = data.field_node_features
        num_graphs = _infer_num_graphs(data)

        if field_node_features_batch_in is not None and field_node_features_batch_in.ndim == 1:
            # Total dimension of field node features (1x1o + 1x2e) = 3 + 5 = 8
            FIELD_DIM_TOTAL = o3.Irreps("1x1o + 1x2e").dim
            # Reshape from 1D (N*8) to 2D (N, 8). The -1 infers N from the total length.
            try:
                field_node_features_batch_in = field_node_features_batch_in.reshape(-1, FIELD_DIM_TOTAL)
            except RuntimeError as e:
                print(f"ERROR: Could not reshape field features. Tensor size: {field_node_features_batch_in.shape[0]}, Expected size: {num_graphs * FIELD_DIM_TOTAL}")
                raise e

        # Atomic embeddings
        x_atomic_scalar_embedded = self.embed_atomic_scalar(Z_atoms)
        x_atoms = torch.zeros(Z_atoms.shape[0], self.node_features_irreps.dim, device=device, dtype=DEFAULT_TORCH_DTYPE)
        x_atoms[:, self.slice_for_atomic_scalar] = x_atomic_scalar_embedded

        # Field node embeddings
        x_field_nodes = torch.zeros(num_graphs, self.node_features_irreps.dim, device=device, dtype=DEFAULT_TORCH_DTYPE)
        if num_graphs>0 and field_node_features_batch_in is not None:
            dip_dim = o3.Irreps("1x1o").dim
            x_field_nodes[:, self.slice_for_field_1o] = field_node_features_batch_in[:,:dip_dim]
            x_field_nodes[:, self.slice_for_field_2e] = field_node_features_batch_in[:,dip_dim:]

        x_combined = torch.cat([x_atoms, x_field_nodes], dim=0)

        # Node attributes
        embedded_attr_atoms = self.node_attr_mlp(atomic_node_attr_input)
        embedded_attr_field = torch.zeros(num_graphs, self.embedded_node_attr_irreps.dim, device=device, dtype=DEFAULT_TORCH_DTYPE)
        node_attr_combined = torch.cat([embedded_attr_atoms, embedded_attr_field], dim=0)

        # Edge construction (atoms + atom-field)
        edge_index_atoms = data.edge_index_atoms
        if edge_index_atoms.numel()>0:
            rel_pos_aa = pos_atoms[edge_index_atoms[1]] - pos_atoms[edge_index_atoms[0]]
            edge_attr_sh_atoms = o3.spherical_harmonics(self.irreps_edge_sh_for_conv, rel_pos_aa, normalize=True, normalization='component')
            edge_attr_distances_atoms_val = rel_pos_aa.norm(dim=1, keepdim=True)
        else:
            edge_attr_sh_atoms = torch.empty((0,self.irreps_edge_sh_for_conv.dim), device=device, dtype=DEFAULT_TORCH_DTYPE)
            edge_attr_distances_atoms_val = torch.empty((0,1), device=device, dtype=DEFAULT_TORCH_DTYPE)

        edge_index_af, sh_af, dist_af = self._build_atom_field_edges(data.ptr, pos_atoms, scatter_mean(pos_atoms, data.batch, dim=0), device, DEFAULT_TORCH_DTYPE)

        # Combine edges
        if edge_index_atoms.numel()>0:
            edge_index_combined = torch.cat([edge_index_atoms, edge_index_af], dim=1)
            edge_attr_sh_combined = torch.cat([edge_attr_sh_atoms, sh_af], dim=0)
            edge_attr_distances_combined = torch.cat([edge_attr_distances_atoms_val, dist_af], dim=0)
        else:
            edge_index_combined = edge_index_af
            edge_attr_sh_combined = sh_af
            edge_attr_distances_combined = dist_af

        # Convolution layers with stabilized residuals
        x_conv_out = x_combined
        for i_conv, conv in enumerate(self.convs):
            residual = x_conv_out
            batch_info = {'edge_src': edge_index_combined[0], 'edge_dst': edge_index_combined[1]}
            x_conv_out = conv(x_conv_out, node_attr_combined, edge_attr_sh_combined, edge_attr_distances_combined, batch_info)
            if (i_conv+1)%2==0 and x_conv_out.shape==residual.shape:
                x_conv_out = self.residual_mix * x_conv_out + (1.0 - self.residual_mix) * residual  # tunable residual mix

        # Graph-level pooling
        batch_combined = torch.cat([data.batch, torch.arange(num_graphs, device=device)], dim=0)
        x_scattered = scatter_mean(x_conv_out, batch_combined, dim=0, dim_size=num_graphs)

        # Combine with scalar globals
        h = torch.cat([x_scattered, scalar_globals_u_in], dim=1) if num_graphs>0 else torch.empty((0,self.node_features_irreps.dim + self.scalar_global_irreps.dim), device=device, dtype=DEFAULT_TORCH_DTYPE)

        
# ?V prediction (signed!)
        delta_v_pred = self.mlp(h).squeeze(-1)

        # IMPORTANT:
        # Your target is |V| (magnitude) so Vref must be >= 0, BUT ?V = Vref - V0 can be negative.
        # Therefore, DO NOT constrain delta_v_pred to be non-negative (softplus here breaks delta-learning).
        # Positivity is enforced after summation: Vref_pred = clamp(V0 + ?V, min=0).

        # Step 2: Apply the final output scale
        delta_v_pred = delta_v_pred * self.output_scale

        return delta_v_pred

def mae_vref_fn(pred_vref: torch.Tensor, target_vref: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error on Vref (in the same units as the coupling target)."""
    if pred_vref.numel() == 0:
        return torch.tensor(0.0, device=pred_vref.device, dtype=DEFAULT_TORCH_DTYPE)
    return torch.mean(torch.abs(target_vref - pred_vref))

def rel_mae_percent_with_floor(pred_vref: torch.Tensor, target_vref: torch.Tensor, floor: float = 1e-3) -> torch.Tensor:
    """Relative MAE in percent with a denominator floor to avoid blow-ups near zero targets."""
    if pred_vref.numel() == 0:
        return torch.tensor(0.0, device=pred_vref.device, dtype=DEFAULT_TORCH_DTYPE)
    denom = torch.clamp(torch.abs(target_vref), min=floor)
    return torch.mean(torch.abs(target_vref - pred_vref) / denom) * 100.0

def distance_weighted_loss(pred_delta_v, target_delta_v, R_internal_batch, V0_batch,
                           graph_cutoff_for_weighting, negative_penalty_factor=2.0,
                           eps=1e-3, softplus_beta=1.0):
    """
    Stable distance-weighted loss for delta-learning.
    """

    if pred_delta_v.numel() == 0:
        return torch.tensor(0.0, device=pred_delta_v.device, dtype=torch.float32, requires_grad=True)

    # --- START OF CRITICAL EDIT ---
    # 1. REMOVE the redundant F.softplus constraint here. 
    # Vref is now calculated as V0 + DV, where DV is already guaranteed to be >= 0 by the model's forward pass.
    
    if USE_LOG_TARGET:
        # pred_delta_v is ?logV; reconstruct Vref in linear space for loss
        logv0 = _v0_to_log(V0_batch)
        pred_logvref = logv0 + pred_delta_v
        pred_vref_raw = _log_to_vref_train(pred_logvref)
    else:
        pred_vref_raw = V0_batch + pred_delta_v
    # Magnitude target: enforce physical constraint on the final coupling, not on ?V.
    pred_vref = torch.clamp(pred_vref_raw, min=0.0)

    if USE_LOG_TARGET:
        target_logvref = _v0_to_log(V0_batch) + target_delta_v
        target_vref = _log_to_vref_train(target_logvref)
    else:
        target_vref = V0_batch + target_delta_v

    # Make sure shapes match
    R_internal_batch = R_internal_batch.to(pred_vref.device)
    if R_internal_batch.ndim == 1:
        R_internal_batch = R_internal_batch.unsqueeze(-1)
    if pred_vref.ndim == 1:
        pred_vref = pred_vref.unsqueeze(-1)
    if target_vref.ndim == 1:
        target_vref = target_vref.unsqueeze(-1)

    # Per-sample weights by internal distance
    weights = 1.0 + (R_internal_batch / graph_cutoff_for_weighting)

    # Absolute error term (primary)
    abs_error = torch.abs(target_vref - pred_vref)
    base_loss = torch.mean(weights * abs_error)

    # Safe relative error (clamp denom)
    denom = torch.clamp(torch.abs(target_vref), min=eps)
    rel_error = abs_error / denom
    relative_loss = torch.mean(weights * rel_error)

    # --- THIS PENALTY IS NOW ACTIVE AND PROPORTIONAL ---
    # It will correctly penalize samples where pred_vref < 0.
    penalty_loss = torch.mean(torch.relu(-pred_vref_raw) * negative_penalty_factor) 

    # Scale factors - tune these if needed
    total_loss = base_loss + 0.05 * relative_loss + 1.00 * penalty_loss
    return total_loss

def run_epoch(
    loader,
    model,
    optimizer=None,
    scheduler=None,
    device="cpu",
    loss_for_backward_fn=None,
    metric_fn_for_eval=mae_vref_fn,
    is_training: bool = False,
    dist_weight_cutoff: float = DEFAULT_PHYSICAL_CUTOFF * INT_ANGSTROM_SCALE,
    use_amp: bool = True,
    gradient_clip_val: float = 1.0,
):
    """
    Unified training / evaluation loop.

    Supports:
      - linear delta targets
      - log-delta targets (USE_LOG_TARGET)
      - AMP with safe fallback
      - gradient clipping
      - correct scheduler stepping

    IMPORTANT:
      If USE_LOG_TARGET=True and you're using a cached dataset, the cached `y`
      might be linear ?V from an older run. In that case we MUST recompute ?logV
      from (y_true_vref, V0) to avoid a silent target mismatch.
    """

    model.train(is_training)
    total_display_val = 0.0
    n_graphs_processed = 0

    if loader is None or len(loader) == 0:
        return float("nan")

    scaler = torch.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    for batch_idx, data_obj in enumerate(loader):
        try:
            data_obj = data_obj.to(device)

            # ---- infer graph count + reshape scalar globals ----
            num_graphs = _infer_num_graphs(data_obj)
            data_obj.u = _ensure_u_2d(
                data_obj,
                num_graphs,
                SCALAR_GLOBAL_IRREPS_DIM,
                device,
            )
            v0_from_u = _extract_v0_from_u(data_obj.u)

            # ---- AMP dtype selection ----
            amp_dtype = torch.float16
            if device.type == "cuda" and use_amp:
                major_cc, _ = torch.cuda.get_device_capability(device)
                if major_cc >= 8:
                    amp_dtype = torch.bfloat16

            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=(use_amp and device.type == "cuda"),
            ):
                with torch.set_grad_enabled(is_training):

                    # ---- forward ----
                    pred_delta_v = model(data_obj)
                    actual_graphs = pred_delta_v.shape[0]

                    if actual_graphs == 0:
                        continue
                    if actual_graphs != num_graphs:
                        # allow only trivial zero case
                        if not (actual_graphs == 0 and num_graphs == 0):
                            continue

                    # ---- reconstruct Vref for evaluation ----
                    if USE_LOG_TARGET:
                        log_v0 = _v0_to_log(v0_from_u[:actual_graphs])
                        pred_vref_eval = _log_to_vref_report(log_v0 + pred_delta_v)
                    else:
                        pred_vref_eval = torch.clamp(
                            v0_from_u[:actual_graphs] + pred_delta_v,
                            min=0.0,
                        )

                    target_vref_eval = (
                        data_obj.y_true_vref.squeeze(-1)[:actual_graphs]
                    )

                    # ---- loss (training only) ----
                    loss_opt = torch.tensor(
                        0.0,
                        device=device,
                        dtype=DEFAULT_TORCH_DTYPE,
                        requires_grad=is_training,
                    )

                    if is_training and optimizer and loss_for_backward_fn:
                        # Cache-proof target handling:
                        if USE_LOG_TARGET:
                            # IMPORTANT: cached datasets may store y as linear ?V.
                            # Recompute ?logV from true Vref to avoid cache/flag mismatch.
                            target_delta_v = (
                                _vref_to_log(data_obj.y_true_vref.squeeze(-1)[:actual_graphs])
                                - _v0_to_log(v0_from_u[:actual_graphs])
                            )
                        else:
                            target_delta_v = (
                                data_obj.y.squeeze(-1)[:actual_graphs]
                            )

                        r_internal = (
                            data_obj.R_internal_val.squeeze(-1)[:actual_graphs]
                        )

                        loss_opt = loss_for_backward_fn(
                            pred_delta_v,
                            target_delta_v,
                            r_internal,
                            v0_from_u[:actual_graphs],
                            dist_weight_cutoff,
                        )

            # ---- metric (eval only) ----
            metric_val = float("nan")
            if (
                metric_fn_for_eval
                and pred_vref_eval.numel() > 0
                and not is_training
            ):
                metric_val = metric_fn_for_eval(
                    pred_vref_eval.float(),
                    target_vref_eval.float(),
                )
                if torch.is_tensor(metric_val):
                    metric_val = metric_val.item()

            # ---- backward ----
            if is_training and optimizer and loss_opt.requires_grad:
                if not (torch.isnan(loss_opt) or torch.isinf(loss_opt)):
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss_opt).backward()

                    # gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        gradient_clip_val,
                    )

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    print(
                        f"Warning: NaN/Inf loss at batch {batch_idx}, skipping."
                    )

                # ---- scheduler stepping ----
                if scheduler is not None and isinstance(
                    scheduler, torch.optim.lr_scheduler.OneCycleLR
                ):
                    scheduler.step()

            # ---- accumulate display metric ----
            if actual_graphs > 0:
                if is_training:
                    val_to_add = (
                        loss_opt.item()
                        if not (
                            torch.isnan(loss_opt) or torch.isinf(loss_opt)
                        )
                        else 0.0
                    )
                else:
                    val_to_add = (
                        metric_val
                        if not (
                            math.isnan(metric_val)
                            or math.isinf(metric_val)
                        )
                        else 0.0
                    )

                if val_to_add != 0.0:
                    total_display_val += val_to_add * actual_graphs
                    n_graphs_processed += actual_graphs

        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return (
        total_display_val / n_graphs_processed
        if n_graphs_processed > 0
        else float("nan")
    )



def plot_regression(model,loader,device,title="Regression Plot",save_path=None):
    model.eval(); all_preds_vref,all_targets_vref=[],[]
    if loader is None or not hasattr(loader,'dataset') or len(loader.dataset)==0: print(f"No plot: {title}, loader/dataset empty."); return
    with torch.no_grad():
        for batch_idx, data_obj in enumerate(loader):
            try:
                data_obj=data_obj.to(device)
                num_graphs_in_this_batch = _infer_num_graphs(data_obj)
                
                current_u = data_obj.u
                if not hasattr(data_obj, 'u') or data_obj.u is None:
                    current_u = torch.empty(num_graphs_in_this_batch, SCALAR_GLOBAL_IRREPS_DIM, device=device, dtype=DEFAULT_TORCH_DTYPE)
                elif current_u.ndim == 1:
                    if num_graphs_in_this_batch == 0 and current_u.numel() == SCALAR_GLOBAL_IRREPS_DIM:
                        num_graphs_in_this_batch = 1 # Update local num_graphs for this iteration
                        # Note: data_obj.num_graphs is not updated here as it's just for local processing in plot
                        current_u = current_u.unsqueeze(0)
                    elif num_graphs_in_this_batch == 1 and current_u.numel() == SCALAR_GLOBAL_IRREPS_DIM:
                        current_u = current_u.unsqueeze(0)
                    elif num_graphs_in_this_batch > 0 and current_u.numel() == num_graphs_in_this_batch * SCALAR_GLOBAL_IRREPS_DIM:
                         current_u = current_u.reshape(num_graphs_in_this_batch, SCALAR_GLOBAL_IRREPS_DIM)
                    elif num_graphs_in_this_batch == 0 and current_u.numel() == 0 :
                        current_u = current_u.reshape(0, SCALAR_GLOBAL_IRREPS_DIM)
                    else:
                        # If it's still 1D and doesn't fit known patterns, it's problematic for plotting this batch.
                        print(f"Plotting Warning: u shape {current_u.shape} for batch_idx {batch_idx} with num_graphs {num_graphs_in_this_batch} is unexpected. Skipping this batch for plot.")
                        continue # Skip to the next data_obj in the loader
                data_obj.u = current_u # Assign reshaped u back to data_obj for this scope

                # Assertions to ensure correctness (can be removed after debugging)
                if num_graphs_in_this_batch > 0:
                    if not (data_obj.u.ndim == 2 and \
                            data_obj.u.shape[0] == num_graphs_in_this_batch and \
                            data_obj.u.shape[1] == SCALAR_GLOBAL_IRREPS_DIM):
                        print(f"Plotting Warning: data_obj.u shape validation failed after reshape. Shape: {data_obj.u.shape}, Expected: ({num_graphs_in_this_batch}, {SCALAR_GLOBAL_IRREPS_DIM}). Skipping batch.")
                        continue
                elif num_graphs_in_this_batch == 0:
                     if not (data_obj.u.ndim == 2 and data_obj.u.shape[0] == 0 and data_obj.u.shape[1] == SCALAR_GLOBAL_IRREPS_DIM):
                        print(f"Plotting Warning: data_obj.u shape validation failed for empty batch after reshape. Shape: {data_obj.u.shape}, Expected: (0, {SCALAR_GLOBAL_IRREPS_DIM}). Skipping batch.")
                        continue

                v0 = data_obj.u[:,V0_SCALAR_GLOBAL_INDEX] if data_obj.u.numel()>0 and data_obj.u.shape[0] > 0 else torch.empty(0,device=device, dtype=DEFAULT_TORCH_DTYPE)
                pred_delta_v=model(data_obj)
                actual_model_out_graphs = pred_delta_v.shape[0]

                if actual_model_out_graphs == 0 and num_graphs_in_this_batch == 0: continue
                if actual_model_out_graphs != num_graphs_in_this_batch:
                     print(f"Plotting: Model out {actual_model_out_graphs} != batch graphs {num_graphs_in_this_batch}. Skip."); continue
                
                target_vref = data_obj.y_true_vref.squeeze(-1)
                if USE_LOG_TARGET:
                    pred_vref = _log_to_vref(_v0_to_log(v0) + pred_delta_v)
                else:
                    pred_vref = torch.clamp(v0 + pred_delta_v, min=0.0)
                if pred_vref.numel() > 0: all_preds_vref.append(pred_vref.cpu()); all_targets_vref.append(target_vref.cpu())
            except Exception as e: print(f"Error plotting batch: {e}"); continue
    if not all_preds_vref: print(f"No data for plot: {title}."); return
    preds_np = torch.cat(all_preds_vref).numpy() if all_preds_vref else np.array([])
    targets_np = torch.cat(all_targets_vref).numpy() if all_targets_vref else np.array([])

    if preds_np.size==0 or targets_np.size==0: print(f"No valid data for plot {title}"); return
    plt.figure(figsize=(8,8)); plt.scatter(targets_np,preds_np,alpha=0.5,label="Pred vs Actual Vref")
    min_v_list, max_v_list = [], []
    if targets_np.size > 0: min_v_list.append(np.nanmin(targets_np)); max_v_list.append(np.nanmax(targets_np))
    if preds_np.size > 0: min_v_list.append(np.nanmin(preds_np)); max_v_list.append(np.nanmax(preds_np))
    min_v = np.nanmin(min_v_list) if min_v_list else np.nan
    max_v = np.nanmax(max_v_list) if max_v_list else np.nan
    if not (np.isnan(min_v) or np.isnan(max_v)): plt.plot([min_v,max_v],[min_v,max_v],'r--',lw=2,label="Ideal")
    plt.xlabel("Actual Vref"); plt.ylabel("Predicted Vref"); plt.title(title); plt.legend(); plt.grid(True)
    if save_path:
        try: plt.savefig(save_path); print(f"Plot saved: {save_path}")
        except Exception as e: print(f"Error saving plot {save_path}: {e}")
    else: plt.show()
    plt.close()

@torch.no_grad()
def _print_debug_stats(model, loader, device, loss_fn_base, internal_cutoff_val, epoch, use_adaptive, args):
    """Prints critical debug statistics on the first batch of the loader."""
    try:
        debug_batch = next(iter(loader)).to(device)
    except StopIteration:
        print("DEBUG: Cannot get a batch for critical debugging.")
        return

    model.eval()
    
    try:
        # --- FIX: Ensure u is 2D for indexing ---
        debug_num_graphs = _infer_num_graphs(debug_batch)
        debug_batch.u = _ensure_u_2d(debug_batch, debug_num_graphs, SCALAR_GLOBAL_IRREPS_DIM, device) 
        # --- END FIX ---

        # V0_SCALAR_GLOBAL_INDEX is an existing constant
        v0_batch = debug_batch.u[:, V0_SCALAR_GLOBAL_INDEX]
        # ... rest of the code
        target_vref = debug_batch.y_true_vref.squeeze(-1)
        pred_delta_v = model(debug_batch)
        if USE_LOG_TARGET:
            pred_vref = _log_to_vref(_v0_to_log(v0_batch) + pred_delta_v)
        else:
            pred_vref = torch.clamp(v0_batch + pred_delta_v, min=0.0)

        # --- Calculate Metrics ---
        batch_mae = torch.abs(pred_vref - target_vref).mean().item()
        num_negative = (pred_vref < 0).sum().item()
        v0_mean = v0_batch.mean().item()
        target_vref_min = target_vref.min().item()
        target_vref_max = target_vref.max().item()

        # Determine the correct loss function to use for the debug batch (must match the training loss)
        current_loss_fn_base = loss_fn_base
        if use_adaptive:
            # Recreate the adaptive loss lambda function to include the current epoch
            def loss_fn_for_debug(pred, target, R, V0, cutoff):
                return loss_fn_base(pred, target, R, V0, cutoff, 
                                           negative_penalty_factor=args.negative_penalty_factor, 
                                           epoch=epoch, max_epochs=args.epochs)
            current_loss_fn_base = loss_fn_for_debug
            
        # Calculate loss value on the debug batch (must match training target convention)
        if USE_LOG_TARGET:
            # Recompute ?logV from true Vref to avoid cache/flag mismatch
            target_delta_dbg = _vref_to_log(debug_batch.y_true_vref.squeeze(-1)) - _v0_to_log(v0_batch)
        else:
            target_delta_dbg = debug_batch.y.squeeze(-1)

        if hasattr(debug_batch, "R_internal_val") and debug_batch.R_internal_val is not None:
            R_dbg = debug_batch.R_internal_val.squeeze(-1)
        else:
            R_dbg = debug_batch.u[:, SCALAR_GLOBAL_FEATURES_ORDER.index("R")]

        loss_val = current_loss_fn_base(
            pred_delta_v,
            target_delta_dbg,
            R_dbg,
            v0_batch,
            internal_cutoff_val,
        ).item()

        print("======================================================================")
        print(f"CRITICAL DEBUGGING - EPOCH {epoch} (Validation Batch 1):")
        print(f"Input stats: V0 mean={v0_mean:.6f}, target Vref range=[{target_vref_min:.6f},{target_vref_max:.6f}]")
        print(f"  Pred delta_v: min={pred_delta_v.min().item():.6f}, max={pred_delta_v.max().item():.6f}, mean={pred_delta_v.mean().item():.6f}")
        print(f"  Pred Vref: min={pred_vref.min().item():.6f}, max={pred_vref.max().item():.6f}, MAE={batch_mae:.6f}")
        print(f"  Negative preds: {num_negative}/{pred_vref.numel()}")
        print(f"  Loss value (debug batch): {loss_val:.6f}")
        print("======================================================================")

    except Exception as e:
        print(f"ERROR in critical debug printing for Epoch {epoch}: {e}")
        import traceback
        traceback.print_exc()

def normalize_targets(dataset, V0_index=V0_SCALAR_GLOBAL_INDEX):
    """Compute statistics of target Vref values"""
    all_vref = []
    cpu = torch.device('cpu')
    for data in dataset:
        try:
            num_graphs = getattr(data, 'num_graphs', 1) or 1
            u2d = _ensure_u_2d(data, num_graphs=num_graphs, u_dim=SCALAR_GLOBAL_IRREPS_DIM, device=cpu)
            v0 = u2d[0, V0_index]
        except Exception:
            # Fallback: best-effort extraction for older cached formats
            u = getattr(data, 'u', torch.tensor([]))
            if torch.is_tensor(u) and u.numel() >= (V0_index + 1):
                v0 = u.flatten()[V0_index]
            else:
                v0 = torch.tensor(0.0)

        target_delta = data.y.item() if torch.is_tensor(data.y) else float(data.y)
        if USE_LOG_TARGET:
            vref = _log_to_vref(_v0_to_log(torch.tensor([v0])) + torch.tensor([target_delta]))
            all_vref.append(float(vref.item()))
        else:
            all_vref.append(float(v0) + float(target_delta))
            
    mean_vref = torch.tensor(all_vref).mean().item()
    std_vref = torch.tensor(all_vref).std().item()
    
    print(f"Target Vref statistics: mean={mean_vref:.4f}, std={std_vref:.4f}")
    print(f"Target range: [{min(all_vref):.4f}, {max(all_vref):.4f}]")
    
    return mean_vref, std_vref

def save_predictions_to_csv(preds, targets, output_dir: Path, job_id: str, split_name: str = "val"):
    """Saves prediction and target tensors to a CSV file."""
    preds_np = preds.numpy()
    targets_np = targets.numpy()
    
    df = pd.DataFrame({
        'Vref_true': targets_np,
        'Vref_pred': preds_np
    })
    
    filename = f"preds_targets_{job_id}_{split_name}.csv"
    save_path = output_dir / filename
    df.to_csv(save_path, index=False)
    # print(f"Saved predictions to: {save_path}")
    return save_path

def main():
    default_params = {
        'root_dir': str(DEFAULT_ROOT_DIR),
        'output_dir': '/scr/user/blayanlem/FYP/combine/outputs',
        'epochs': DEFAULT_EPOCHS,
        'batch_size': DEFAULT_BATCH,
        'physical_cutoff': DEFAULT_PHYSICAL_CUTOFF,
        'lr': DEFAULT_LR,
        'num_rbf': DEFAULT_NUM_RBF,
        'learnable_rbf_freqs': False,
        'num_conv_layers': DEFAULT_NUM_CONV_LAYERS,
        'apply_augmentations': True,
        'seed': 42,
        'cache_file_name': "combined_HK_all.pt",
        'num_workers': 4,
        'persistent_workers': True,
        'pin_memory': True,
        'augment_cached_train': False,
        'physical_jitter_strength': PHYSICAL_JITTER_STRENGTH,
        'use_amp': True,
        'loss_function': 'log_huber',  # NEW: Choose loss function
        'negative_penalty_factor': 250.0,  # NEW: Penalty strength
        'use_softplus_output': False,  # ?V must be signed; enforce Vref>=0 after summation ,
        'residual_mix': 0.8,            # weight on new message-passing update (1.0 = no damping)
        'log_huber_beta': 0.02,         # Huber beta in delta-space (log-delta when USE_LOG_TARGET)
        'log_huber_distance_weighting': False,  # optionally weight by distance
}
    
    ap = argparse.ArgumentParser(description="e3nn Delta-Learning Vref")
    for k, v in default_params.items():
        action = argparse.BooleanOptionalAction if isinstance(v, bool) else None
        type_ = type(v) if not isinstance(v, bool) and v is not None else str
        if type_ == Path: type_ = str
        ap.add_argument(f"--{k.replace('_', '-')}", type=type_, default=v, action=action)
    
    # Add special argument for loss function choice
    # ap.add_argument('--loss-function', type=str, default='step_penalty',
    #                choices=['step_penalty', 'huber_step', 'adaptive', 'distance_weighted'],
    #                help='Loss function to use for training')
    
    if RUN_FROM_IDE:
        args = argparse.Namespace(**default_params)
        args.root_dir = str(default_params['root_dir'])
        args.output_dir = str(default_params['output_dir'])
    else:
        args = ap.parse_args()

    internal_cutoff_val = args.physical_cutoff * INT_ANGSTROM_SCALE
    
    # DATA directory
    current_root_dir = Path(args.root_dir)
    if not current_root_dir.exists():
        print(f"Error: Root {current_root_dir} missing.")
        sys.exit(1)

    # OUTPUT directory
    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Get SLURM job ID
    import os
    job_id = os.environ.get('SLURM_JOB_ID', None)
    if job_id:
        run_identifier = f"job_{job_id}"
    else:
        from datetime import datetime
        run_identifier = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    plots_dir = output_base_dir / "plots" / run_identifier
    models_dir = output_base_dir / "models" / run_identifier
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directories created:")
    print(f"    Plots: {plots_dir}")
    print(f"    Models: {models_dir}")
    print("------------------------------------------------------------------------------")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print("------------------------------------------------------------------------------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loss function: {args.loss_function}")
    print(f"Negative penalty factor: {args.negative_penalty_factor}")
    print(f"Softplus output: {args.use_softplus_output}")
    if args.use_softplus_output:
        print("WARNING: use_softplus_output constrains ?V>=0 and breaks delta-learning when V0 can overestimate Vref. Strongly recommend --no-use-softplus-output.")
    print(f"Other params: {vars(args)}")
    
    print("------------------------------------------------------------------------------")

    # [Keep all cache loading code - lines 745-788]
    cache_fp = current_root_dir / args.cache_file_name
    if not cache_fp.exists():
        print(f"Cache file {cache_fp} not found. Building it now...")
        try:
            from preprocess_and_cache import build_cache
            build_cache(current_root_dir, args.physical_cutoff, outfile_name=args.cache_file_name,
                       internal_angstrom_scale=INT_ANGSTROM_SCALE, default_torch_dtype=DEFAULT_TORCH_DTYPE)
            print(f"Cache built: {cache_fp}")
        except ImportError:
            print("Error: Could not import 'build_cache'.")
            sys.exit(1)
        except Exception as e:
            print(f"Error building cache: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    loaded_cache_metadata = {}
    if cache_fp.exists():
        try:
            cache_content = torch.load(cache_fp, weights_only=False)
            if isinstance(cache_content, dict) and 'metadata' in cache_content:
                loaded_cache_metadata = cache_content['metadata']
                cached_internal_cutoff = loaded_cache_metadata.get('internal_cutoff_val', -1.0)
                cached_scale = loaded_cache_metadata.get('INT_ANGSTROM_SCALE', -1.0)
                if not math.isclose(cached_internal_cutoff, internal_cutoff_val) or \
                   not math.isclose(cached_scale, INT_ANGSTROM_SCALE):
                    print(f"Critical Warning: Mismatch between cached metadata!")
                    sys.exit(1)
        except Exception as e:
            print(f"Could not verify cache metadata: {e}")

    # [Keep all dataset loading code - lines 791-848]
    ds_train_full = CachedDimerDataset(
        cache_fp,
        augment_with_rotations=args.apply_augmentations,
        physical_jitter_strength=args.physical_jitter_strength if args.apply_augmentations else 0.0
    )
    ds_train_full.metadata = loaded_cache_metadata
    print(f"Using CachedDimerDataset for training. Augment: {args.apply_augmentations}")

    ds_eval_full = CachedDimerDataset(cache_fp, augment_with_rotations=False, physical_jitter_strength=0.0)
    ds_eval_full.metadata = loaded_cache_metadata
    print("Using CachedDimerDataset for validation/test.")

    print("------------------------------------------------------------------------------")

    n = len(ds_eval_full)
    print(f"Total dataset size: {n}")
    current_batch_size = max(1, min(args.batch_size, n)) if n > 0 else 1
    
    all_idx = list(range(n))
    np.random.shuffle(all_idx)
    
    if n >= 10:
        n_val_count = max(1, int(round(n * 0.1)))
        n_te_count = max(1, int(round(n * 0.1)))
        n_tr_count = n - n_val_count - n_te_count
    elif n > 0:
        n_tr_count, n_val_count, n_te_count = n, 0, 0
    else:
        n_tr_count, n_val_count, n_te_count = 0, 0, 0
    
    train_indices = all_idx[:n_tr_count]
    val_indices = all_idx[n_tr_count:n_tr_count + n_val_count]
    test_indices = all_idx[n_tr_count + n_val_count:]
    
    print(f"Splitting: Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    print("------------------------------------------------------------------------------")

    datasets = {}
    if train_indices:
        datasets['train'] = torch.utils.data.Subset(ds_train_full, train_indices)
    if val_indices:
        datasets['val'] = torch.utils.data.Subset(ds_eval_full, val_indices)
    if test_indices:
        datasets['test'] = torch.utils.data.Subset(ds_eval_full, test_indices)
    
    if 'train' in datasets and len(datasets['train']) > 0:
        mean_vref, std_vref = normalize_targets(datasets['train'], V0_index=V0_SCALAR_GLOBAL_INDEX)
    
    dl_args = {
        'num_workers': args.num_workers,
        'persistent_workers': args.persistent_workers if args.num_workers > 0 else False,
        'pin_memory': args.pin_memory if device.type == 'cuda' else False
    }
    
    tr_loader = DataLoader(datasets['train'], batch_size=current_batch_size, shuffle=True,
                          drop_last=(len(train_indices) > current_batch_size), **dl_args) if 'train' in datasets else None
    va_loader = DataLoader(datasets['val'], batch_size=current_batch_size, shuffle=False,
                          drop_last=(len(val_indices) > current_batch_size), **dl_args) if 'val' in datasets else None
    te_loader = DataLoader(datasets['test'], batch_size=current_batch_size, shuffle=False,
                          drop_last=(len(test_indices) > current_batch_size), **dl_args) if 'test' in datasets else None

    max_Z_val = max(data.z_atoms.max().item() for data in ds_train_full.graphs)
    # Create model with softplus option
    model = DeltaCoupling(
        max_Z=max_Z_val,
        num_rbf=args.num_rbf,
        internal_rbf_cutoff=internal_cutoff_val,
        learnable_rbf_freqs=args.learnable_rbf_freqs,
        num_conv_layers=args.num_conv_layers,
        residual_mix=args.residual_mix,
        use_softplus_output=args.use_softplus_output
    ).to(device)
    
    if device.type == 'cuda' and hasattr(model, 'to') and hasattr(torch, 'channels_last'):
        model = model.to(memory_format=torch.channels_last)
    
    # print("Running without torch.compile (better compatibility with e3nn on HPC)")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=5e-4,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    sched = None
    # Scheduler policy:
    # - Default to an epoch-stepped cosine schedule (stable for long runs).
    # - OneCycleLR is optional but NOT used by default because with large epochs it keeps LR tiny for too long.
    if tr_loader and len(tr_loader) > 0:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=max(1, args.epochs),
            eta_min=args.lr * 0.01,
        )


    best_val_mape, patience_count, EARLY_STOPPING_PATIENCE = float('inf'), 0, 50

    # --- REPLACEMENT: improved, robust training loop (drop-in) ---
    # SELECT LOSS FUNCTION and prepare training utilities
    if tr_loader:
        # --- SELECT LOSS FUNCTION and prepare training utilities ---
        # print(f"Training: RBFs:{args.num_rbf}, Lyrs:{args.num_conv_layers}, MaxLR:{args.lr}, Aug:{args.apply_augmentations}")

        # Initialize global loss variables (must be outside if/elif blocks)
        loss_fn_base = None  # Will hold the core function for the adaptive case
        use_adaptive = False

        # Choose loss function (safer defaults / clear wrappers)
        if args.loss_function == 'step_penalty':
            print(f"Using step_penalty_loss with penalty factor {args.negative_penalty_factor}")
            def loss_fn_to_use(pred, target, R, V0, cutoff):
                return _call_loss_with_optional_log_transform(
                    step_penalty_loss,
                    pred, target, R, V0, cutoff,
                    negative_penalty_factor=args.negative_penalty_factor,
                )

        elif args.loss_function == 'log_huber':
            print(f"Using log_delta_huber_loss (Huber in delta-space) with beta {args.log_huber_beta}")
            def loss_fn_to_use(pred, target, R, V0, cutoff):
                return _call_loss_with_optional_log_transform(
                    log_delta_huber_loss,
                    pred, target, R, V0, cutoff,
                    beta=args.log_huber_beta,
                    weight_by_distance=args.log_huber_distance_weighting,
                )

        elif args.loss_function == 'huber_step':
            print(f"Using huber_step_penalty_loss with penalty factor {args.negative_penalty_factor}")
            def loss_fn_to_use(pred, target, R, V0, cutoff):
                return _call_loss_with_optional_log_transform(
                    huber_step_penalty_loss,
                    pred, target, R, V0, cutoff,
                    negative_penalty_factor=args.negative_penalty_factor,
                )

        elif args.loss_function == 'adaptive':
            print(f"Using adaptive_loss_with_step_penalty (penalty grows during training)")
            use_adaptive = True
            # This is the base function used by the adaptive wrapper
            def loss_fn_base(pred, target, R, V0, cutoff, **kw):
                return _call_loss_with_optional_log_transform(
                    adaptive_loss_with_step_penalty,
                    pred, target, R, V0, cutoff,
                    **kw,
                )
            # The function used for training is dynamically generated inside the loop
            # We skip defining loss_fn_to_use here as it's not needed for the training step.

        elif args.loss_function == 'distance_weighted':
            print("Using distance_weighted_loss")
            def loss_fn_to_use(pred, target, R, V0, cutoff):
                return distance_weighted_loss(pred, target, R, V0, cutoff,
                                             negative_penalty_factor=max(1.0, args.negative_penalty_factor))

        else:
            print(f"Unknown loss function: {args.loss_function}. Using distance_weighted as fallback.")
            def loss_fn_to_use(pred, target, R, V0, cutoff):
                return distance_weighted_loss(pred, target, R, V0, cutoff,
                                             negative_penalty_factor=max(1.0, args.negative_penalty_factor))

        # --- CRITICAL DEFINITION FOR DEBUGGING FUNCTION ---
        # The debugging function needs a consistent name for the core loss function.
        # If adaptive, use the base function. Otherwise, use the wrapper function.
        if use_adaptive:
            # We defined it as loss_fn_base above
            loss_fn_base_global = loss_fn_base
        else:
            # We defined the specific lambda wrapper as loss_fn_to_use
            loss_fn_base_global = loss_fn_to_use
        
        # Now, loss_fn_base_global is defined and holds the correct function (or base function).
        # --- END CRITICAL DEFINITION ---

        # Stable scheduler: step per epoch (avoid per-batch resets)
        # NOTE: sched is already set above (CosineAnnealingLR with eta_min=args.lr*0.01).
        # Do NOT overwrite it with a hard-coded eta_min (e.g. 1e-7), which can collapse LR to ~0 and stall learning.

        # Use MAE on Vref for early stopping / main validation metric (more stable than MAPE near zero)
        def mae_vref(pred_vref, target_vref):
            if pred_vref.numel() == 0: return float('nan')
            return float(torch.mean(torch.abs(pred_vref - target_vref)).item())

        @torch.no_grad()
        def baseline_v0_mae(loader):
            """Baseline where the model predicts delta=0, i.e. Vref_pred ≈ |V0| (or exp(log|V0|) - eps in log mode)."""
            if loader is None:
                return float('nan')
            preds, targs = [], []
            for b in loader:
                b = b.to(device)
                # Ensure scalar globals are shaped [num_graphs, u_dim]
                ng = _infer_num_graphs(b)
                b.u = _ensure_u_2d(b, ng, SCALAR_GLOBAL_IRREPS_DIM, device)
                v0 = _extract_v0_from_u(b.u)
                if USE_LOG_TARGET:
                    pred = _log_to_vref_report(_v0_to_log(v0))
                else:
                    pred = torch.clamp(v0, min=0.0)
                targ = b.y_true_vref.to(device)
                preds.append(pred.reshape(-1))
                targs.append(targ.reshape(-1))
            p = torch.cat(preds) if preds else torch.empty(0, device=device)
            t = torch.cat(targs) if targs else torch.empty(0, device=device)
            if p.numel() == 0:
                return float('nan')
            return float(torch.mean(torch.abs(p - t)).item())

        if va_loader is not None:
            base_mae = baseline_v0_mae(va_loader)
            print(f"Baseline (delta=0 → Vref≈|V0|) Val MAE: {base_mae:.6f}")

        best_val_metric = float('inf')
        patience_count = 0
        EARLY_STOPPING_PATIENCE = 50

        # Training loop
        for epoch in range(1, args.epochs + 1):
            # If using adaptive loss, create wrapper with epoch
            if use_adaptive:
                def loss_fn_to_use(pred, target, R, V0, cutoff):
                    return loss_fn_base(pred, target, R, V0, cutoff,
                                        negative_penalty_factor=args.negative_penalty_factor,
                                        epoch=epoch, max_epochs=args.epochs)

            # Train epoch
            train_loss = run_epoch(
                tr_loader, model, opt, None, device,
                loss_for_backward_fn=loss_fn_to_use,
                metric_fn_for_eval=None,   # training loop uses loss, not validation metric
                is_training=True,
                dist_weight_cutoff=internal_cutoff_val,
                use_amp=args.use_amp,
                gradient_clip_val=1.0
            )

            # Validation epoch: compute both MAE (for stopping) and MAPE for reporting (but avoid MAPE division by zero)
            # We call run_epoch with is_training=False and a metric function that returns MAE on Vref.
            val_mae = run_epoch(
                va_loader, model, None, None, device,
                loss_for_backward_fn=None,
                metric_fn_for_eval=lambda p, t: mae_vref(p, t),
                is_training=False,
                dist_weight_cutoff=internal_cutoff_val,
                use_amp=args.use_amp
            )

            # Also compute a stable "reported MAPE" but only over targets > small_threshold to avoid blow-ups
            def rel_mae_report_with_floor(loader, model, device, floor: float = 1e-3, debug: bool = False):
                """Compute a few robust validation metrics on physical Vref.

                Returns:
                    rel_mean_floor: mean(|err|/max(|target|,floor))*100 over all samples
                    rel_median_floor: median(|err|/max(|target|,floor))*100 over all samples
                    rel_mean_masked: mean(|err|/|target|)*100 but only where |target|>=floor
                    abs_mae: mean absolute error in Vref units
                    log_mae: mean absolute error in log(|V|+LOG_EPS)
                    preds, targets: concatenated tensors on CPU
                """
                model.eval()
                all_preds, all_targets = [], []

                # static flag attached to the function (persists across epochs)
                if not hasattr(rel_mae_report_with_floor, "_printed_u_once"):
                    rel_mae_report_with_floor._printed_u_once = False

                with torch.no_grad():
                    for data_obj in loader:
                        data_obj = data_obj.to(device)

                        # ---- OPTIONAL one-time schema print ----
                        if debug and (not rel_mae_report_with_floor._printed_u_once):
                            print("DEBUG u.shape:", tuple(data_obj.u.shape) if hasattr(data_obj, "u") else None)
                            u0 = data_obj.u[0].detach().cpu().numpy() if (hasattr(data_obj, "u") and data_obj.u.numel()) else "<empty>"
                            print("DEBUG u[0] (R,theta,V0,PR):", u0)
                            rel_mae_report_with_floor._printed_u_once = True

                        # Ensure u is [num_graphs, u_dim]
                        num_graphs = _infer_num_graphs(data_obj)
                        data_obj.u = _ensure_u_2d(data_obj, num_graphs, SCALAR_GLOBAL_IRREPS_DIM, device)

                        v0_from_u = _extract_v0_from_u(data_obj.u)

                        pred_delta_v = model(data_obj)
                        n = pred_delta_v.shape[0]
                        if n == 0:
                            continue

                        # Use the REPORTING inverse here because we want physical Vref for metrics/plots
                        if USE_LOG_TARGET:
                            pred_vref = _log_to_vref(_v0_to_log(v0_from_u[:n]) + pred_delta_v)
                        else:
                            pred_vref = torch.clamp(v0_from_u[:n] + pred_delta_v, min=0.0)

                        target_vref = data_obj.y_true_vref.squeeze(-1)[:n]

                        all_preds.append(pred_vref.detach().cpu())
                        all_targets.append(target_vref.detach().cpu())

                if not all_preds:
                    empty = torch.tensor([])
                    nan = float("nan")
                    return nan, nan, nan, nan, nan, empty, empty

                preds = torch.cat(all_preds)
                targets = torch.cat(all_targets)

                # Absolute MAE in Vref units
                abs_mae = torch.mean(torch.abs(preds - targets))

                # Relative errors
                denom_floor = torch.clamp(torch.abs(targets), min=floor)
                rel_floor = (torch.abs(preds - targets) / denom_floor) * 100.0
                rel_mean_floor = torch.mean(rel_floor)
                rel_median_floor = torch.median(rel_floor)

                mask = torch.abs(targets) >= floor
                if torch.any(mask):
                    rel_mean_masked = torch.mean((torch.abs(preds[mask] - targets[mask]) / torch.abs(targets[mask])) * 100.0)
                else:
                    rel_mean_masked = torch.tensor(float("nan"))

                # Log-space MAE: more stable when values span orders of magnitude
                preds_nonneg = torch.clamp(preds, min=0.0)
                targets_nonneg = torch.clamp(targets, min=0.0)
                log_preds = torch.log(preds_nonneg + LOG_EPS)
                log_targets = torch.log(targets_nonneg + LOG_EPS)
                log_mae = torch.mean(torch.abs(log_preds - log_targets))

                return (float(rel_mean_floor.item()), float(rel_median_floor.item()),
                        float(rel_mean_masked.item()) if not torch.isnan(rel_mean_masked) else float("nan"),
                        float(abs_mae.item()), float(log_mae.item()), preds, targets)
            val_rel_mean_floor, val_rel_median_floor, val_rel_mean_masked, val_abs_mae_report, val_log_mae_report, val_preds, val_targets = rel_mae_report_with_floor(va_loader, model, device, floor=1e-3, debug=False)

            job_id = os.environ.get("SLURM_JOB_ID", "local")
    
            # Save the data
            # if not math.isnan(val_rel_mean_floor):
            #     save_predictions_to_csv(val_preds, val_targets, plots_dir, job_id, split_name="val")

            # Logging
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                from datetime import datetime
                val_mape_str = f"{val_rel_mean_masked:.2f}" if not math.isnan(val_rel_mean_masked) else "   N/A"
                lr_val = opt.param_groups[0]['lr']
                val_mae_str = f"{val_mae:.6f}" if not math.isnan(val_mae) else "N/A"
                val_mape_str = f"{val_rel_mean_floor:.2f}" if not math.isnan(val_rel_mean_floor) else "   N/A"
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print("------------------------------------------------------------------------------")
                print(f"[{timestamp}] E {epoch:3d}|Train Loss:{train_loss:8.3e}|Val MAE(Vref):{val_mae_str}|Val MAPE_report:{val_mape_str}%|LR:{lr_val:.2e}")

            # Detailed Debugging on Epoch 1 and every 10th epoch
            DO_CRITICAL_DEBUG = False
            if DO_CRITICAL_DEBUG and tr_loader and (epoch == 1 or epoch % 10 == 0):
                # Using the function defined outside the main loop for cleanliness
                _print_debug_stats(model, tr_loader, device, loss_fn_base_global, internal_cutoff_val, epoch, use_adaptive, args)

            # Early stopping & model saving uses MAE on Vref
            if not math.isnan(val_mae):
                if val_mae < best_val_metric:
                    best_val_metric = val_mae
                    torch.save(model.state_dict(), models_dir / "best_model_checkpoint.pt")
                    patience_count = 0
                    print(f"---> Best val MAE(Vref):{best_val_metric:.6e}. Saved.")
                else:
                    patience_count += 1

                if patience_count >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stop at epoch {epoch}.")
                    break

            # Step scheduler per epoch (if present)
            if sched is not None:
                try:
                    sched.step()
                except Exception:
                    pass

    else:
        print("Skipping training: no training data.")
    # --- END REPLACEMENT ---

    
    # [Keep all evaluation code - lines 920-950]
    best_model_path = models_dir / "best_model_checkpoint.pt"
    if best_model_path.exists():
        print_banner("FINAL EVALUATION (BEST CHECKPOINT)")
        print_kv("Best checkpoint", str(best_model_path))
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print_banner("FINAL EVALUATION (LAST CHECKPOINT)")
        print("No best model checkpoint. Using last state.")

    job_id = os.environ.get("SLURM_JOB_ID", "local")

    print_banner("EXPORTING PREDICTIONS")
    report_and_export_split(va_loader, model, device, plots_dir, job_id, split_name="val")
    report_and_export_split(te_loader, model, device, plots_dir, job_id, split_name="test")
    
    torch.save(model.state_dict(), models_dir / "final_model_delta_learning.pt")
    print(f"Final model saved at {models_dir / 'final_model_delta_learning.pt'}")

if __name__ == "__main__":
    main()