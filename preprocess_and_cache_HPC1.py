
from __future__ import annotations

from pathlib import Path
import math
import torch
from tqdm import tqdm
from e3nn import o3
import pandas as pd
import numpy as np

from torch_geometric.data import Data, Dataset as PyGDataset
from torch_geometric.nn import radius_graph

# Default dtype set early (as in main script)
torch.set_default_dtype(torch.float32)
DEFAULT_TORCH_DTYPE_PREPROCESS = torch.get_default_dtype()

# Physical constants (should match main script)
NM_TO_PHYSICAL_ANGSTROM_PREPROCESS = 10.0
DEBYE_TO_E_PHYSICAL_ANGSTROM_PREPROCESS = 0.20819434


# Quadrupole ordering convention:
#   Gaussian FCHK stores: [XX, YY, ZZ, XY, XZ, YZ]
#   Training/model expects: [XX, XY, XZ, YY, YZ, ZZ]
GAUSSIAN_FCHK_QUAD_TO_TRAINING_IDX = np.array([0, 3, 4, 1, 5, 2], dtype=np.int64)
PT_PREPROCESS = {}
for Z, sym in enumerate(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca".split(), 1):
    PT_PREPROCESS[sym] = Z

IRREPS_EDGE_SH_PRECOMPUTE_PREPROCESS = o3.Irreps("4x0e+4x1o+4x2e+4x3o") # Should match main script's IRREPS_EDGE_SH_PRECOMPUTE

@torch.no_grad()
def transform_by_matrix_preprocess(
    irreps: o3.Irreps, 
    feats: torch.Tensor, 
    rotation: torch.Tensor, 
    *, 
    check: bool = False
) -> torch.Tensor:
    if rotation.shape != (3, 3): raise ValueError("`rotation` must have shape (3, 3)")
    original_shape = feats.shape
    if feats.ndim == 0 or feats.shape[-1] != irreps.dim:
        raise ValueError(f"Last dimension of feats ({feats.shape[-1]}) must match irreps.dim ({irreps.dim})")
    
    # Reshape to [batch_dims_combined, irreps.dim]
    feats_matrix = feats.reshape(-1, irreps.dim) if feats.ndim > 1 else feats.unsqueeze(0)

    D = irreps.D_from_matrix(rotation)
    if check and not torch.allclose(D @ D.T, torch.eye(D.shape[0], dtype=D.dtype, device=D.device), atol=1e-4):
        print("Warning: D @ D.T is not close to identity in transform_by_matrix_preprocess.")
    
    transformed_feats_matrix = torch.matmul(feats_matrix, D.T) # Use matmul
    
    return transformed_feats_matrix.reshape(*original_shape[:-1], irreps.dim) if feats.ndim > 1 else transformed_feats_matrix.squeeze(0)


def decompose_quadrupole_to_real_spherical_preprocess(q_vec: torch.Tensor) -> torch.Tensor:
    if q_vec.ndim < 1 or q_vec.shape[-1] != 6: raise ValueError(f"Expected last dimension = 6, got {q_vec.shape}")
    Qxx,Qxy,Qxz,Qyy,Qyz,Qzz = q_vec[...,0],q_vec[...,1],q_vec[...,2],q_vec[...,3],q_vec[...,4],q_vec[...,5]
    tr = (Qxx + Qyy + Qzz) / 3.0
    Qxx_t,Qyy_t,Qzz_t = Qxx-tr, Qyy-tr, Qzz-tr
    c0 = (2*Qzz_t - Qxx_t - Qyy_t)/math.sqrt(6.0); c1=math.sqrt(2.0)*Qxz; c2=math.sqrt(2.0)*Qyz
    c3 = (Qxx_t - Qyy_t)/math.sqrt(2.0); c4=math.sqrt(2.0)*Qxy
    return torch.stack([c0,c1,c2,c3,c4],dim=-1).to(q_vec.device)

def read_pdb_preprocess(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    elems, xyz = [], []
    if not path.exists(): raise FileNotFoundError(f"PDB missing: {path}")
    with path.open() as fh:
        for line_idx, line in enumerate(fh):
            if line.startswith(("ATOM  ", "HETATM")):
                atom_name_field = line[12:16].strip(); elem_symbol = ""
                if len(line) >= 78:
                    elem_field_std = line[76:78].strip()
                    if elem_field_std:
                        cap_elem_field_std = elem_field_std[0].upper()+(elem_field_std[1:].lower() if len(elem_field_std)>1 else "")
                        if cap_elem_field_std in PT_PREPROCESS: elem_symbol = cap_elem_field_std
                if (not elem_symbol or elem_symbol not in PT_PREPROCESS) and atom_name_field:
                    if len(atom_name_field) >= 2 and atom_name_field[0].isalpha():
                        potential_sym_2_cand = atom_name_field[:2].capitalize() if atom_name_field[1].islower() else atom_name_field[0].capitalize() + atom_name_field[1].lower()
                        if potential_sym_2_cand in PT_PREPROCESS: elem_symbol = potential_sym_2_cand
                    if not elem_symbol and atom_name_field[0].isalpha():
                        potential_sym_1 = atom_name_field[0].capitalize()
                        if potential_sym_1 in PT_PREPROCESS: elem_symbol = potential_sym_1
                if elem_symbol and elem_symbol in PT_PREPROCESS:
                    elems.append(PT_PREPROCESS[elem_symbol])
                    try: xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    except ValueError: raise ValueError(f"PDB coord parse error: {path} L{line_idx+1}")
    if not elems: raise ValueError(f"No atoms in PDB: {path}")
    return torch.tensor(elems, dtype=torch.long), torch.tensor(xyz, dtype=DEFAULT_TORCH_DTYPE_PREPROCESS)

class DimerDatasetForCache(PyGDataset):
    def __init__(self, root: Path, internal_cutoff: float, internal_angstrom_scale: float):
        super().__init__(root=None, transform=None, pre_transform=None, pre_filter=None)
        self.root_path = Path(root).resolve()
        self.internal_cutoff = internal_cutoff
        self.internal_angstrom_scale = internal_angstrom_scale # Store for use
        meta_file_path = self.root_path / "meta.xlsx"
        if not meta_file_path.exists(): raise FileNotFoundError(f"meta.xlsx missing: {meta_file_path}")
        self.df = pd.read_excel(meta_file_path)
        self.df.columns = self.df.columns.str.strip().str.lower()
        required_cols = ["molecule a", "molecule b", "npz", "r", "θ", "v(0)", "vref"]
        for col in required_cols:
            if col not in self.df.columns: raise ValueError(f"Missing column '{col}'")

    def __len__(self) -> int: return len(self.df)

    def __getitem__(self, idx: int) -> Data:
        row = self.df.iloc[idx]
        ida,idb,npz_filename = int(row["molecule a"]),int(row["molecule b"]),str(row["npz"])
        pdb_a_path,pdb_b_path = self.root_path/f"{ida}.pdb", self.root_path/f"{idb}.pdb"
        Za, Xa_phys = read_pdb_preprocess(pdb_a_path)
        Zb, Xb_phys = read_pdb_preprocess(pdb_b_path)
        npz_path = self.root_path / npz_filename
        if not npz_path.exists(): raise FileNotFoundError(f"NPZ missing: {npz_path}")
        with np.load(npz_path) as data_npz:
            mulliken_spd = data_npz['mulliken_spd'].astype(np.float32)
            pr = float(data_npz['participation_ratio'])
            dipole_np = data_npz['dipole_vector'].astype(np.float32)
            quad_np = data_npz['quadrupole'].astype(np.float32)
        if dipole_np.shape != (3,): dipole_np = np.zeros(3,dtype=np.float32) if dipole_np.shape==() and np.isscalar(dipole_np.item()) else (_ for _ in ()).throw(ValueError("Dipole shape error"))
        if quad_np.shape != (6,): raise ValueError("Quadrupole shape error")
        
        # Reorder quadrupole from Gaussian FCHK convention to training/model convention
        quad_np = quad_np[GAUSSIAN_FCHK_QUAD_TO_TRAINING_IDX]
        Xa_phys, Xb_phys = Xa_phys.to(DEFAULT_TORCH_DTYPE_PREPROCESS), Xb_phys.to(DEFAULT_TORCH_DTYPE_PREPROCESS)
        Xa_ctr_phys,Xb_ctr_phys = Xa_phys-Xa_phys.mean(0,keepdim=True), Xb_phys-Xb_phys.mean(0,keepdim=True)
        Xa_int,Xb_int = Xa_ctr_phys*self.internal_angstrom_scale, Xb_ctr_phys*self.internal_angstrom_scale
        R_nm,theta_deg,V0,Vref = float(row["r"]),float(row["θ"]),float(row["v(0)"]),float(row["vref"])
        R_phys_A = R_nm * NM_TO_PHYSICAL_ANGSTROM_PREPROCESS
        R_int_shift = R_phys_A * self.internal_angstrom_scale
        Xb_int += torch.tensor([0.,0.,R_int_shift], dtype=DEFAULT_TORCH_DTYPE_PREPROCESS)
        pos_atoms_int, Z_atoms_all = torch.cat([Xa_int,Xb_int]), torch.cat([Za,Zb])
        n_atoms = Z_atoms_all.shape[0]
        if mulliken_spd.shape[0]!=n_atoms: raise ValueError("Mulliken vs PDB atom count mismatch")
        
        s,p = torch.from_numpy(mulliken_spd[:,0]).unsqueeze(-1), torch.from_numpy(mulliken_spd[:,1]).unsqueeze(-1)
        d = torch.from_numpy(mulliken_spd[:,2]).unsqueeze(-1) if mulliken_spd.shape[1]>2 else torch.zeros_like(s)
        node_attr = torch.cat([s,p,d],dim=1)

        dip_D = torch.from_numpy(dipole_np); dip_eA = dip_D*DEBYE_TO_E_PHYSICAL_ANGSTROM_PREPROCESS
        dip_int = dip_eA * self.internal_angstrom_scale 
        quad_DA = torch.from_numpy(quad_np.flatten()); quad_eA2 = quad_DA*DEBYE_TO_E_PHYSICAL_ANGSTROM_PREPROCESS
        quad_int_presph = quad_eA2 * (self.internal_angstrom_scale**2)
        field_feat_unrot = torch.cat([dip_int, decompose_quadrupole_to_real_spherical_preprocess(quad_int_presph)])
        
        glob_feat = torch.tensor([R_int_shift, math.radians(theta_deg), V0, pr], dtype=DEFAULT_TORCH_DTYPE_PREPROCESS)
        delta_v, y_vref = torch.tensor([Vref-V0]), torch.tensor([Vref])
        
        edge_idx = radius_graph(pos_atoms_int,r=self.internal_cutoff,loop=False,flow='source_to_target',max_num_neighbors=1000)
        if edge_idx.dtype != torch.long: edge_idx = edge_idx.long()
        edge_dist = (pos_atoms_int[edge_idx[1]]-pos_atoms_int[edge_idx[0]]).norm(dim=1,keepdim=True) if edge_idx.numel()>0 else torch.empty((0,1),dtype=DEFAULT_TORCH_DTYPE_PREPROCESS)
        
        data = Data(z_atoms=Z_atoms_all, pos_atoms=pos_atoms_int, atomic_node_attr=node_attr,
                    edge_index_atoms=edge_idx, edge_attr_atoms=edge_dist,
                    field_node_features=field_feat_unrot, u=glob_feat, y=delta_v,
                    y_true_vref=y_vref, R_internal_val=torch.tensor([R_int_shift]),
                    id_a=ida, id_b=idb, npz_file=npz_filename, num_atomic_nodes=n_atoms)

        if data.edge_index_atoms.numel() > 0:
            rel_pos = data.pos_atoms[data.edge_index_atoms[1]] - data.pos_atoms[data.edge_index_atoms[0]]
            data.edge_attr_sh = o3.spherical_harmonics(
                IRREPS_EDGE_SH_PRECOMPUTE_PREPROCESS, rel_pos, normalize=True, normalization='component'
            )
        else:
            data.edge_attr_sh = torch.empty(
                (0, IRREPS_EDGE_SH_PRECOMPUTE_PREPROCESS.dim),
                dtype=data.pos_atoms.dtype if data.pos_atoms.numel() > 0 else DEFAULT_TORCH_DTYPE_PREPROCESS,
                device=data.pos_atoms.device if data.pos_atoms.numel() > 0 else torch.device('cpu')
            )
        return data

def build_cache(root_dir_path: Path, physical_cutoff_val: float, outfile_name: str | Path, 
                internal_angstrom_scale: float, default_torch_dtype: torch.dtype) -> None:
    root_dir_path = root_dir_path.expanduser().resolve()
    actual_outfile_path  = root_dir_path / outfile_name
    internal_cutoff_for_cache = physical_cutoff_val * internal_angstrom_scale

    print(f"Initializing DimerDatasetForCache from: {root_dir_path} with internal_cutoff: {internal_cutoff_for_cache}")
    dataset  = DimerDatasetForCache(root_dir_path, internal_cutoff_for_cache, internal_angstrom_scale)
    cached_graphs   = []
    print(f"Found {len(dataset)} items in meta.xlsx to process for caching.")

    for i in tqdm(range(len(dataset)), desc="Preprocessing and Caching Graphs"):
        try:
            data_item = dataset[i]
        except Exception as e:
            print(f"Error processing item {i} for caching: {e}. Skipping this item.")
            import traceback; traceback.print_exc()
            continue
        cached_graphs.append(data_item)

    if not cached_graphs:
        print("No graphs were successfully processed. Output file will not be created.")
        return
    
    metadata_to_save = {
        "INT_ANGSTROM_SCALE": internal_angstrom_scale,
        "internal_cutoff_val": internal_cutoff_for_cache,
        "physical_cutoff_val": physical_cutoff_val,
        "IRREPS_EDGE_SH_PRECOMPUTE": str(IRREPS_EDGE_SH_PRECOMPUTE_PREPROCESS)
    }
    data_to_save = {"graphs": cached_graphs, "metadata": metadata_to_save}
    torch.save(data_to_save, actual_outfile_path)
    print(f"Wrote {len(cached_graphs)} graphs and metadata to {actual_outfile_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
    	param_root_dir_str = sys.argv[1]
    else:
    	param_root_dir_str = "/scr/user/blayanlem/FYP/HK4/data/molecule/"

    param_physical_cutoff = 12.0 
    param_outfile = "cached_graphs.pt"
    param_internal_angstrom_scale = 0.1 

    param_root_dir = Path(param_root_dir_str)

    print(f"Running preprocess_and_cache.py with parameters set for Spyder:")
    print(f"  Root Directory: {param_root_dir}")
    print(f"  Physical Cutoff: {param_physical_cutoff}")
    print(f"  Internal Angstrom Scale: {param_internal_angstrom_scale}")
    print(f"  Output File: {param_outfile}")
    
    if not param_root_dir.exists():
        print(f"Error: Root directory '{param_root_dir}' does not exist.")
    else:
        build_cache(param_root_dir, param_physical_cutoff, param_outfile, 
                    param_internal_angstrom_scale, DEFAULT_TORCH_DTYPE_PREPROCESS)