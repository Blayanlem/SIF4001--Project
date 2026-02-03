# SIF4001--Project
This is the repository which contains all my code for my final year research, titled:

PHYSICS-GUIDED Δ-MACHINE LEARNING OF ELECTRONIC COUPLINGS IN MOLECULAR DIMERS

Supervisor:
Assoc. Prof. Dr. Woon Kai Lin

Low Dimensional Materials Research Centre,
Department of Physics, 
Universiti Malaya,
50603 Kuala Lumpur,
Malaysia

This repository trains an **E(3)-equivariant** graph neural network to predict a delta target between a reference coupling and a baseline value stored in the dataset.

We model the delta:
$$
\Delta V = V_{\mathrm{ref}} - V_{0},
$$
where:
- $V_{\mathrm{ref}} \ge 0$ is the (non-negative) reference coupling magnitude
- $V_0$ is the baseline value provided in the dataset

Optionally, training can run in a **log-target mode** to stabilize learning across wide magnitude ranges:
$$
\Delta \log |V| = \log(|V_{\mathrm{ref}}| + \varepsilon) - \log(|V_{0}| + \varepsilon),
$$
and the loss reconstructs $V_{\mathrm{ref}}$ consistently from the predicted delta.

---

## Repository Files

- **`mainscript_rectified_v3.py`**  
  Main training/evaluation entrypoint (CLI via `argparse`). Loads cached graphs, applies optional rotation/jitter augmentation, trains the equivariant model, saves the best checkpoint, and exports prediction CSVs + metrics.

- **`preprocess_and_cache_HPC1.py`**  
  Preprocessing pipeline that builds **PyTorch Geometric** `Data` graphs from `meta.xlsx` + PDB/NPZ inputs and writes a `.pt` cache containing `{graphs, metadata}`.

- **`improved_loss_functions_rectified_v2.py`**  
  Collection of loss functions (log-target aware), including distance weighting and penalties for unphysical negative $V_{\mathrm{ref}}$ predictions.

- **`run3.sh`**  
  Example **SLURM** script for HPC usage. Runs preprocessing if the cache is missing, then launches training (configured for an A100 partition in the script).

---

## Expected Data Layout

Your dataset root directory should contain:

- `meta.xlsx` with columns such as (case-insensitive after normalization):
  - `molecule a`, `molecule b`
  - `npz`
  - `r`, `θ`
  - `v(0)`, `vref`

- `{id}.pdb` files for each molecule ID

- `.npz` files referenced by the `npz` column containing features such as:
  - Mulliken features
  - dipole
  - quadrupole
  - participation ratio

---

## Typical Workflow

### 1) Preprocess + Cache Graphs

```bash
python preprocess_and_cache_HPC1.py /path/to/data/molecule/
