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

This repo trains an E(3)-equivariant graph neural network to predict $\Delta V = V_{\mathrm{ref}} - V_0$, where $V_{\mathrm{ref}}$ is a non-negative coupling magnitude and $V_0$ is a baseline value stored in the dataset. Training can be done in log-target mode ($\Delta \log |V|$) to stabilize learning across magnitudes, with losses that reconstruct $V_{\mathrm{ref}}$ consistently.

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
```
This writes a cache like \texttt{cached\_graphs.pt} (or whatever you set) containing \texttt{\{graphs, metadata\}}.

### 2) Train
```bash
python mainscript_rectified_v3.py --root-dir /path/to/data/molecule --output-dir ./outputs --cache-file-name cached_graphs.pt
```
Training creates per-run subfolders for plots and model checkpoints and performs evaluation using the best saved checkpoint when available.

### 3) HPC (SLURM)
```bash
sbatch run3.sh
```
The script checks for a cache file; if missing it runs preprocessing, then launches training with your selected hyperparameters/loss flags.
