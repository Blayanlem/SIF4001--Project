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

\section*{Repository Files}
\begin{itemize}
  \item \texttt{mainscript\_rectified\_v3.py} --- Main training/evaluation entrypoint (argparse CLI). Loads cached graphs, applies optional rotation/jitter augmentation, trains the equivariant model, saves best checkpoint, and exports prediction CSVs/metrics.
  \item \texttt{preprocess\_and\_cache\_HPC1.py} --- Builds PyTorch Geometric \texttt{Data} graphs from \texttt{meta.xlsx} + PDB/NPZ inputs and saves a \texttt{.pt} cache with metadata.
  \item \texttt{improved\_loss\_functions\_rectified\_v2.py} --- Loss functions (log-target aware) with distance weighting and negative-$V_{\mathrm{ref}}$ penalties.
  \item \texttt{run3.sh} --- Example SLURM script to run preprocessing (if cache missing) and then training on an A100 partition.
\end{itemize}

\section*{Data Layout Expected}
Your dataset root directory is expected to contain:
\begin{itemize}
  \item \texttt{meta.xlsx} with columns like \texttt{molecule a}, \texttt{molecule b}, \texttt{npz}, \texttt{r}, $\theta$, \texttt{v(0)}, \texttt{vref} (case-insensitive after normalization)
  \item \texttt{\{id\}.pdb} files for each molecule ID
  \item \texttt{.npz} files referenced by the \texttt{npz} column (containing Mulliken features, dipole, quadrupole, participation ratio)
\end{itemize}

\section*{Typical Workflow}
\subsection*{Preprocess + Cache Graphs}
\begin{verbatim}
python preprocess_and_cache_HPC1.py /path/to/data/molecule/
\end{verbatim}
This writes a cache like \texttt{cached\_graphs.pt} (or whatever you set) containing \texttt{\{graphs, metadata\}}.

\subsection*{Train}
\begin{verbatim}
python mainscript_rectified_v3.py --root-dir /path/to/data/molecule --output-dir ./outputs --cache-file-name cached_graphs.pt
\end{verbatim}
Training creates per-run subfolders for plots and model checkpoints and performs evaluation using the best saved checkpoint when available.

\subsection*{HPC (SLURM)}
\begin{verbatim}
sbatch run3.sh
\end{verbatim}
The script checks for a cache file; if missing it runs preprocessing, then launches training with your selected hyperparameters/loss flags.
