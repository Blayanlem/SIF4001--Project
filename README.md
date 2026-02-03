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

This repo trains an E(3)-equivariant graph neural network to predict $\DeltaV = V_{\mathrm{ref}} − V_0$, where Vref is a non-negative coupling magnitude and V0 is a baseline value stored in the dataset. Training can be done in log-target mode () to stabilize learning across magnitudes, with losses that reconstruct Vref consistently.
