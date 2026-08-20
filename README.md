# RFG-GNN
Code,data and experiment logs for paper submission
Repository Structure
RFG-GNN/
├── code/                   # Source code
├── elliptic_log/           # Experiment logs for the Elliptic dataset
├── tfinance_log/           # Experiment logs for the T-Finance dataset
├── elliptic_of_amnet_strict.dgldata
├── tfinance_strict_scaler.npz
├── tfinance_strict_split.npz
├── result/
└── README.md
#Datasets
Experiments are conducted on two financial fraud detection datasets:
Elliptic
T-Finance
The repository contains processed data and split-related files required for reproducing the experiments.

#Environment
The experiments were tested with the following main dependencies:
Python 3.10
PyTorch
DGL
PyTorch Lightning
scikit-learn
pandas
NumPy
Hydra
Weights & Biases
For the exact package versions, please refer to the environment or requirements file provided in the source-code directory.

#Running the Experiments
Enter the source-code directory:
cd code
Run the corresponding training script with the desired dataset configuration.
Example:
python train1221.py --config-name elliptic_of_amnet
Please check the configuration files in the configs/ directory before running an experiment.

#Experiment Logs
Experimental logs are organized by dataset:
elliptic_log/
tfinance_log/
The logs contain the outputs of the corresponding experimental runs and can be used to verify the reported results.

#Reproducibility
To improve experimental reproducibility:
Dataset splitting and preprocessing are performed before model evaluation.
Feature encoding parameters are fitted using the training data.
Validation and test data are transformed using parameters determined from the training data.
Experiment configurations and logs are retained for result verification.

#Citation
If this repository is useful for your research, please cite the corresponding paper.

@article{rfggnn,
  title   = {RFG-GNN},
  author  = {Anonymous},
  journal = {Under Review},
  year    = {2026}
}
#Note

This repository is provided for academic research and paper reproducibility.
