# RFG-GNN
Code,data and experiment logs for paper submission

#The repository contains the following directories and files:
code/ — Source code for model training and evaluation.
elliptic_log/ — Experiment logs for the Elliptic dataset.
tfinance_log/ — Experiment logs for the T-Finance dataset.
elliptic_of_amnet_strict.dgldata — Processed Elliptic dataset used in the experiments.
tfinance_strict_scaler.npz — Preprocessing/scaling information for the T-Finance dataset.
tfinance_strict_split.npz — Train/validation/test split information for the T-Finance dataset.
result/ — Experimental results.
README.md — Repository documentation.

#Datasets
Experiments are conducted on two financial fraud detection datasets:
Elliptic
T-Finance
The repository contains the processed data and split-related files required for reproducing the experiments.

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
Run the corresponding training script with the desired dataset configuration. For example:
python train1221.py --config-name elliptic_of_amnet
Please check the configuration files in the configs/ directory before running an experiment.

#Experiment Logs
Experimental logs are organized by dataset:
elliptic_log/ — Elliptic experiment logs.
tfinance_log/ — T-Finance experiment logs.
These logs contain outputs from the corresponding experimental runs and can be used to verify the reported results.

#Reproducibility
To improve experimental reproducibility:
Dataset splitting and preprocessing are performed before model evaluation.
Training-dependent preprocessing parameters are fitted using the training data.
Dataset split files are provided where applicable.
Experimental logs are retained for verification of the reported results.
Configuration files specify the main experimental settings.

For reproduction, please use the provided processed datasets, split files, source code, and corresponding configuration files.
