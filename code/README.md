\# RFG-GNN



Reproducible implementation for the revised experiments of RFG-GNN for graph-based financial fraud detection.



The repository contains strict preprocessing, feature encoding, main-model experiments, ablation studies, baseline implementations, and experiment summarization scripts for the Elliptic and T-Finance datasets.



\## 1. Repository Structure



```text

RFG-GNN-GitHub/

├── configs/

│   ├── elliptic\_of\_amnet.yaml

│   └── tfinance.yaml

│

├── train1221\_dtbe\_strict.py

├── train1221\_woe\_strict.py

├── train1221\_iv\_strict.py

├── train1221\_dtbe\_no\_group\_strict.py

├── train1221\_dtbe\_no\_risk\_strict.py

│

├── train\_graphsage\_strict.py

├── train\_gcn\_dtbe\_strict.py

├── train\_mlp\_dtbe\_strict.py

├── train\_bwgnn\_dtbe\_strict.py

├── bwgnn\_strict\_model.py

│

├── build\_elliptic\_strict.py

├── build\_tfinance\_strict.py

├── build\_tfinance\_isolated.py

├── audit\_tfinance\_edges.py

│

├── summarize\_elliptic\_logs.py

├── summarize\_tfinance\_logs.py

│

├── data\_handle.py

├── myutils.py

├── requirements.txt

└── .gitignore

```



\## 2. Environment



The reference environment uses:



\* Python 3.10

\* PyTorch 1.13.1

\* DGL 1.1.2

\* PyTorch Lightning 1.9.5



Create a virtual environment:



```bash

python -m venv .venv

```



Activate it on Windows:



```powershell

.venv\\Scripts\\activate

```



Install dependencies:



```bash

python -m pip install --upgrade pip

python -m pip install -r requirements.txt

```



The supplied `requirements.txt` reproduces the CPU reference environment used for the experiments.



\## 3. Data



Raw datasets are not included in this repository.



Prepare the datasets locally under a `data/` directory.



The repository supports:



\* Elliptic

\* T-Finance



Strict dataset construction scripts are provided to prevent information leakage between training, validation, and test partitions.



\### Elliptic



```bash

python build\_elliptic\_strict.py

```



\### T-Finance



```bash

python build\_tfinance\_strict.py

```



For the isolated T-Finance graph setting:



```bash

python build\_tfinance\_isolated.py

```



The edge-isolation result can be checked with:



```bash

python audit\_tfinance\_edges.py

```



\## 4. Main Experiment



The main RFG-GNN experiment uses DTBE encoding:



```bash

python train1221\_dtbe\_strict.py --config-name elliptic\_of\_amnet

```



For T-Finance:



```bash

python train1221\_dtbe\_strict.py --config-name tfinance

```



\## 5. Feature-Encoding Experiments



\### DTBE



```bash

python train1221\_dtbe\_strict.py --config-name elliptic\_of\_amnet

```



\### WOE



```bash

python train1221\_woe\_strict.py --config-name elliptic\_of\_amnet

```



\### IV



```bash

python train1221\_iv\_strict.py --config-name elliptic\_of\_amnet

```



Replace `elliptic\_of\_amnet` with `tfinance` when running the corresponding T-Finance experiments.



\## 6. Ablation Studies



Without dynamic grouping:



```bash

python train1221\_dtbe\_no\_group\_strict.py --config-name elliptic\_of\_amnet

```



Without the risk-propagation component:



```bash

python train1221\_dtbe\_no\_risk\_strict.py --config-name elliptic\_of\_amnet

```



\## 7. Baselines



\### GraphSAGE



```bash

python train\_graphsage\_strict.py --config-name elliptic\_of\_amnet

```



\### GCN



```bash

python train\_gcn\_dtbe\_strict.py --config-name elliptic\_of\_amnet

```



\### MLP



```bash

python train\_mlp\_dtbe\_strict.py --config-name elliptic\_of\_amnet

```



\### BWGNN



```bash

python train\_bwgnn\_dtbe\_strict.py --config-name elliptic\_of\_amnet

```



The same scripts can be run with:



```text

\--config-name tfinance

```



for T-Finance.



\## 8. Reproducibility



The strict experiment pipeline follows these principles:



1\. Data splitting is performed before supervised feature encoding.

2\. Encoding parameters are fitted using the training split only.

3\. Validation and test samples are transformed using rules determined from the training split.

4\. Random seeds are explicitly controlled.

5\. Baselines use consistent dataset partitions and evaluation procedures.

6\. Experiment outputs should be retained for every independent random-seed run.



For formal reporting, multiple random seeds should be used and results should be reported as mean ± standard deviation.



\## 9. Result Summarization



Summarize Elliptic experiment logs with:



```bash

python summarize\_elliptic\_logs.py

```



Summarize T-Finance experiment logs with:



```bash

python summarize\_tfinance\_logs.py

```



\## 10. Notes



W\&B can be used in offline mode when no online account is required.



On PowerShell:



```powershell

$env:WANDB\_MODE="offline"

```



Large datasets, checkpoints, W\&B logs, and generated experiment outputs are intentionally excluded from version control.



\## License



A license has not yet been specified for this repository.



