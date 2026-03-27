# Citation Network GNN Explorer

This repository documents a full Cora citation-network workflow:

1. prepare and inspect the dataset,
2. train and compare four GNN families,
3. select stable "champion" models,
4. run extension experiments aimed at deployment and interpretability,
5. package saved artifacts for inspection and app usage.

The project is organized so a reviewer can either:

- inspect precomputed artifacts in `app_data/`, `results/`, and `ml_extensions/`, or
- rerun the pipeline from notebooks and scripts.

## What This Project Is Trying To Show

The core project is not only "can a model classify Cora well?" It is also:

- how different graph architectures behave under the same dataset,
- how to choose a final model based on more than a single lucky run,
- how to reason about inference, explainability, pruning, steering, and mobile deployment.

## Repository Map

| Path | Purpose |
| --- | --- |
| `data_prep/` | Data loading, inspection, and early graph understanding |
| `models/` | Baseline training notebooks for GCN, GAT, GraphSAGE, and APPNP |
| `results/` | CSV exports from grid searches for both split strategies |
| `experiments/` | Champion-model follow-up experiments and diagnostics |
| `ml_extensions/` | Task 1 to Task 5 extension work for deployment and interpretability |
| `app_data/` | Saved weights, tensors, CSVs, and mobile export artifacts |
| `app.py` | Streamlit app for model and graph exploration |

Directory-level guides are included here:

- [ml_extensions/README.md](/Users/emaheshwari/Project2/ml_extensions/README.md)
- [models/README.md](/Users/emaheshwari/Project2/models/README.md)
- [experiments/README.md](/Users/emaheshwari/Project2/experiments/README.md)
- [results/README.md](/Users/emaheshwari/Project2/results/README.md)
- [data_prep/README.md](/Users/emaheshwari/Project2/data_prep/README.md)
- [datascripts/README.md](/Users/emaheshwari/Project2/datascripts/README.md)

## Experimental Design Decisions

### Why two split strategies?

- `60/20/20` was used as the broader exploration regime. It leaves more validation and test signal while tuning many configurations.
- `80/10/10` was used as the stronger final-training regime. It gives the model more supervision once promising settings are known.

This makes the project read naturally as:

```text
torch==2.10.0
torch-geometric==2.7.0
pandas
numpy
scikit-learn
matplotlib
streamlit==1.28.0
pyvis
```
How to install Brew 
```bash 
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Install Python 3.9 version 
```bash
brew install python@3.9
```

### Why compare four architectures?

- `GCN` provides the simplest graph-convolution baseline.
- `GAT` adds learned attention over neighbors and is the natural choice for pruning and attention analysis.
- `GraphSAGE` tests inductive neighborhood aggregation behavior.
- `APPNP` acts as a propagation-focused alternative with different smoothing behavior.

Using four families makes the champion selection more credible than optimizing a single architecture in isolation.

### Why revisit the champions?

Single-run scores on Cora can look better than the model actually is. The later notebooks therefore revisit the strongest candidates with repeat training and extra diagnostics before saving the final artifacts in `app_data/`.

### Why these extension tasks?

The extension tasks were chosen to cover production concerns that do not show up in a normal leaderboard notebook:

- ONNX export and batching for deployment,
- GNNExplainer for interpretability,
- head ablation for compression,
- concept steering for representation analysis,
- quantization and CoreML export for edge deployment.

## Quick Start

### 1. Environment

From the repository root:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Optional task-specific packages used by some extensions:

```bash
python3 -m pip install psutil networkx certifi onnxscript
python3 -m pip install coremltools
```

Notes:

- `coremltools` is only needed for the Task 5 export path and is most practical on macOS.
- `onnxscript` is needed by newer PyTorch ONNX export flows even though `onnxruntime` is already listed in `requirements.txt`.

### 2. Data

The repository already contains local Cora data under `data/Cora/`, so most notebooks and scripts can run without downloading the dataset again.

### 3. Streamlit App

```bash
streamlit run app.py
```

The app uses saved artifacts from `app_data/` and is the fastest way to inspect the trained models without rerunning training.

## Reproducing The Core Pipeline

Run from the repository root.

### Phase 1: Data Familiarization

1. Open [data_prep/1_data.ipynb](/Users/emaheshwari/Project2/data_prep/1_data.ipynb)
2. Open [data_prep/2_data_explore_and_visualisation.ipynb](/Users/emaheshwari/Project2/data_prep/2_data_explore_and_visualisation.ipynb)

This phase explains what Cora looks like before any model choices are made.

### Phase 2: Architecture Sweeps

Use the notebooks in [models/](/Users/emaheshwari/Project2/models/README.md):

- [models/3_baseline_gcn.ipynb](/Users/emaheshwari/Project2/models/3_baseline_gcn.ipynb)
- [models/3_baseline_gcn80.ipynb](/Users/emaheshwari/Project2/models/3_baseline_gcn80.ipynb)
- [models/4_gat_model.ipynb](/Users/emaheshwari/Project2/models/4_gat_model.ipynb)
- [models/4_gat80_model.ipynb](/Users/emaheshwari/Project2/models/4_gat80_model.ipynb)
- [models/5_graphsage_model.ipynb](/Users/emaheshwari/Project2/models/5_graphsage_model.ipynb)
- [models/5_graphsage80_model.ipynb](/Users/emaheshwari/Project2/models/5_graphsage80_model.ipynb)
- [models/6_appnp_model.ipynb](/Users/emaheshwari/Project2/models/6_appnp_model.ipynb)
- [models/6_appnp80_model.ipynb](/Users/emaheshwari/Project2/models/6_appnp80_model.ipynb)

CSV summaries are written to [results/](/Users/emaheshwari/Project2/results/README.md).

### Phase 3: Champion Follow-Up

Run the notebooks in [experiments/](/Users/emaheshwari/Project2/experiments/README.md):

- [experiments/7_champion_experiments.ipynb](/Users/emaheshwari/Project2/experiments/7_champion_experiments.ipynb)
- [experiments/7_champion_expirements2.ipynb](/Users/emaheshwari/Project2/experiments/7_champion_expirements2.ipynb)

These notebooks are where the project moves from "best score seen" to "final model worth keeping."

### Phase 4: Extension Tasks

The deployment and interpretability extensions are documented in [ml_extensions/README.md](/Users/emaheshwari/Project2/ml_extensions/README.md).

## Best Recorded Core Results

The grid-search CSVs in `results/` show the following best test accuracies by file:

| File | Best test accuracy |
| --- | --- |
| `results/60_20_20_split/gcn60_split_grid_search.csv` | `0.8690` |
| `results/60_20_20_split/gat60_split_grid_search.csv` | `0.8875` |
| `results/60_20_20_split/graphsage60_split_grid_search.csv` | `0.8985` |
| `results/60_20_20_split/appnp60_split_grid_search.csv` | `0.8819` |
| `results/80_10_10_split/gcn80_split_grid_search.csv` | `0.8967` |
| `results/80_10_10_split/gat80_split_grid_search.csv` | `0.8893` |
| `results/80_10_10_split/graphsage80_split_grid_search.csv` | `0.8967` |
| `results/80_10_10_split/appnp80_split_grid_search.csv` | `0.9004` |

These numbers are useful for orientation, but the later extension tasks also save their own artifacts and may retrain fresh local variants for a specific purpose.

## Saved Artifacts

The repository keeps important outputs so the work can be inspected without rerunning everything:

- champion weights such as `champion_gcn.pth`, `champion_gat.pth`, and `champion_appnp.pth`,
- graph tensors and CSVs used by the app,
- ONNX export artifacts in `ml_extensions/task1_onnx/`,
- explainer HTMLs in `ml_extensions/task2_explainer/explanations/`,
- pruning logs in `ml_extensions/task3_/`,
- steering vectors and audit tables in `ml_extensions/task4_/`,
- quantized and mobile-export artifacts in `app_data/` and `ml_extensions/task5_/`.

## Reproducibility Notes

This repository is reproducible enough to inspect the project flow and rerun major experiments, but it is important to be precise about what "same result" means here.

- Several notebooks use stochastic training and fresh `RandomNodeSplit` calls. Exact numeric parity is not guaranteed without manually fixing seeds across every notebook.
- Some extension scripts require packages that are not in `requirements.txt` by default. The optional install commands above cover the missing pieces used in this repository.
- Task 5's CoreML export path is intentionally platform-specific because it targets the Apple ecosystem.

The project therefore supports two levels of reproduction:

- `artifact reproduction`: inspect committed outputs and rerun the app,
- `experiment reproduction`: rerun notebooks and scripts to recover the same workflow and similar conclusions.

## Recommended Read Order

If you are reviewing this project for the first time, the clearest order is:

1. [README.md](/Users/emaheshwari/Project2/README.md)
2. [models/README.md](/Users/emaheshwari/Project2/models/README.md)
3. [experiments/README.md](/Users/emaheshwari/Project2/experiments/README.md)
4. [ml_extensions/README.md](/Users/emaheshwari/Project2/ml_extensions/README.md)

That sequence matches the way the work matured from baseline modeling into architecture-specific extensions.
