# Citation Network GNN Explorer

*Graph Neural Networks on the Cora dataset – from experimentation to a Streamlit dashboard*

---

This repository collects the full research & development pipeline for a project that:

1. **Loads the Cora citation graph**,  
2. **Runs exhaustive hyper‑parameter searches** over four different GNN architectures (GCN, GAT, GraphSAGE, APPNP) using custom 60/20/20 and 80/10/10 train/val/test splits,  
3. **Identifies the "champion" configuration** of each model,  
4. **Extends with robustness/regularisation experiments** (edge dropout, residual links, LR scheduling, etc.),  
5. **Saves the best models and graph data**, and  
6. **Serves an interactive Streamlit app** that lets you visualise the graph, select models and make quick predictions.

The notebooks document every step of the process; the Streamlit app lets you share the results with others without re‑training.

---

## 🚀 Live Streamlit App

Access the interactive dashboard here: **[Streamlit App](https://gnn-citation-explorer.streamlit.app/)**

The app loads instantly with pre-trained models and visualizes the Cora citation network in real-time.

### Features:
- **Four GNN Models**: APPNP, GAT, GraphSAGE, GCN
- **Interactive Graph Visualization**: Explore the Cora citation network
- **Real-time Predictions**: Select a model and inspect node classifications
- **Color-coded Topics**: 7 research topics with distinct colors

### Deployment:
- Repository linked to Streamlit Cloud via GitHub
- Automatic deployments on `git push`
- Models and graph data loaded from `app_data/`

---

## 🔧 Prerequisites

- macOS (tested on a Mac with `mps` support; GPU/CPU also OK)
- Python 3.11+ (virtual environment recommended)
- GitHub account (for deployment)

Requirements are listed in `requirements.txt`:

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

Install with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
Project2/
├── .gitignore                   # ignores envs, data, caches, etc.
├── requirements.txt             # Python deps for deployment
├── README.md                    # This file
├── app.py                       # Streamlit dashboard
├── .streamlit/                  # Streamlit configuration (theme, logger)
│   └── config.toml
├── app_data/                    # ✅ committed: saved models & graph data
│   ├── champion_appnp.pth
│   ├── champion_gcn.pth
│   ├── champion_gat.pth
│   ├── champion_graphsage.pth
│   ├── cora_features.pt
│   ├── cora_edge_index.pt
│   ├── cora_nodes.csv
│   └── cora_edges.csv
├── data/                        # ❌ ignored raw dataset (Planetoid download)
├── 4_gat_model.ipynb            # 60/20/20 split GAT experiments
├── 4_gat80_model.ipynb          # 80/10/10 split GAT experiments
├── 5_graphsage_model.ipynb      # 60/20/20 GraphSAGE experiments
├── 5_graphsage80_model.ipynb    # 80/10/10 GraphSAGE experiments
├── 6_appnp_model.ipynb          # APPNP hyper‑search & champion selection
├── 7_champion_experiments.ipynb # final APPNP tuning & diagnostics
├── 7_champion_expirements2.ipynb# GCN champion robustness experiments
└── … other notebooks (early trials, deleted versions)
```

---

## 🧠 How the Project Was Carried Out

### 1. Environment & Data Setup
All notebooks start by:
- Bypassing macOS SSL certificates
- Selecting `mps`/`cuda`/`cpu` device
- Loading `Planetoid(root='./data/Cora', name='Cora', transform=T.NormalizeFeatures())`
- Applying `RandomNodeSplit` with custom train/val/test splits

### 2. Model Definitions
Lightweight PyTorch Geometric modules for:
- **GCN_Standard** (+ residual variant)
- **FlexibleGAT** (variable layers/heads)
- **FlexibleGraphSAGE** (variable layers)
- **APPNPNet** (with propagation and residual MLP)

### 3. Grid Search Loops
For each architecture:
- Iterate over learning rates, hidden dims, dropouts
- Vary number of layers or PageRank hops/α
- Record test accuracy/F1 for best val epoch
- Early stopping applied where appropriate

### 4. Result Aggregation
- Store results in lists
- Convert to `pandas.DataFrame`
- Sort by macro‑F1
- Check thresholds (target_acc=0.82, target_f1=0.80)
- Extract champions and save CSVs

### 5. Stability Testing
- Re-train top‑5 configs 5× with different seeds
- Compute mean/std of test metrics
- Pick "true champion" based on consistency

### 6. Final Training & Diagnostics
- Re‑train true champion for 200 epochs
- Plot learning curves
- Produce classification reports
- Generate t‑SNE embeddings of node representations

### 7. Additional Experiments
Later notebooks introduce:
- Feature normalization ablation
- Edge dropout data augmentation
- Residual links in MLP/GCN
- Learning‑rate schedulers (ReduceLROnPlateau)
- Comparisons via confusion matrices and t‑SNE

### 8. Model Saving
- Save champion models to `app_data/*.pth`
- Dump graph tensors to `app_data/*.pt`
- Save node/edge metadata as CSV

---

## 📦 What Each File Contains

### Jupyter Notebooks (Research & Development)

| Notebook | Purpose | Train/Val/Test Split |
|----------|---------|----------------------|
| `4_gat_model.ipynb` | GAT grid search, stability, diagnostics | 60/20/20 |
| `4_gat80_model.ipynb` | GAT grid search, stability, diagnostics | 80/10/10 |
| `5_graphsage_model.ipynb` | GraphSAGE grid search, stability, diagnostics | 60/20/20 |
| `5_graphsage80_model.ipynb` | GraphSAGE grid search, stability, diagnostics | 80/10/10 |
| `6_appnp_model.ipynb` | APPNP hyper-search, selection, validation | 80/10/10 |
| `7_champion_experiments.ipynb` | APPNP final training, normalization ablation, edge dropout, residual links, LR scheduling | 80/10/10 |
| `7_champion_expirements2.ipynb` | GCN champion robustness, residual variants, diagnostics | 80/10/10 |

### Application Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit dashboard with model selection, graph visualization, and inference |
| `.streamlit/config.toml` | Streamlit theme and logger configuration |
| `requirements.txt` | Python dependencies for deployment |
| `.gitignore` | Git ignore rules for clean repository |

### Data Directories

| Directory | Purpose | Git Status |
|-----------|---------|-----------|
| `app_data/` | Saved model weights, graph tensors, and metadata | ✅ Committed |
| `data/` | Raw Cora dataset (auto-downloaded by PyG) | ❌ Ignored |

---

## 🎯 Key Results

### Champion Models (80/10/10 Split)

| Model | LR | Hidden Dim | Dropout | Layers/Hops | Test Accuracy | Macro-F1 |
|-------|----|----|---------|----------|---------------|----------|
| APPNP | 0.01 | 64 | 0.3 | K=10, α=0.1 | ~90.77% | ~91.11% |
| GAT | 0.01 | 64 | 0.3 | 2-4 | ~88.1% | ~87% |
| GraphSAGE | 0.01 | 64 | 0.3 | 2-4 | ~88.6% | ~89% |
| GCN | 0.01 | 64 | 0.3 | 2 | ~90.77% | ~90.60% |

*Note: Exact values vary by seed; see notebooks for full results.*

---

## 💾 Cora Dataset Overview

- **Nodes**: 2,708 papers
- **Edges**: 5,429 citations
- **Features**: 1,433 bag-of-words
- **Classes**: 7 research topics
  - Theory
  - Reinforcement Learning
  - Genetic Algorithms
  - Neural Networks
  - Probabilistic Methods
  - Case Based
  - Rule Learning
- **Size**: ~2.8 MB (ignored in repo)

---

## 📝 Notes

- The raw Cora dataset lives in `data/` and is ignored by Git – you don't need it for deployment
- The notebooks document exact hyperparameters for all grid searches; re-run them to extend experiments
- The Streamlit app supports four architectures and lets you visualise individual nodes, examine neighbours, and compare predicted labels with ground truth
- All saved models in `app_data/` are committed to GitHub for instant deployment

---

## 🏁 Summary

This repository is both a **research log** (via Jupyter notebooks) and a **deployable application**. It demonstrates:

- Systematic GNN benchmarking across four architectures
- Reproducibility through saved model weights
- Interactive interface for exploring a citation network
- Production-ready Streamlit deployment

Use the notebooks to extend experiments; use the Streamlit app to share results with collaborators.

---

## 🔬 Running All Experiments

To reproduce all results from scratch, you can run all notebooks sequentially. This will:
- Perform grid searches for each architecture
- Run stability tests
- Generate final diagnostics and visualizations
- Save all champion models to `app_data/`

```bash
# 60/20/20 Split Experiments
jupyter notebook 3_baseline_gcn.ipynb
jupyter notebook 4_gat_model.ipynb
jupyter notebook 5_graphsage_model.ipynb
jupyter notebook 6_appnp_model.ipynb

# 80/10/10 Split Experiments
jupyter notebook 3_baseline_gcn90.ipynb
jupyter notebook 4_gat80_model.ipynb
jupyter notebook 5_graphsage80_model.ipynb
jupyter notebook 6_appnp_model80.ipynb

# Champion & Robustness Experiments
jupyter notebook 7_champion_experiments.ipynb
jupyter notebook 7_champion_expirements2.ipynb
```
## 📧 Questions or Contributions?

Feel free to open issues or submit pull requests. For deployment help, refer to [Streamlit Cloud documentation](https://docs.streamlit.io/streamlit-cloud).

Happy exploring! 🚀
