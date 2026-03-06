# Citation Network GNN Explorer

*Graph Neural Networks on the Cora dataset – from experimentation to a Streamlit dashboard*

---

This repository collects the full research & development pipeline for a project that:

1. **Loads the Cora citation graph**,  
2. **Runs exhaustive hyper‑parameter searches** over four different GNN architectures (GCN, GAT, GraphSAGE, APPNP) using custom 60/20/20 and 80/10/10 train/val/test splits,  
3. **Identifies the “champion” configuration** of each model,  
4. **Extends with robustness/regularisation experiments** (edge dropout, residual links, LR scheduling, etc.),  
5. **Saves the best models and graph data**, and  
6. **Serves an interactive Streamlit app** that lets you visualise the graph, select models and make quick predictions.

The notebooks document every step of the process; the Streamlit app lets you share the results with others without re‑training.

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

## 📁 Project structure

```
Project2/
├── .gitignore                   # ignores envs, data, caches, etc.
├── requirements.txt             # Python deps for deployment
├── app.py                       # Streamlit dashboard
├── .streamlit/                  # Streamlit configuration (theme, logger)
│   └── config.toml
├── app_data/                    #  committed: saved models & graph data
│   ├── champion_appnp.pth
│   ├── champion_gcn.pth
│   ├── champion_gat.pth
│   ├── champion_graphsage.pth
│   ├── cora_features.pt
│   ├── cora_edge_index.pt
│   ├── cora_nodes.csv
│   └── cora_edges.csv
├── data/                        #  ignored raw dataset (Planetoid download)
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

##  How the project was carried out

1. **Environment & data setup** – all notebooks start by bypassing macOS SSL, selecting `mps`/`cuda`/`cpu` device, loading `Planetoid(root='./data/Cora', name='Cora', transform=T.NormalizeFeatures())` and applying `RandomNodeSplit`.

2. **Model definitions** – lightweight PyG modules for

   - `GCN_Standard` (+ residual variant),
   - `FlexibleGAT` (variable layers/heads),
   - `FlexibleGraphSAGE` (variable layers),
   - `APPNPNet` (with propagation and residual MLP).

3. **Grid search loops** – for each architecture, iterate over learning rates, hidden dims, dropouts, (and where appropriate) number of layers or PageRank hops/α. Record test accuracy/F1 for the best val epoch. Early stopping used in some notebooks.

4. **Result aggregation** – results stored in lists, converted to `pandas.DataFrame`, sorted by macro‑F1, thresholds checked (target_acc=0.82, target_f1=0.80), and champions extracted. CSVs saved for each grid search.

5. **Stability tests** – top‑5 configs retrained 5× with different seeds; mean/std of test metrics computed to pick a “true champion”.

6. **Final training & diagnostics** – re‑train the true champion for 200 epochs, plot learning curves, produce classification reports and t‑SNE embeddings of node representations.

7. **Additional experiments** – later notebooks introduce:

   - Feature normalization ablation,
   - Edge dropout data augmentation,
   - Residual links in MLP/GCN,
   - Learning‑rate schedulers (ReduceLROnPlateau),
   - Comparisons via confusion matrices and t‑SNE.

8. **Model saving** – the champion models and graph tensors/CSV metadata are dumped to `app_data/` for deployment.

---

##  Deploying the Streamlit app

The file `app.py` contains:

- definitions of the four GNN classes,
- a `get_model()` helper loading weights from `app_data`,
- sidebar controls (model selection, centre node slider, edge‑weight threshold, etc.),
- functions to run inference and visualise the graph using PyVis,
- miscellaneous UI layout and color maps.

To start locally:

```bash
streamlit run app.py
```

The app will load the saved `.pth` models and the pre‑computed features/edge_index, so it starts instantly.

### GitHub & Streamlit Cloud

1. Ensure `.gitignore` does **not** ignore `app_data/`, but still ignores `data/`.
2. Add/commit everything, including the `app_data` directory:

   ```bash
   git add app.py requirements.txt .streamlit/ app_data/
   git commit -m "Prepare for Streamlit deployment"
   git push origin main
   ```

3. On [share.streamlit.io](https://share.streamlit.io), link your GitHub repository and specify `app.py` as the main file. Streamlit will install dependencies from `requirements.txt` and serve the dashboard.

---

##  Notes

- The raw Cora dataset downloaded by PyG lives in `data/`. It is large and ignored by Git – you don’t need it for deployment.
- The notebooks document the exact hyperparameters tried in the grid searches. Feel free to re‑run them (they can take a few minutes each).
- The Streamlit app supports four architectures and lets you visualise individual nodes, examine neighbours, and compare predicted labels with ground truth.

---

##  Summary

This repository is both a **research log** (via Jupyter notebooks) and a **deployable application**. It demonstrates:

- systematic GNN benchmarking,
- reproducibility through saved model weights,
- and an interactive interface for exploring a citation network.

Use the notebooks to extend experiments; use `app.py` and `requirements.txt` to deploy your own version.