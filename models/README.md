# Model Training Notebooks

This folder contains the baseline model-development notebooks for the four core architectures used in the project.

## Why The Notebooks Are Split This Way

Each architecture is trained under two data split regimes:

- `60/20/20` for broader exploration,
- `80/10/10` for stronger final-training conditions.

This structure makes it easier to distinguish exploratory tuning from final candidate selection.

## Notebook Map

| Notebook | Architecture | Split | Purpose |
| --- | --- | --- | --- |
| `3_baseline_gcn.ipynb` | GCN | 60/20/20 | Baseline GCN search and evaluation |
| `3_baseline_gcn80.ipynb` | GCN | 80/10/10 | Higher-training-data GCN sweep |
| `4_gat_model.ipynb` | GAT | 60/20/20 | GAT exploration and tuning |
| `4_gat80_model.ipynb` | GAT | 80/10/10 | GAT under the final split regime |
| `5_graphsage_model.ipynb` | GraphSAGE | 60/20/20 | GraphSAGE sweep |
| `5_graphsage80_model.ipynb` | GraphSAGE | 80/10/10 | GraphSAGE under the final split regime |
| `6_appnp_model.ipynb` | APPNP | 60/20/20 | APPNP exploration |
| `6_appnp80_model.ipynb` | APPNP | 80/10/10 | APPNP under the final split regime |

## Experimental Decisions

### Why these four architectures?

- GCN gives the simplest spectral-style baseline.
- GAT tests whether attention improves behavior on citation neighborhoods.
- GraphSAGE provides a different aggregation family.
- APPNP separates prediction from propagation and is useful for oversmoothing comparisons.

### Why notebook-based sweeps?

The project was developed as a research workflow first. The notebooks keep plots, metric tables, and commentary close to the code that generated them, which makes design decisions easier to justify during review.

## Outputs

The main outputs from these notebooks are the CSVs in [results/README.md](/Users/emaheshwari/Project2/results/README.md) plus saved model artifacts later promoted into `app_data/`.
