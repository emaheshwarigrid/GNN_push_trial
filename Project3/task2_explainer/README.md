# Task 2: GNNExplainer For Citation Pattern Interpretation

## Objective

This task studies how the model uses the citation graph when classifying Cora papers. The deliverable is not just an explainer run, but a comparison between a graph-convolution model and an attention-based model on the same kinds of difficult nodes.

## Experimental Decisions

### Why compare GCN and GAT?

GCN and GAT differ in how they aggregate neighbors:

- GCN smooths over neighborhoods more uniformly.
- GAT can reweight neighbors through learned attention.

That makes them a strong pair for interpretability comparison because they should react differently to noisy or heterophilous citation neighborhoods.

### Why mine archetypal nodes instead of choosing random examples?

The script `2a_explainer_miner.py` deliberately selects informative cases:

- confident correct predictions,
- confident mistakes,
- narrow-margin wins,
- heterophily hubs,
- isolated nodes.

This choice makes the explanations useful for diagnosis. Random nodes would be easier to generate but much weaker as evidence.

### Why both HTML and notebook views?

- The HTML exports support interactive review and sharing.
- The notebook gives a side-by-side analytic view that is easier to annotate and compare in a report.

### Why use smaller subgraphs in the notebook?

The mining/export step saves broader local explanation context, while the notebook focuses on more readable comparison plots for human interpretation.

## Files In This Folder

| File | Purpose |
| --- | --- |
| `2a_explainer_miner.py` | Selects target nodes, runs GNNExplainer, and saves HTML outputs and metadata |
| `2b_explainer_analysis.ipynb` | Notebook dashboard for cross-model interpretation |
| `explainer_metadata.csv` | Summary table for selected nodes and top edge weights |
| `explanations/` | HTML explanation artifacts |

## How To Reproduce

Run from the repository root:

```bash
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install networkx certifi
python3 ml_extensions/task2_explainer/2a_explainer_miner.py
```

Then open:

- [ml_extensions/task2_explainer/2b_explainer_analysis.ipynb](/Users/emaheshwari/Project2/ml_extensions/task2_explainer/2b_explainer_analysis.ipynb)

## Expected Inputs

- `app_data/champion_gcn.pth`
- `app_data/champion_gat.pth`
- local Cora data in `data/Cora/`

## Expected Outputs

- `ml_extensions/task2_explainer/explainer_metadata.csv`
- HTML files under `ml_extensions/task2_explainer/explanations/`

## How To Read The Results

This task is designed to answer questions like:

- which specific citations dominate a prediction,
- whether the model is relying on a broad neighborhood or a few strong links,
- whether GAT is filtering noisy neighbors more aggressively than GCN.

The metadata CSV is a quick summary layer. The HTML files and notebook are the evidence layer.

## Reproducibility Notes

- The task depends on saved champion models rather than retraining models inside the explainer script.
- The miner currently produces a large number of HTML outputs because it searches multiple archetypes across all classes. That is intentional: the project is building a reusable explanation set, not a single demo figure.
