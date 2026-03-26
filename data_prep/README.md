# Data Preparation And Exploration

This folder contains the earliest project notebooks, where the focus is understanding Cora before optimizing any model.

## Notebook Map

| Notebook | Purpose |
| --- | --- |
| `1_data.ipynb` | Load and inspect the dataset |
| `2_data_explore_and_visualisation.ipynb` | Explore graph structure and class distribution visually |

## Why This Phase Exists

Starting with data inspection helps ground later decisions:

- what the feature space looks like,
- how many classes exist,
- how sparse the citation structure is,
- what kinds of imbalances or structural quirks might affect training.

These notebooks provide the context for later architecture choices and explain why graph-aware methods are appropriate for the dataset.
