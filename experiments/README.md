# Champion Experiments

This folder contains the follow-up notebooks that revisit the strongest baseline models after the main sweeps in `models/`.

## Purpose

The project does not treat the best grid-search row as the final answer. These notebooks exist to answer the next question:

"Does the best-looking model remain strong when we probe it more carefully?"

## Notebook Map

| Notebook | Focus |
| --- | --- |
| `7_champion_experiments.ipynb` | APPNP-focused follow-up experiments and diagnostics |
| `7_champion_expirements2.ipynb` | GCN-focused follow-up experiments and diagnostics |

## What Happens Here

These notebooks are where the project moves from selection to justification. Typical activities include:

- retraining strong candidates,
- checking robustness-oriented variants,
- generating diagnostic plots,
- deciding which artifacts deserve to be saved for the app and extension tasks.

## Why This Step Matters

Separating champion analysis from the first-pass sweeps keeps the project honest:

- `models/` answers "what looked promising?"
- `experiments/` answers "what is worth keeping?"
