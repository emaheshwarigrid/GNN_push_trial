# Task 3: Attention Head Pruning Via Ablation Study

## Objective

This task investigates whether all GAT attention heads are equally useful. The end goal is a smaller model that preserves most of the original accuracy while reducing parameter count and deployment cost.

## Experimental Decisions

### Why analyze the first attention layer?

The first GAT layer is where multi-head attention is easiest to isolate and interpret. Pruning there makes the ablation mechanically clear:

- remove one head,
- re-evaluate the model,
- measure the accuracy change.

### Why use ablation before pruning?

Directly pruning half the heads at once would hide which heads matter. The ablation step creates an importance ranking first, which turns pruning from guesswork into a decision backed by measurement.

### Why prune iteratively with short retraining?

Ablation tells us which heads appear redundant, but the remaining heads still need time to adapt. The notebook therefore removes heads one at a time and briefly retrains after each cut.

### Why compare both masked and native-architecture views?

The saved checkpoint is a masked version of the trained 8-head network, while the hardware-footprint comparison also instantiates a native 4-head architecture to estimate structural savings. This separates:

- decision making inside the trained model,
- deployment-style size and latency comparison.

## Files In This Folder

| File | Purpose |
| --- | --- |
| `3_ablation.ipynb` | Full pruning workflow notebook |
| `task3_head_ranking.csv` | Head-importance ranking from ablation |
| `task3_pruning_history.csv` | Accuracy and size trade-off log |
| `task3_visual.png` | Saved visualization artifact |

## How To Reproduce

Open and run:

- [ml_extensions/task3_/3_ablation.ipynb](/Users/emaheshwari/Project2/ml_extensions/task3_/3_ablation.ipynb)

Run it from the repository root or with the notebook kernel started in the project folder so the path detection works correctly.

## Expected Outputs

- `ml_extensions/task3_/task3_head_ranking.csv`
- `ml_extensions/task3_/task3_pruning_history.csv`
- `app_data/champion_pruned_gat.pth`

## How To Read The Results

The task should be read in four stages:

1. establish baseline accuracy,
2. rank heads by single-head removal impact,
3. prune the lowest-value heads,
4. compare accuracy retention against footprint reduction.

That structure matters because it explains the pruning decision instead of only presenting the final compressed model.

## Reproducibility Notes

- The notebook trains a fresh task-specific GAT rather than loading the saved `champion_gat.pth`.
- Because the split and training are stochastic, exact values can vary between runs unless seeds are fixed manually.
- The saved artifact includes a head mask alongside the state dict. Reviewers should interpret it as a task-specific pruned checkpoint rather than a universal drop-in replacement for every GAT script in the repository.
