# Task 4: Concept Vector Steering For Topic Bias Analysis

## Objective

This task studies whether internal topic representations can be manipulated at inference time without retraining the model. The experiment asks: if we move a paper's hidden state toward another topic centroid, how much does the prediction change?

## Experimental Decisions

### Why use mean class representations?

The mean hidden vector for a class is a simple and interpretable approximation of that topic's representation in latent space. It is not the only possible choice, but it is a strong first mechanism for concept steering because it is easy to compute and explain.

### Why steer hidden activations instead of retraining?

The point of the task is to inspect and perturb what the model already knows. Steering at inference time isolates representation behavior more cleanly than modifying weights.

### Why use `target centroid - source centroid`?

This creates a directional vector that moves a source example toward a target class manifold. It is the most direct experimental construction for testing transferability between learned concepts.

### Why evaluate multiple alpha values?

Using `alpha = [0.1, 0.3, 0.5]` tests whether the effect is:

- negligible at low intensity,
- measurable at medium intensity,
- strong enough to force meaningful drift at higher intensity.

### Why include t-SNE?

The audit CSV measures confidence changes numerically. The t-SNE view adds an intuitive spatial explanation for how a node moves in representation space as steering strength increases.

## Files In This Folder

| File | Purpose |
| --- | --- |
| `4_steering.ipynb` | Full steering workflow notebook |
| `task4_steering_vectors.pt` | Saved centroid and adversarial vectors |
| `task4_steering_audit.csv` | Confidence-drop audit across source-target pairs and alpha values |

## How To Reproduce

Open and run:

- [ml_extensions/task4_/4_steering.ipynb](/Users/emaheshwari/Project2/ml_extensions/task4_/4_steering.ipynb)

Recommended environment additions:

```bash
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install plotly ipywidgets seaborn
```

## Expected Outputs

- `ml_extensions/task4_/task4_steering_vectors.pt`
- `ml_extensions/task4_/task4_steering_audit.csv`

## How To Read The Results

The key output is the transferability pattern across topic pairs:

- which topics are easiest to push away from their original class,
- which target directions are most disruptive,
- how sensitive the model is to activation-level intervention.

This task is best interpreted as representation analysis, not as a production mitigation technique on its own.

## Reproducibility Notes

- The notebook trains a task-specific GCN and then derives concept vectors from that run.
- Exact numerical values may vary between runs because the model training and split are stochastic.
- The notebook includes notebook-local package installation cells. Those are convenient for interactive use, but a clean environment setup from the repository root is more reliable for long-term reproduction.
