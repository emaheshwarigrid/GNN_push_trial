# Task 1: Batched Inference With ONNX Export

## Objective

This task turns a trained Cora classifier into a deployment-style inference artifact. The goal is not just to save an ONNX file, but to answer three practical questions:

- can the model be exported cleanly,
- how does latency change with batch size,
- when does MPS help or hurt on a small graph workload like Cora?

## Experimental Decisions

### Why export the GCN champion?

The implementation uses the saved GCN checkpoint from `app_data/champion_gcn.pth`. That is a sensible export candidate because it is compact, stable, and easy to benchmark without introducing attention-specific complexity.

### Why dynamic axes?

The export is configured with dynamic axes so the same ONNX graph can handle varying node counts and edge counts. That matches the deployment question better than exporting a single fixed-shape graph.

### Why compare ONNX CPU against PyTorch CPU and MPS?

Cora is small. On small workloads, GPU dispatch overhead can dominate the actual forward pass. This task was therefore designed to measure whether MPS acceleration is actually useful rather than assuming it is.

### Why log preprocessing and postprocessing separately?

For a tiny model, framework overhead can exceed pure model time. Splitting the timing into preprocessing, inference, and postprocessing makes the bottleneck visible.

## Files In This Folder

| File | Purpose |
| --- | --- |
| `1_onnx_benchmark.py` | Exports the model, runs inference benchmarks, and saves the timing CSV |
| `plot_task1.py` | Plots latency and throughput from the CSV |
| `task1_timing_breakdown.csv` | Saved benchmark output |
| `task1_benchmark_plot.png` | Visualization of benchmark results |
| `cora_model.onnx` | Exported ONNX artifact |

## How To Reproduce

Run from the repository root:

```bash
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install psutil onnxscript
python3 ml_extensions/task1_onnx/1_onnx_benchmark.py
python3 ml_extensions/task1_onnx/plot_task1.py
```

## Expected Inputs

- `app_data/champion_gcn.pth`
- local Cora data already stored in `data/Cora/`

## Expected Outputs

- `ml_extensions/task1_onnx/cora_model.onnx`
- `ml_extensions/task1_onnx/task1_timing_breakdown.csv`
- `ml_extensions/task1_onnx/task1_benchmark_plot.png`

## How To Read The Results

The most important output is not a single best latency number. It is the pattern:

- batch size should reduce per-sample ONNX latency,
- ONNX CPU should be easy to compare with the PyTorch baseline,
- MPS may lose on small graphs because transfer and launch overhead are a large fraction of total work.

That is the experimental point of this task: deployment decisions should be based on measured workload behavior, not hardware branding.

## Reproducibility Notes

- `onnxruntime` is listed in `requirements.txt`, but newer PyTorch ONNX export flows also require `onnxscript`.
- The benchmark script generates synthetic batch inputs for timing rather than replaying a fixed held-out subset of 100 real papers. That choice is acceptable for throughput profiling, but it should be read as a systems benchmark, not as a semantic accuracy benchmark.
