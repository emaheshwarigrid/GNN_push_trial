# Task 5: Quantization For Mobile Deployment

## Objective

This task explores how far a Cora GAT can be pushed toward edge deployment. It covers three connected problems:

- compress the model with INT8 quantization,
- check the speed and accuracy trade-offs,
- package the math for an Apple-style mobile deployment path.

## Experimental Decisions

### Why use a GAT baseline?

GAT is the most natural candidate here because it is already the attention-heavy architecture used in the pruning task and is a better stress test for deployment than a very small plain GCN.

### Why do PTDQ before QAT?

The task first applies post-training dynamic quantization to establish a low-effort compression baseline. Only after that does it run QAT to test whether learned adaptation improves accuracy enough to justify the added complexity.

### Why swap PyG linear layers?

PyG modules are convenient for model building but not always friendly to downstream quantization/export tools. Replacing internal linear layers with standard PyTorch modules makes quantization easier to control.

### Why use a decoupled CoreML export?

CoreML does not natively support all PyG message-passing operators used in graph networks. The export script therefore separates the problem into:

- feature transformation math for CoreML,
- graph routing and aggregation outside CoreML.

This is an architectural workaround, not a claim that PyG graphs convert directly to a single mobile artifact.

## Files In This Folder

| File | Purpose |
| --- | --- |
| `5_quantization.ipynb` | Baseline training, PTDQ, benchmarking, and QAT |
| `5_export_decoupled.py` | CoreML-oriented decoupled export script |
| `GraphEngine.swift` | Sketch of the iOS-side bridge for graph routing |

## How To Reproduce

From the repository root:

```bash
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install coremltools
```

Then:

- run [ml_extensions/task5_/5_quantization.ipynb](/Users/emaheshwari/Project2/ml_extensions/task5_/5_quantization.ipynb)
- optionally run `python3 ml_extensions/task5_/5_export_decoupled.py`

## Expected Outputs

- `app_data/task5_baseline_fp32.pth`
- `app_data/task5_quantized_ptdq.pth`
- `app_data/task5_qat_model.pth`
- `app_data/CoraMath_Quantized.mlmodel`
- `app_data/cora_edges_mobile.csv`

## How To Read The Results

This task should be read as a staged deployment analysis:

1. train a full-precision baseline,
2. compress it with PTDQ,
3. benchmark whether INT8 actually helps on this workload,
4. run QAT to recover or improve performance,
5. export the pieces needed for a mobile deployment prototype.

The interesting result is not only compression ratio. It is whether compression helps on the actual hardware and software path being targeted.

## Reproducibility Notes

- The notebook's benchmarking section compares FP32 on CPU and MPS, and INT8 on CPU. The CoreML export path is a separate deployment experiment.
- `coremltools` is not included in the base `requirements.txt`, so it must be installed separately for export.
- Because the graph routing is decoupled from the CoreML math module, the `.mlmodel` artifact is best understood as a mobile building block, not a complete standalone graph inference runtime.
