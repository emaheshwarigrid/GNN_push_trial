# ML Extensions for Cora GNN Engineering

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](#installation)
[![PyG](https://img.shields.io/badge/PyTorch%20Geometric-2.x-green)](#installation)
[![HTML Report](https://img.shields.io/badge/HTML-Live_Report-red)](https://emaheshwarigrid.github.io/project3html/)

**ML extensions for a Cora citation-network project, covering deployment, explainability, pruning, representation steering, and mobile quantization workflows.**

## Description

This directory contains the engineering extension tasks built on top of the main Cora Graph Neural Network project. While the base project focuses on training and selecting strong GNN models, `ml_extensions/` explores the next stage of model development:

- deployment readiness,
- interpretability,
- compression,
- representation analysis, and
- mobile inference constraints.

These extensions exist to answer practical questions that a baseline accuracy score cannot answer on its own. For example:

- Can the trained model be exported and benchmarked in ONNX?
- Which citation edges most influence a prediction?
- Are all GAT attention heads necessary?
- Can learned topic representations be steered at inference time?
- How well does quantization support mobile deployment?

In short, this folder turns a research-style GNN project into a more complete model engineering portfolio.

## Features

- **Task 1: ONNX export and batching**
  - Export a trained Cora classifier to ONNX
  - Benchmark batched inference across multiple batch sizes
  - Compare PyTorch CPU, MPS, and ONNX runtime behavior

- **Task 2: GNN explainability**
  - Use `GNNExplainer` to analyze influential citation edges
  - Compare explanation patterns between GCN and GAT
  - Generate HTML-based explanation artifacts

- **Task 3: Attention head pruning**
  - Rank GAT heads by ablation impact
  - Iteratively prune low-value heads
  - Measure the trade-off between model size and accuracy

- **Task 4: Concept vector steering**
  - Extract topic-level latent concept vectors
  - Apply inference-time activation steering
  - Audit how class confidence changes under perturbation

- **Task 5: Quantization and mobile export**
  - Apply post-training dynamic quantization
  - Fine-tune with QAT and fake quantization
  - Export a CoreML-compatible mobile deployment artifact

## Table of Contents

- [Project Title](#ml-extensions-for-cora-gnn-engineering)
- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Task Overview](#task-overview)
- [Outputs](#outputs)
- [Reproducibility Notes](#reproducibility-notes)
- [Placeholders](#placeholders)

## Installation

Run all commands from the repository root, not from inside `ml_extensions/`.

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install base dependencies

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### 3. Install task-specific optional dependencies

Some extension tasks require packages beyond the base requirements.

```bash
python3 -m pip install psutil networkx certifi onnxscript
python3 -m pip install plotly ipywidgets seaborn
python3 -m pip install coremltools
```

### 4. Confirm required project assets exist

These extensions expect the main project artifacts to already exist, especially:

- `app_data/champion_gcn.pth`
- `app_data/champion_gat.pth`
- `app_data/champion_graphsage.pth`
- `app_data/champion_appnp.pth`
- local Cora data under `data/Cora/`

## Usage

### Run Task 1: ONNX benchmarking

```bash
python3 ml_extensions/task1_onnx/1_onnx_benchmark.py
python3 ml_extensions/task1_onnx/plot_task1.py
```

### Run Task 2: Explainability pipeline

```bash
python3 ml_extensions/task2_explainer/2a_explainer_miner.py
```

Then open the notebook:

```bash
jupyter notebook ml_extensions/task2_explainer/2b_explainer_analysis.ipynb
```

### Run Task 3: Attention head ablation

Open and run:

```bash
jupyter notebook ml_extensions/task3_/3_ablation.ipynb
```

### Run Task 4: Concept steering

Open and run:

```bash
jupyter notebook ml_extensions/task4_/4_steering.ipynb
```

### Run Task 5: Quantization and mobile export

Open and run:

```bash
jupyter notebook ml_extensions/task5_/5_quantization.ipynb
```

Optional mobile export:

```bash
python3 ml_extensions/task5_/5_export_decoupled.py
```

## Project Structure

```text
ml_extensions/
├── README.md
├── task1_onnx/
│   ├── 1_onnx_benchmark.py
│   ├── plot_task1.py
│   ├── cora_model.onnx
│   ├── task1_timing_breakdown.csv
│   └── task1_benchmark_plot.png
├── task2_explainer/
│   ├── 2a_explainer_miner.py
│   ├── 2b_explainer_analysis.ipynb
│   ├── explainer_metadata.csv
│   └── explanations/
├── task3_/
│   ├── 3_ablation.ipynb
│   ├── task3_head_ranking.csv
│   ├── task3_pruning_history.csv
│   └── task3_visual.png
├── task4_/
│   ├── 4_steering.ipynb
│   ├── task4_steering_audit.csv
│   └── task4_steering_vectors.pt
└── task5_/
    ├── 5_quantization.ipynb
    ├── 5_export_decoupled.py
    └── GraphEngine.swift
```

## Task Overview

### Task 1: ONNX Export and Inference Profiling

Focuses on deployment-style inference benchmarking. The script exports a trained model to ONNX, runs batched inference, and compares timing across runtime modes.

**Why it matters:**  
It tests whether the model is practical outside the training notebook.

### Task 2: Citation Explainability

Focuses on interpreting predictions through subgraph-level explanation. It compares how GCN and GAT prioritize citation edges for different node archetypes.

**Why it matters:**  
It helps explain model behavior, especially on mistakes and noisy neighborhoods.

### Task 3: GAT Head Pruning

Focuses on compression through ablation. It ranks attention heads, prunes low-impact heads, and tracks how much accuracy is preserved.

**Why it matters:**  
It identifies architectural redundancy and tests whether the model can be made smaller without major performance loss.

### Task 4: Concept Steering

Focuses on representation-level intervention. It computes topic centroids in latent space and tests whether prediction confidence can be shifted by injecting concept vectors.

**Why it matters:**  
It probes what the model has learned internally, not just what it predicts.

### Task 5: Quantization and Mobile Deployment

Focuses on deployment constraints. It applies dynamic quantization, explores QAT, and exports a decoupled CoreML-compatible inference component for iOS-style deployment.

**Why it matters:**  
It connects the research model to a realistic mobile deployment story.

## Outputs

Common outputs produced by these tasks include:

- **ONNX artifacts**
  - exported model files
  - timing CSVs
  - benchmark plots

- **Explainability artifacts**
  - explanation metadata tables
  - interactive HTML visualizations

- **Pruning artifacts**
  - head ranking tables
  - pruning history logs
  - saved pruned checkpoints

- **Steering artifacts**
  - concept vector tensors
  - audit CSVs
  - interactive notebook visualizations

- **Quantization artifacts**
  - FP32 baseline checkpoint
  - PTDQ checkpoint
  - QAT checkpoint
  - CoreML export files
  - mobile edge CSVs

## Reproducibility Notes

- Several notebooks retrain task-specific models, so exact numeric parity may vary without fixed seeds.
- Some tasks use `RandomNodeSplit`, which introduces additional randomness if not explicitly seeded.
- ONNX export may require `onnxscript` depending on your PyTorch version.
- CoreML export is macOS-focused and depends on `coremltools`.
- These tasks are best understood as engineering experiments, not only as leaderboard-style benchmarks.




