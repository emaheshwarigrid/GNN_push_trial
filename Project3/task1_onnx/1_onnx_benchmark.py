import os
import sys
import time
import psutil
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import onnxruntime as rt
from pathlib import Path
import copy

# ==========================================
# PATH SAFETY BLOCK
# ==========================================
# Ensure we are in the PROJECT2 root directory
current_dir = Path.cwd()
if current_dir.parent.name == 'ml_extensions':
    os.chdir(current_dir.parent.parent)
    sys.path.append(os.getcwd())

print(f"🚀 Initializing Task 1 Benchmarking from: {os.getcwd()}\n")

# ==========================================
# INSTRUCTION 1: Train GCN/GAT to 82%+ accuracy
# ==========================================
# You already achieved this! We just need to define the architecture 
# so we can load your saved weights. (Adjust hidden dimensions if your champion differs).
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

num_features = 1433
num_classes = 7
model = GCN(num_features, num_classes)

# Load your champion model weights from app_data
model_path = 'app_data/champion_gcn.pth' # Update this filename if yours is different!
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    print(f"✅ Loaded champion weights from {model_path}")
else:
    print(f"⚠️ Warning: Could not find {model_path}. Using random weights for benchmark.")
model.eval()


# ==========================================
# INSTRUCTION 2: Export trained model to ONNX format
# ==========================================
print("\n📦 Exporting to ONNX with Dynamic Axes...")
os.makedirs("ml_extensions/task1_onnx", exist_ok=True)
onnx_path = "ml_extensions/task1_onnx/cora_model.onnx"

# Dummy data for the PyTorch tracer
dummy_x = torch.randn(5, num_features)
dummy_edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

torch.onnx.export(
    model, 
    (dummy_x, dummy_edge_index), 
    onnx_path,
    export_params=True,
    opset_version=14,
    input_names=['x', 'edge_index'],
    output_names=['output'],
    # DYNAMIC AXES: Critical for variable batch sizes
    dynamic_axes={
        'x': {0: 'num_nodes'},
        'edge_index': {1: 'num_edges'},
        'output': {0: 'num_nodes'}
    }
)
print(f"✅ ONNX model saved to {onnx_path}")


# ==========================================
# INSTRUCTION 3: Load the ONNX model with onnxruntime
# ==========================================
# sess = rt.InferenceSession('cora_model.onnx')
# --- ADVANCED MLOPS: Session Optimization ---
sess_options = rt.SessionOptions()
# Set execution mode to parallel to utilize multiple cores
sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
# Bind the threads strictly to the physical cores of your Mac (avoids virtual core overhead)
sess_options.intra_op_num_threads = psutil.cpu_count(logical=False) 

sess = rt.InferenceSession(onnx_path, sess_options=sess_options)
print("✅ ONNX Runtime Session Initialized.")


# ==========================================
# INSTRUCTION 4, 5 & 6: Batched inference, Measure Latency, Compare MPS vs CPU
# ==========================================
print("\n⏱️ Starting Hardware Profiling...")

# Setup PyTorch devices
device_cpu = torch.device('cpu')
device_mps = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

if device_mps.type == 'mps':
    print("🍎 Apple Silicon (MPS) detected! Proceeding with benchmark.\n")
    model_mps = copy.deepcopy(model).to(device_mps)
else:
    print("⚠️ MPS not detected. Falling back to CPU only.\n")

# Process 100 papers at once, plus the other requested sizes
batch_sizes = [1, 10, 32, 64, 100]
results_log = []

for batch in batch_sizes:
    # 1. PREPROCESSING (Simulating data prep)
    t0_prep = time.perf_counter()
    b_x = torch.randn(batch, num_features)
    num_edges = min(batch * 3, batch * batch) 
    b_edge_index = torch.randint(0, batch, (2, num_edges), dtype=torch.long)
    t1_prep = time.perf_counter()
    prep_time = (t1_prep - t0_prep) * 1000

    # 2. INFERENCE: PyTorch CPU
    t0_inf_cpu = time.perf_counter()
    with torch.no_grad():
        out_cpu = model(b_x, b_edge_index)
    t1_inf_cpu = time.perf_counter()
    inf_time_cpu = (t1_inf_cpu - t0_inf_cpu) * 1000

    # 3. INFERENCE: PyTorch MPS
    inf_time_mps = 0.0
    if device_mps.type == 'mps':
        b_x_mps = b_x.to(device_mps)
        b_edge_index_mps = b_edge_index.to(device_mps)
        with torch.no_grad():
            for _ in range(3):
                _ = model_mps(b_x_mps, b_edge_index_mps)
            torch.mps.synchronize()
        
        t0_inf_mps = time.perf_counter()
        with torch.no_grad():
            out_mps = model_mps(b_x_mps, b_edge_index_mps)
            torch.mps.synchronize() # Wait for GPU to finish
        t1_inf_mps = time.perf_counter()
        inf_time_mps = (t1_inf_mps - t0_inf_mps) * 1000

    # 4. INFERENCE: ONNX Runtime (CPU)
    ort_inputs = {'x': b_x.numpy(), 'edge_index': b_edge_index.numpy()}
    for _ in range(3):
        _ = sess.run(None, ort_inputs)
    t0_inf_onnx = time.perf_counter()
    out_onnx = sess.run(None, ort_inputs)
    t1_inf_onnx = time.perf_counter()
    inf_time_onnx = (t1_inf_onnx - t0_inf_onnx) * 1000

    # 5. POSTPROCESSING (Taking argmax to get class predictions)
    t0_post = time.perf_counter()
    preds = out_cpu.argmax(dim=1)
    t1_post = time.perf_counter()
    post_time = (t1_post - t0_post) * 1000

    # 6. MEASURE MEMORY USAGE
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)

    # 7. LOGGING THE TIMING BREAKDOWN & LATENCY PER SAMPLE
    results_log.append({
        'Batch_Size': batch,
        'Prep_Time_ms': round(prep_time, 3),
        'Inf_PyTorch_CPU_ms': round(inf_time_cpu, 3),
        'Inf_PyTorch_MPS_ms': round(inf_time_mps, 3),
        'Inf_ONNX_ms': round(inf_time_onnx, 3),
        'Post_Time_ms': round(post_time, 3),
        # Latency per sample = Total Inference Time / Batch Size
        'Latency_Per_Sample_ONNX_ms': round(inf_time_onnx / batch, 3),
        'Memory_Usage_MB': round(mem_mb, 1)
    })

# ==========================================
# INSTRUCTION 7: Log results 
# ==========================================
df_results = pd.DataFrame(results_log)
log_path = 'ml_extensions/task1_onnx/task1_timing_breakdown.csv'
df_results.to_csv(log_path, index=False)

print("\n📊 Benchmarking Results:")
print(df_results.to_string(index=False))
print(f"\n✅ Full timing breakdown saved to {log_path}")
print("🎉 Task 1 Complete!")