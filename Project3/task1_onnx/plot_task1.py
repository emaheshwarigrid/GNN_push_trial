import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# ==========================================
# PATH SAFETY BLOCK
# ==========================================
current_dir = Path.cwd()
if current_dir.name == 'task1_onnx':
    os.chdir(current_dir.parent.parent)
    sys.path.append(os.getcwd())
elif current_dir.name == 'ml_extensions':
    os.chdir(current_dir.parent)
    sys.path.append(os.getcwd())

print(f"📊 Generating charts from: {os.getcwd()}")

# 1. Load the benchmark data
csv_path = 'ml_extensions/task1_onnx/task1_timing_breakdown.csv'
if not os.path.exists(csv_path):
    print(f"❌ Error: Could not find {csv_path}. Run the benchmark script first!")
    sys.exit()

df = pd.read_csv(csv_path)

# 2. Calculate Throughput (Papers processed per second)
# Formula: (Batch Size / Total Inference Time in ms) * 1000 ms/sec
df['Throughput_ONNX'] = (df['Batch_Size'] / df['Inf_ONNX_ms']) * 1000
df['Throughput_CPU'] = (df['Batch_Size'] / df['Inf_PyTorch_CPU_ms']) * 1000

# 3. Create a beautiful 1x2 plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('GNN Inference Performance: PyTorch CPU vs ONNX Runtime', fontsize=16, fontweight='bold')

# --- SUBPLOT 1: Latency Per Sample (Lower is Better) ---
ax1.plot(df['Batch_Size'], df['Latency_Per_Sample_ONNX_ms'], marker='o', linewidth=2, color='#2ca02c', label='ONNX (CPU)')
# We approximate PyTorch Latency per sample by dividing total time by batch
ax1.plot(df['Batch_Size'], df['Inf_PyTorch_CPU_ms'] / df['Batch_Size'], marker='s', linewidth=2, linestyle='--', color='#1f77b4', label='PyTorch (CPU)')

ax1.set_title('Latency Per Paper (ms) - Lower is Better', fontsize=12)
ax1.set_xlabel('Batch Size (Number of Papers)', fontsize=11)
ax1.set_ylabel('Milliseconds per Paper', fontsize=11)
ax1.set_xticks(df['Batch_Size'])
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# --- SUBPLOT 2: Throughput (Higher is Better) ---
ax2.plot(df['Batch_Size'], df['Throughput_ONNX'], marker='o', linewidth=2, color='#2ca02c', label='ONNX (CPU)')
ax2.plot(df['Batch_Size'], df['Throughput_CPU'], marker='s', linewidth=2, linestyle='--', color='#1f77b4', label='PyTorch (CPU)')

ax2.set_title('Throughput (Papers / Second) - Higher is Better', fontsize=12)
ax2.set_xlabel('Batch Size (Number of Papers)', fontsize=11)
ax2.set_ylabel('Papers Processed per Second', fontsize=11)
ax2.set_xticks(df['Batch_Size'])
# Use a log scale for Y because ONNX throughput gets massive
ax2.set_yscale('log') 
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

# 4. Save the plot
plt.tight_layout()
plot_path = 'ml_extensions/task1_onnx/task1_benchmark_plot.png'
plt.savefig(plot_path, dpi=300)
print(f"✅ Success! Beautiful benchmark chart saved to: {plot_path}")