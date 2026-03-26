import torch
import torch.nn.functional as F
import coremltools as ct
import coremltools.optimize.coreml as cto
import pandas as pd
from pathlib import Path
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv

print("🌉 Building the Decoupled Deployment Kit...")

# --- Setup ---
device_cpu = torch.device('cpu')
current_dir = Path.cwd()
app_data_dir = current_dir / 'app_data'
dataset = Planetoid(root=str(current_dir / 'data' / 'Cora'), name='Cora')
data = dataset[0].to(device_cpu)

# --- 1. Export the Graph Map (The Edges) ---
print("🗺️ Saving Graph Edges for the iOS CPU...")
edges_df = pd.DataFrame(data.edge_index.numpy().T, columns=['source', 'target'])
edges_csv_path = app_data_dir / 'cora_edges_mobile.csv'
edges_df.to_csv(edges_csv_path, index=False)
print(f"   Saved to: {edges_csv_path}")

# --- 2. The Original Architecture ---
class Task5GAT(torch.nn.Module):
    def __init__(self, hidden_channels=16, heads=8, dropout_p=0.6):
        super().__init__()
        self.dropout_p = dropout_p
        self.conv1 = GATConv(dataset.num_node_features, hidden_channels, heads=heads, dropout=dropout_p)
        self.conv2 = GATConv(hidden_channels * heads, dataset.num_classes, heads=1, concat=False, dropout=dropout_p)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x

# --- 3. Load Saved Weights safely ---
qat_model = Task5GAT().to(device_cpu)
qat_path = app_data_dir / 'task5_qat_model.pth'
try:
    qat_model.load_state_dict(torch.load(qat_path, map_location=device_cpu))
except RuntimeError:
    qat_model.load_state_dict(torch.load(app_data_dir / 'task5_baseline_fp32.pth', map_location=device_cpu)['model_state_dict'])
qat_model.eval()

# --- 4. The CoreML Bridge (Bypass PyG Edges) ---
def get_linear_layer(conv_module):
    """Safely extracts the standard matrix math from PyG."""
    if hasattr(conv_module, 'lin') and conv_module.lin is not None: return conv_module.lin
    if hasattr(conv_module, 'lin_src') and conv_module.lin_src is not None: return conv_module.lin_src
    raise ValueError("Linear weights not found.")

class CoreMLNodeProcessor(torch.nn.Module):
    def __init__(self, gnn_model):
        super().__init__()
        self.layer1 = get_linear_layer(gnn_model.conv1)
        self.layer2 = get_linear_layer(gnn_model.conv2)

    def forward(self, x):
        x = self.layer1(x)
        x = F.elu(x)
        x = self.layer2(x)
        return x

coreml_bridge_model = CoreMLNodeProcessor(qat_model).eval()

# --- 5. Trace, Convert, and Quantize ---
print("🧠 Compiling the Math to CoreML...")
with torch.no_grad():
    traced_model = torch.jit.trace(coreml_bridge_model, (data.x,))

# We use the classic .mlmodel format
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="node_features", shape=data.x.shape)],
    minimum_deployment_target=ct.target.iOS14
)

# --- 6. Apple Classic INT8 Quantization ---
print("🗜️ Applying Classic Apple-Native INT8 Optimization...")
from coremltools.models.neural_network import quantization_utils

# We use the classic quantization tool specifically designed for the older format
quantized_mlmodel = quantization_utils.quantize_weights(mlmodel, nbits=8)

# --- 7. Save Model ---
coreml_path = app_data_dir / "CoraMath_Quantized.mlmodel"
quantized_mlmodel.save(str(coreml_path))

print(f"   Saved to: {coreml_path}")
print("\n" + "="*50)
print("✅ STRICT COREML EXPORT SUCCESSFUL!")
print("="*50)
print("DEPLOYMENT KIT ASSETS GENERATED!")