import os
import sys
import certifi
import pandas as pd
import networkx as nx
from pyvis.network import Network
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph, degree
from pathlib import Path

# ==========================================
# 0. PATH SAFETY & SSL SETUP
# ==========================================
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

current_dir = Path.cwd()
if current_dir.name == 'task2_explainer':
    os.chdir(current_dir.parent.parent)
    sys.path.append(os.getcwd())
elif current_dir.name == 'ml_extensions':
    os.chdir(current_dir.parent)
    sys.path.append(os.getcwd())

# ==========================================
# 1. CLASSES & HELPER FUNCTIONS (Safe to import)
# ==========================================
print("📦 Loading Cora Dataset...")
dataset = Planetoid(root='./data/Cora', name='Cora')
data = dataset[0]
class_names = ['Theory', 'RL', 'Gen_Algos', 'Neural_Nets', 'Prob_Methods', 'Case_Based', 'Rule_Learning']

# Define GCN
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, num_classes)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return F.log_softmax(self.conv2(x, edge_index), dim=1)

# Define GAT 
class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(num_features, 32, heads=8, dropout=0.6))
        self.convs.append(GATConv(32 * 8, num_classes, heads=1, concat=False, dropout=0.6))

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.convs[0](x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        return F.log_softmax(self.convs[1](x, edge_index), dim=1)

def save_html_subgraph(node_idx, edge_index, edge_mask, model_name, archetype, class_name):
    subset, sub_edge_index, mapping, edge_mask_subset = k_hop_subgraph(node_idx, num_hops=2, edge_index=edge_index, num_nodes=data.num_nodes)
    subset_weights = edge_mask[edge_mask_subset]
    
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    
    for i, node in enumerate(subset.numpy()):
        color = "#e74c3c" if node == node_idx else "#3498db"
        label = f"Target {node}" if node == node_idx else f"ID: {node}"
        net.add_node(int(node), label=label, color=color, size=25 if node == node_idx else 15)

    for i in range(sub_edge_index.size(1)):
        src = int(sub_edge_index[0, i].item())
        dst = int(sub_edge_index[1, i].item())
        weight = float(subset_weights[i].item())
        
        if weight > 0.1: 
            net.add_edge(src, dst, value=weight, title=f"Importance: {weight:.3f}", color="#2ecc71" if weight > 0.5 else "#7f8c8d")

    net.toggle_physics(True)
    filename = f"ml_extensions/task2_explainer/explanations/{class_name}_{archetype}_{model_name}.html"
    net.write_html(filename)

# ==========================================
# 2. EXECUTION SCRIPT (Guarded)
# ==========================================
# Everything below this line will ONLY run if you execute this specific file.
# It will NOT run if you import functions into your Jupyter Notebook!
if __name__ == "__main__":
    print(f"🚀 Initializing Advanced Data Miner from: {os.getcwd()}")
    os.makedirs("ml_extensions/task2_explainer/explanations", exist_ok=True)

    # Load Models
    gcn_model = GCN(dataset.num_node_features, dataset.num_classes)
    gat_model = GAT(dataset.num_node_features, dataset.num_classes)

    try:
        gcn_model.load_state_dict(torch.load('app_data/champion_gcn.pth', map_location='cpu', weights_only=True))
        gat_model.load_state_dict(torch.load('app_data/champion_gat.pth', map_location='cpu', weights_only=True))
        print("✅ Champion Models Loaded.")
    except FileNotFoundError:
        print("⚠️ Warning: Model weights not found. Check paths.")
        sys.exit()

    gcn_model.eval()
    gat_model.eval()

    print("🔍 Running Advanced Heuristics (Margin & Heterophily) to isolate optimal nodes...")

    with torch.no_grad():
        gcn_out = gcn_model(data.x, data.edge_index)
        gcn_probs = torch.exp(gcn_out)
        
        # Calculate Top 1 Confidence and Top 2 Margin
        top2_probs, top2_preds = torch.topk(gcn_probs, 2, dim=1)
        gcn_conf = top2_probs[:, 0]
        gcn_preds = top2_preds[:, 0]
        margins = top2_probs[:, 0] - top2_probs[:, 1]

    # Calculate Standard Degree
    node_degrees = degree(data.edge_index[0], num_nodes=data.num_nodes)

    # Calculate Heterophily Degree (Edges connecting to DIFFERENT classes)
    src, dst = data.edge_index
    hetero_edges = (data.y[src] != data.y[dst]).float()
    hetero_degree = torch.zeros(data.num_nodes)
    hetero_degree.index_add_(0, src, hetero_edges)

    # Build Master DataFrame
    df_nodes = pd.DataFrame({
        'Node_ID': range(data.num_nodes),
        'True_Class': data.y.numpy(),
        'Pred_Class': gcn_preds.numpy(),
        'Confidence': gcn_conf.numpy(),
        'Margin': margins.numpy(),
        'Degree': node_degrees.numpy(),
        'Hetero_Degree': hetero_degree.numpy(),
        'Is_Correct': (data.y == gcn_preds).numpy()
    })

    selected_nodes = []
    used_nodes = set()

    # Hunt for the 5 optimized archetypes across all 7 classes
    for c in range(7):
        # 1. Slam Dunk (Correct, 95th Percentile Confidence)
        class_df = df_nodes[(df_nodes['True_Class'] == c) & (~df_nodes['Node_ID'].isin(used_nodes))]
        correct_df = class_df[class_df['Is_Correct'] == True].sort_values('Confidence', ascending=False)
        if not correct_df.empty:
            idx = min(max(0, int(len(correct_df) * 0.05)), len(correct_df)-1)
            slam_dunk = correct_df.iloc[idx]
            selected_nodes.append({'Archetype': 'Slam_Dunk', 'Class': class_names[c], 'Node_ID': slam_dunk['Node_ID']})
            used_nodes.add(slam_dunk['Node_ID'])

        # 2. The Mistake (Incorrect, 95th Percentile Confidence)
        class_df = df_nodes[(df_nodes['True_Class'] == c) & (~df_nodes['Node_ID'].isin(used_nodes))]
        incorrect_df = class_df[class_df['Is_Correct'] == False].sort_values('Confidence', ascending=False)
        if not incorrect_df.empty:
            idx = min(max(0, int(len(incorrect_df) * 0.05)), len(incorrect_df)-1)
            mistake = incorrect_df.iloc[idx]
            selected_nodes.append({'Archetype': 'Mistake', 'Class': class_names[c], 'Node_ID': mistake['Node_ID']})
            used_nodes.add(mistake['Node_ID'])

        # 3. Lucky Guess (Correct, Slimmest Margin of Victory)
        class_df = df_nodes[(df_nodes['True_Class'] == c) & (~df_nodes['Node_ID'].isin(used_nodes))]
        lucky_df = class_df[class_df['Is_Correct'] == True].sort_values('Margin', ascending=True)
        if not lucky_df.empty:
            lucky = lucky_df.iloc[0]
            selected_nodes.append({'Archetype': 'Lucky_Guess', 'Class': class_names[c], 'Node_ID': lucky['Node_ID']})
            used_nodes.add(lucky['Node_ID'])

        # 4. Heterophily Hub (Highest Conflicting Citations)
        class_df = df_nodes[(df_nodes['True_Class'] == c) & (~df_nodes['Node_ID'].isin(used_nodes))]
        hub_df = class_df.sort_values('Hetero_Degree', ascending=False)
        if not hub_df.empty and hub_df.iloc[0]['Hetero_Degree'] > 0:
            hub = hub_df.iloc[0]
            selected_nodes.append({'Archetype': 'Heterophily_Hub', 'Class': class_names[c], 'Node_ID': hub['Node_ID']})
            used_nodes.add(hub['Node_ID'])

        # 5. Isolated Node (Degree of exactly 1 or 2)
        class_df = df_nodes[(df_nodes['True_Class'] == c) & (~df_nodes['Node_ID'].isin(used_nodes))]
        iso_df = class_df[(class_df['Degree'] > 0) & (class_df['Degree'] <= 2)]
        if not iso_df.empty:
            iso = iso_df.iloc[0] 
            selected_nodes.append({'Archetype': 'Isolated', 'Class': class_names[c], 'Node_ID': iso['Node_ID']})
            used_nodes.add(iso['Node_ID'])

    print(f"🎯 Successfully mathematically isolated {len(selected_nodes)} critical nodes.")

    # ==========================================
    # 3. THE EXPLAINER LOOP & HTML GENERATOR
    # ==========================================
    explainer_config = dict(mode='multiclass_classification', task_level='node', return_type='log_probs')
    gcn_explainer = Explainer(model=gcn_model, algorithm=GNNExplainer(epochs=150), explanation_type='model', node_mask_type='attributes', edge_mask_type='object', model_config=explainer_config)
    gat_explainer = Explainer(model=gat_model, algorithm=GNNExplainer(epochs=150), explanation_type='model', node_mask_type='attributes', edge_mask_type='object', model_config=explainer_config)

    results_log = []
    total_nodes = len(selected_nodes)

    print(f"\n⚙️ Starting Explainer Engine on Optimized Nodes. This will take roughly 10-20 minutes...")
    for idx, target in enumerate(selected_nodes):
        n_id = int(target['Node_ID'])
        arch = target['Archetype']
        c_name = target['Class']
        print(f"[{idx+1}/{total_nodes}] Explaining Node {n_id} ({c_name} - {arch})")

        # --- Explain GCN ---
        exp_gcn = gcn_explainer(data.x, data.edge_index, index=n_id)
        save_html_subgraph(n_id, data.edge_index, exp_gcn.edge_mask, "GCN", arch, c_name)
        
        # --- Explain GAT ---
        exp_gat = gat_explainer(data.x, data.edge_index, index=n_id)
        save_html_subgraph(n_id, data.edge_index, exp_gat.edge_mask, "GAT", arch, c_name)

        top_edges_gcn = torch.topk(exp_gcn.edge_mask, 3).values.tolist()
        top_edges_gat = torch.topk(exp_gat.edge_mask, 3).values.tolist()

        results_log.append({
            'Node_ID': n_id,
            'Class': c_name,
            'Archetype': arch,
            'GCN_Top_Edge_Weight': round(top_edges_gcn[0], 4) if top_edges_gcn else 0,
            'GAT_Top_Edge_Weight': round(top_edges_gat[0], 4) if top_edges_gat else 0,
        })

    df_results = pd.DataFrame(results_log)
    df_results.to_csv("ml_extensions/task2_explainer/explainer_metadata.csv", index=False)
    print("\n✅ Phase A Complete! Optimized HTML visualizations and metadata saved.")