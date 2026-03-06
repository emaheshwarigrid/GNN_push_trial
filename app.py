import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.nn import APPNP, GCNConv, GATConv, SAGEConv
from torch.nn import Linear
from pyvis.network import Network
import streamlit.components.v1 as components

# --- 1. MODEL DEFINITIONS ---
class GCN_Standard(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_p=0.3):
        super(GCN_Standard, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout_p = dropout_p

    def forward(self, x, edge_index, force_dropout=False):
        p = self.dropout_p if (self.training or force_dropout) else 0
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=p, training=(self.training or force_dropout))
        return self.conv2(x, edge_index)

class FlexibleGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout_p=0.6, heads=8):
        super(FlexibleGAT, self).__init__()
        self.dropout_p = dropout_p
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout_p))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout_p))
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout_p))

    def forward(self, x, edge_index, force_dropout=False):
        p = self.dropout_p if (self.training or force_dropout) else 0
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=p, training=(self.training or force_dropout))
        return self.convs[-1](x, edge_index)

class FlexibleGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout_p=0.3):
        super(FlexibleGraphSAGE, self).__init__()
        self.dropout_p = dropout_p
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index, force_dropout=False):
        p = self.dropout_p if (self.training or force_dropout) else 0
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=p, training=(self.training or force_dropout))
        return self.convs[-1](x, edge_index)

class APPNPNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_p=0.3, K=10, alpha=0.1):
        super(APPNPNet, self).__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin_res = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.prop1 = APPNP(K=K, alpha=alpha)
        self.dropout_p = dropout_p

    def forward(self, x, edge_index, force_dropout=False):
        p = self.dropout_p if (self.training or force_dropout) else 0
        do_train = True if force_dropout else self.training
        x = F.dropout(x, p=p, training=do_train)
        h1 = F.relu(self.lin1(x))
        h2 = F.relu(self.lin_res(F.dropout(h1, p=p, training=do_train)))
        x_res = h1 + h2 
        x = self.lin2(F.dropout(x_res, p=p, training=do_train))
        return self.prop1(x, edge_index)

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_static_data():
    nodes_df = pd.read_csv("app_data/cora_nodes.csv")
    edges_df = pd.read_csv("app_data/cora_edges.csv")
    features = torch.load("app_data/cora_features.pt", map_location='cpu')
    edge_index = torch.load("app_data/cora_edge_index.pt", map_location='cpu')
    return nodes_df, edges_df, features, edge_index

nodes_df, edges_df, features, edge_index = load_static_data()

COLOR_MAP = {
    "Theory": "#FF5733", "Reinforcement Learning": "#33FF57", 
    "Genetic Algorithms": "#3357FF", "Neural Networks": "#F333FF", 
    "Probabilistic Methods": "#FF33A1", "Case Based": "#33FFF5", "Rule Learning": "#F3FF33"
}
ui_labels = list(COLOR_MAP.keys())

def get_model(model_choice):
    if model_choice == "APPNP (Champion)":
        m = APPNPNet(1433, 64, 7)
        path = "app_data/champion_appnp.pth"
    elif model_choice == "GAT (Attention)":
        m = FlexibleGAT(1433, 32, 7, num_layers=2, heads=8) 
        path = "app_data/champion_gat.pth"
    elif model_choice == "GraphSAGE (Aggregator)":
        m = FlexibleGraphSAGE(1433, 32, 7, num_layers=3)
        path = "app_data/champion_graphsage.pth"
    else: 
        m = GCN_Standard(1433, 64, 7)
        path = "app_data/champion_gcn.pth"
    
    m.load_state_dict(torch.load(path, map_location='cpu'))
    m.eval()
    return m

# --- 3. STATE & UI CONFIG ---
st.set_page_config(layout="wide", page_title="GNN Citation Explorer")

if 'center_node' not in st.session_state:
    st.session_state.center_node = 210

# --- 4. SIDEBAR CONTROLS (MOVED UP) ---
with st.sidebar:
    st.header("🏆 Model Selection")
    # THE DROPDOWN DEFINED FIRST
    model_choice = st.selectbox(
        "Select GNN Architecture:",
        ["APPNP (Champion)", "GAT (Attention)", "GraphSAGE (Aggregator)", "GCN (Runner-up)"]
    )
    model = get_model(model_choice)
    
    st.markdown("---")
    with st.expander("Topic Color Legend", expanded=False):
        for topic, color in COLOR_MAP.items():
            st.markdown(
                f'<div style="display: flex; align-items: center; margin-bottom: 5px;">'
                f'<div style="width: 20px; height: 20px; background-color: {color}; border-radius: 50%; margin-right: 10px;"></div>'
                f'<span>{topic}</span>'
                f'</div>', 
                unsafe_allow_html=True
            )
    
    st.markdown("---")
    st.header("Network Controls")
    highlight_errors = st.checkbox("Highlight Prediction Errors", value=True)
    hops = st.slider("Neighborhood Depth (Hops):", 1, 3, 1)
    selected_id = st.number_input("Inspect Node ID:", 0, 2707, value=int(st.session_state.center_node))
    st.session_state.center_node = selected_id

# --- 5. MAIN PAGE CONTENT ---
st.title(" GNN Citation & Error Analysis")

st.markdown("""
This demo shows how **Graph Neural Networks** classify research papers in the **Cora citation network**. 
A paper is represented as a **node**, and citations are **edges** connecting papers. By leveraging both the text content of the papers and the link structure of the network, GNNs can predict the specific research field of a publication with high accuracy.

### 📖 How to use this Dashboard:
1. **Select a Model**: Choose an architecture from the sidebar to see how different GNN logics (Spectral vs. Spatial) handle the graph.
2. **Review Performance**: Check the **Dataset & Model Overview** metrics below to see the selected model's accuracy on the test set.
3. **Inspect a Node**: Use the sidebar to enter a specific **Node ID** (0-2707). The graph will center on this paper.
4. **Adjust Depth**: Slide the **Neighborhood Depth** to see 1st, 2nd, or 3rd-degree citations.
5. **Analyze Errors**: Toggle **Highlight Prediction Errors** to see where the AI struggled—incorrectly predicted nodes will appear with a thick red border.
""")

# ... followed by your st.expander(" Dataset & Model Overview") ...

with st.expander("Dataset & Model Overview", expanded=True):
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of Papers", "2,708")
    col2.metric("Citation Links", "5,429")
    col3.metric("Research Topics", "7")
    
    st.markdown("---")
    
    if model_choice == "APPNP (Champion)":
        acc, f1 = "90.77%", "91.11%"
    elif model_choice == "GAT (Attention)":
        acc, f1 = "88.10%", "87.00%"
    elif model_choice == "GraphSAGE (Aggregator)":
        acc, f1 = "88.60%", "89.00%"
    else:
        acc, f1 = "90.77%", "90.60%"
        
    c1, c2 = st.columns(2)
    c1.metric(f" {model_choice} Accuracy", acc)
    c2.metric(" Macro-F1 Score", f1)

# --- 6. PREDICTION & DIAGNOSTICS ---
with torch.no_grad():
    std_out = model(features, edge_index, force_dropout=False)
    raw_pred_idx = std_out.argmax(dim=1)[selected_id].item()
    mc_runs = [F.softmax(model(features, edge_index, force_dropout=True)[selected_id]/1.2, dim=0).numpy() for _ in range(100)]
    mean_probs = np.array(mc_runs).mean(axis=0)

actual_topic_clean = nodes_df.iloc[selected_id]['Topic_Name'].replace("_", " ")
predicted_topic_clean = ui_labels[raw_pred_idx]

# Sidebar results section
with st.sidebar:
    st.markdown("---")
    st.subheader(" Model Confidence")
    st.bar_chart(pd.DataFrame({"Topic": ui_labels, "Confidence": mean_probs}).set_index("Topic"))
    st.info(f"**Actual:** {actual_topic_clean}")
    if actual_topic_clean == predicted_topic_clean:
        st.success(f"**AI Prediction:** {predicted_topic_clean} ✅")
    else:
        st.error(f"**AI Prediction:** {predicted_topic_clean} ❌")

# --- 7. GRAPH LOGIC ---
def get_k_hop_data(center_id, edges, k):
    visited_nodes = {int(center_id)}
    last_layer = {int(center_id)}
    hop_edges = [] 
    for hop in range(1, k + 1):
        new_layer = set()
        for node in last_layer:
            t_edges = edges[edges['source'] == node]
            s_edges = edges[edges['target'] == node]
            connections = pd.concat([t_edges, s_edges])
            for _, row in connections.iterrows():
                u, v = int(row['source']), int(row['target'])
                neighbor = v if u == node else u
                hop_edges.append({'source': u, 'target': v, 'hop': hop})
                new_layer.add(neighbor)
        visited_nodes.update(new_layer)
        last_layer = new_layer
        if not last_layer: break
    return list(visited_nodes), hop_edges

nodes_to_show, edges_with_hops = get_k_hop_data(selected_id, edges_df, hops)

net = Network(height="600px", width="100%", bgcolor="#ffffff")
net.barnes_hut(gravity=-4000, spring_length=250)

for nid in nodes_to_show:
    row = nodes_df.iloc[nid]
    this_actual = row['Topic_Name'].replace("_", " ")
    this_pred = ui_labels[std_out.argmax(dim=1)[nid].item()]
    is_wrong = (this_actual != this_pred)
    
    b_width = 6 if (highlight_errors and is_wrong) else 1
    b_color = "#FF0000" if (highlight_errors and is_wrong) else "#444444"
    
    net.add_node(
        int(nid), label=f"#{nid}", title=f"Actual: {this_actual}\nPredicted: {this_pred}",
        color=COLOR_MAP.get(this_actual, "#999999"), size=45 if nid == selected_id else 25,
        borderWidth=b_width, color_border=b_color
    )

edge_styles = {1: {"color": "#666666", "width": 3}, 2: {"color": "#aaaaaa", "width": 1.5}, 3: {"color": "#dddddd", "width": 0.5}}
added_edges = set()
for edge in edges_with_hops:
    u, v = edge['source'], edge['target']
    edge_key = tuple(sorted((u, v)))
    if edge_key not in added_edges and u in nodes_to_show and v in nodes_to_show:
        style = edge_styles.get(edge['hop'], edge_styles[3])
        net.add_edge(u, v, color=style['color'], width=style['width'])
        added_edges.add(edge_key)

net.save_graph("graph.html")
components.html(open("graph.html", 'r').read(), height=650)