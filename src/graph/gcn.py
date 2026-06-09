"""
gcn.py

Node features (x) — 7 features
  0  degree_norm     — normalized degree [0,1]
  1  river_dist_norm — exp(-dist/threshold), close to river → 1.0
  2  betweenness_norm— normalized betweenness centrality
  3  elevation_norm  — SRTM elevation normalized [0,1] (low → high risk)
  4  is_bridge       — 1 if node is on a bridge (OSM)
  5  is_near_river   — 1 if distance <= threshold (critical node)
  6  highway_primary — 1 if major road (primary/trunk/motorway)

Label y
  river_dist_norm * elevation_risk
  elevation_risk = 1 - elevation_norm  (low elevation = high risk)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch_geometric as tg
import numpy as np
import visualization

from loader import build_graph, load_graph
from flood import identify_flood_risk_nodes, compute_river_distance
from metrics import compute_betweenness

torch.manual_seed(42)
np.random.seed(42)

PRIMARY_HIGHWAY_TYPES = {
    "motorway", "trunk", "primary",
    "motorway_link", "trunk_link", "primary_link"
}


class GCNRisk(torch.nn.Module):
    def __init__(self):
        super(GCNRisk, self).__init__()
        self.conv1 = tg.nn.GCNConv(6, 32)
        self.conv2 = tg.nn.GCNConv(32, 16)
        self.conv3 = tg.nn.GCNConv(16, 1)

    def forward(self, x, edge_index, edge_weight):
        x = torch.relu(self.conv1(x, edge_index, edge_weight))
        x = torch.relu(self.conv2(x, edge_index, edge_weight))
        x = self.conv3(x, edge_index, edge_weight)
        return x


def _get_node_highway_types(graph, node):
    """Return a set of highway types from edges incident to the node."""
    highway_types = set()
    for _, _, edge_data in graph.edges(node, data=True):
        road_type = edge_data.get("highway", "")

        if isinstance(road_type, list):
            highway_types.update(road_type)

        elif road_type:
            highway_types.add(road_type)

    return highway_types


def build_pyg_data(place="Sabará, Minas Gerais, Brazil", threshold=50.0):
    graph = build_graph(place)
    nodes = list(graph.nodes())
    node_index = {node: i for i, node in enumerate(nodes)}

    # ---- Normalized degree ----
    degrees = np.zeros(len(nodes), dtype=np.float32)

    for i, node in enumerate(nodes):
        degrees[i] = graph.degree(node)
    max_degree = degrees.max()

    if max_degree > 0:
        degrees_norm = degrees / max_degree
    else:
        degrees_norm = degrees

    # ---- River distance, elevation, and binary risk flag ----
    risk_nodes, _, rivers = identify_flood_risk_nodes(graph, place, threshold=threshold)
    distance_dict = compute_river_distance(graph, place)

    river_dist = np.zeros(len(nodes), dtype=np.float32)
    raw_elevation = np.zeros(len(nodes), dtype=np.float32)

    for i, node in enumerate(nodes):
        info = distance_dict.get(node, {"dist_m": 0.0, "elevation": 0.0})
        river_dist[i] = info["dist_m"]
        elevation = info["elevation"]
        if elevation is None:
            elevation = 0.0
        raw_elevation[i] = elevation

    river_dist_norm = np.exp(-river_dist / threshold)

    max_elevation = raw_elevation.max()

    if max_elevation > 0:
        elevation_norm = raw_elevation / max_elevation
    else:
        elevation_norm = raw_elevation

    is_near_river = np.zeros(len(nodes), dtype=np.float32)

    for i, node in enumerate(nodes):
        if node in risk_nodes:
            is_near_river[i] = 1.0

    # ---- Betweenness centrality ----
    betweenness_dict, _ = compute_betweenness(graph)
    betweenness = np.zeros(len(nodes), dtype=np.float32)

    for i, node in enumerate(nodes):
        betweenness[i] = betweenness_dict.get(node, 0.0)
    max_bet = betweenness.max()

    if max_bet > 0:
        betweenness = betweenness / max_bet

    # ---- Is Bridge?  ----
    bridge= np.zeros(len(nodes), dtype=np.float32)

    for i, node in enumerate(nodes):
        has_bridge = False

        for _, _, edge_data in graph.edges(node, data=True):
            bridge_val = edge_data.get("bridge")

            if bridge_val in (True, "yes", 1):
                has_bridge = True
                break

        if has_bridge:
            bridge[i] = 1.0

    # ---- Primary highway indicator ----
    highway = np.zeros(len(nodes), dtype=np.float32)

    for i, node in enumerate(nodes):
        types_set = _get_node_highway_types(graph, node)

        if types_set & PRIMARY_HIGHWAY_TYPES:
            highway[i] = 1.0

    # ---- Build feature matrix (N x 7) ----
    x_np = np.stack([
        degrees_norm, betweenness,
        elevation_norm, bridge, is_near_river, highway
    ], axis=1)
    x = torch.tensor(x_np, dtype=torch.float)

    # ---- Edges and edge weights ----
    src_nodes = []
    dst_nodes = []
    edge_weights = []

    for u, v, edge_data in graph.edges(data=True):
        i = node_index[u]
        j = node_index[v]
        w = float(edge_data.get("weight", 1.0))
        src_nodes.append(i)
        dst_nodes.append(j)
        edge_weights.append(w)
        src_nodes.append(j)
        dst_nodes.append(i)
        edge_weights.append(w)

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    max_weight = edge_weight.max()

    if max_weight > 0:
        edge_weight = edge_weight / max_weight

    # ---- Label y = river_dist_norm * (1 - elevation_norm) ----
    elevation_risk = 1.0 - elevation_norm
    y_np = river_dist_norm * elevation_risk
    max_y = y_np.max()

    if max_y > 0:
        y_np = y_np / max_y

    for i, node in enumerate(nodes):

        if node in risk_nodes:
            y_np[i] = 1.0

    y = torch.tensor(y_np, dtype=torch.float)

    # ---- Raw data for debugging ----
    raw_data = {
        "degrees": degrees,
        "river_dist": river_dist,
        "elevation": raw_elevation,
        "betweenness": betweenness,
    }

    print("\n---------- Dataset ----------")
    print(f"Nodes: {x.shape[0]}")
    print(f"Edges: {edge_index.shape[1]}")
    print(f"Critical nodes: {len(risk_nodes)}")

    data = tg.data.Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)

    print("\n---------- Dataset ----------")
    print(f"Nodes: {x.shape[0]}")
    print(f"Edges: {edge_index.shape[1]}")
    print(f"Critical nodes: {len(risk_nodes)}")

    return data, nodes, raw_data, graph, risk_nodes, rivers, threshold


def evaluate(predictions, targets, mask):
    pred = predictions.squeeze()
    mse = torch.mean((pred[mask] - targets[mask]) ** 2).item()
    mae = torch.mean(torch.abs(pred[mask] - targets[mask])).item()

    return mse, mae


def print_neighbors(index_node: int, data, data_raw, pred):
    """
    Exibe um nó e todos os seus vizinhos de primeiro nível.
    """
    pred = pred.squeeze()

    mask = data.edge_index[0] == index_node
    neighbors = data.edge_index[1][mask].tolist()

    def formater_line(index):
        return (
            f"  Nó {index:>4} | "
            f"pred: {pred[index].item():.4f} | "
            f"y: {data.y[index].item():.4f} | "
            f"dist_rio: {data_raw['river_dist'][index]:.1f}m | "
            f"elevação: {data_raw['elevation'][index]:.1f}m | "
            f"grau: {int(data_raw['degrees'][index])} | "
            f"betweenness: {data_raw['betweenness'][index]:.6f}"
        )

    print(f"\n---------- Nó {index_node} e seus neighbors ----------")
    print(f"[alvo] {formater_line(index_node)}")

    if not neighbors:
        print("  Nenhum vizinho encontrado.")
        return

    for n in sorted(set(neighbors)):
        print(f"[viz]  {formater_line(n)}")


def loss_optimization(data, nodes, raw_data, graph, risk_nodes, rivers, threshold):
    n_nodes = data.num_nodes
    edge_index = data.edge_index
    edge_weight = data.edge_weight

    # Shuffle and split 70/15/15
    perm = torch.randperm(n_nodes)
    train_end = int(0.70 * n_nodes)
    val_end = int(0.85 * n_nodes)

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    for i in perm[:train_end]:
        train_mask[i] = True

    for i in perm[train_end:val_end]:
        val_mask[i] = True

    for i in perm[val_end:]:
        test_mask[i] = True

    # Model and optimizer
    gcn = GCNRisk()
    optimizer = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=1e-4)

    # Training
    gcn.train()
    loss_history = []

    for epoch in range(1000):
        optimizer.zero_grad()
        output = gcn(data.x, edge_index, edge_weight)
        loss = torch.nn.functional.mse_loss(
            output.squeeze()[train_mask],
            data.y[train_mask]
        )
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch:>4} | Loss: {loss.item():.6f}")
        
    # Evaluation
    gcn.eval()
    with torch.no_grad():
        pred = gcn(data.x, edge_index, edge_weight)

    train_mse, train_mae = evaluate(pred, data.y, train_mask)
    val_mse, val_mae = evaluate(pred, data.y, val_mask)
    test_mse, test_mae = evaluate(pred, data.y, test_mask)

    pred = pred.squeeze()

    # Top 20 predictions
    sorted_vals, sorted_indices = torch.sort(pred, descending=True)
    top20 = sorted_indices[:20]
    print("\nTop 20 nodes by predicted risk:")

    for node in top20:
        i = node.item()
        print(
            f"Node {i:>4} | "
            f"pred: {pred[i].item():.4f} | "
            f"y: {data.y[i].item():.4f} | "
            f"river_dist: {raw_data['river_dist'][i]:.1f}m | "
            f"elevation: {raw_data['elevation'][i]:.1f}m | "
            f"degree: {int(raw_data['degrees'][i])} | "
            f"betweenness: {raw_data['betweenness'][i]:.6f}"
        )

    print("\n---------- Final Evaluation ----------")
    print(f"Train MSE: {train_mse:.4f} | MAE: {train_mae:.4f}")
    print(f"Val MSE: {val_mse:.4f} | MAE: {val_mae:.4f}")
    print(f"Test MSE: {test_mse:.4f} | MAE: {test_mae:.4f}")

    # ---------- Visualizações GCN ----------
    pred_np  = pred.detach().numpy()
    y_np     = data.y.numpy()

    visualization.plot_graph_on_map(
        G=graph,
        nodes=nodes,
        risk_scores=pred_np,
        risk_nodes=risk_nodes,
        rivers=rivers,
        threshold=threshold,
    )
    visualization.plot_top_risk_nodes(
        G=graph,
        nodes=nodes,
        risk_scores=pred_np,
        y_true=y_np,
        risk_nodes=risk_nodes,
        rivers=rivers,
        top_n=20,
        threshold=threshold,
    )

    visualization.plot_training_curve(
        loss_history=loss_history,
        filename="gcn_training_curve.png"
    )

    # ---------- Vizinhos ----------
    print_neighbors(919, data, raw_data, pred)
    print_neighbors(11,  data, raw_data, pred)

def main():
    data, nodes, raw_data, graph, risk_nodes, rivers, threshold = build_pyg_data()
    loss_optimization(data, nodes, raw_data, graph, risk_nodes, rivers, threshold)


if __name__ == "__main__":
    main()