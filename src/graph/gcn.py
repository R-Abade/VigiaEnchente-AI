"""
gcn.py

GCN para predição de risco de inundação em rede viária.

Node features (6):
  0  degree_norm      — grau normalizado [0,1]
  1  betweenness_norm — betweenness normalizada
  2  elevation_norm   — elevação SRTM normalizada [0,1] (baixa = alto risco)
  3  is_near_river    — 1 se distância ao rio <= threshold
  4  is_bridge        — 1 se nó está em uma ponte (OSM)
  5  highway_primary  — 1 se rodovia primária/trunk/motorway

Label y:
  river_dist_norm * elevation_risk
  elevation_risk = 1 - elevation_norm
  Nós dentro do buffer recebem y = 1.0
"""

import torch
import torch_geometric as tg
import numpy as np
from scipy.stats import spearmanr

from src.graph.loader import build_graph
from src.graph.flood import identify_flood_risk_nodes, compute_river_distance
from src.graph.features import compute_betweenness

torch.manual_seed(42)
np.random.seed(42)

PRIMARY_HIGHWAY_TYPES = {
    "motorway", "trunk", "primary",
    "motorway_link", "trunk_link", "primary_link"
}


class GCNRisk(torch.nn.Module):
    """Rede GCN de dois layers para regressão de risco por nó (6→36→1)."""

    def __init__(self):
        super().__init__()
        self.conv1 = tg.nn.GCNConv(6, 36)
        self.conv2 = tg.nn.GCNConv(36, 1)

    def forward(self, features, edge_index, edge_weight):
        features = torch.relu(self.conv1(features, edge_index, edge_weight))
        return self.conv2(features, edge_index, edge_weight)


class MLP(torch.nn.Module):
    """MLP baseline — ignora estrutura de grafo (6→36→1)."""

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(6, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 1),
        )

    def forward(self, features):
        return self.net(features)


def get_node_highway_types(graph, node):
    """Retorna os tipos highway das arestas incidentes ao nó."""
    types = set()
    for _, _, edge_data in graph.edges(node, data=True):
        road_type = edge_data.get("highway", "")
        if isinstance(road_type, list):
            types.update(road_type)
        elif road_type:
            types.add(road_type)
    return types


def build_pyg_data(place="Sabará, Minas Gerais, Brazil", threshold=50.0):
    """Constrói o objeto PyG Data com features, arestas e labels para a GCN."""
    graph = build_graph(place)
    nodes = list(graph.nodes())
    node_index = {node: position for position, node in enumerate(nodes)}
    n_nodes = len(nodes)

    # ---- Grau normalizado ----
    degrees = np.array([graph.degree(node) for node in nodes], dtype=np.float32)
    degrees_norm = degrees / degrees.max() if degrees.max() > 0 else degrees

    # ---- Distância ao rio e elevação ----
    risk_nodes, _, rivers = identify_flood_risk_nodes(graph, place, threshold=threshold)
    distance_dict = compute_river_distance(graph, place)

    river_dist = np.zeros(n_nodes, dtype=np.float32)
    raw_elevation = np.zeros(n_nodes, dtype=np.float32)

    for position, node in enumerate(nodes):
        info = distance_dict.get(node, {"dist_m": 0.0, "elevation": 0.0})
        river_dist[position] = info["dist_m"]
        raw_elevation[position] = info["elevation"] or 0.0

    river_dist_norm = np.exp(-river_dist / threshold)
    elevation_norm = raw_elevation / raw_elevation.max() if raw_elevation.max() > 0 else raw_elevation

    is_near_river = np.zeros(n_nodes, dtype=np.float32)
    for position, node in enumerate(nodes):
        if node in risk_nodes:
            is_near_river[position] = 1.0

    # ---- Betweenness ----
    betweenness_dict, _ = compute_betweenness(graph)
    betweenness = np.array([betweenness_dict.get(node, 0.0) for node in nodes], dtype=np.float32)
    betweenness_norm = betweenness / betweenness.max() if betweenness.max() > 0 else betweenness

    # ---- Ponte ----
    bridge = np.zeros(n_nodes, dtype=np.float32)
    for position, node in enumerate(nodes):
        for _, _, edge_data in graph.edges(node, data=True):
            if edge_data.get("bridge") in (True, "yes", 1):
                bridge[position] = 1.0
                break

    # ---- Rodovia primária ----
    highway = np.zeros(n_nodes, dtype=np.float32)
    for position, node in enumerate(nodes):
        if get_node_highway_types(graph, node) & PRIMARY_HIGHWAY_TYPES:
            highway[position] = 1.0

    # ---- Matriz de features ----
    features_np = np.stack([
        degrees_norm,
        betweenness_norm,
        elevation_norm,
        is_near_river,
        bridge,
        highway
    ], axis=1)
    features = torch.tensor(features_np, dtype=torch.float)

    # ---- Arestas ----
    src_nodes, dst_nodes, edge_weights = [], [], []
    for u, v, edge_data in graph.edges(data=True):
        i, j = node_index[u], node_index[v]
        weight = float(edge_data.get("weight", 1.0))
        src_nodes.extend([i, j])
        dst_nodes.extend([j, i])
        edge_weights.extend([weight, weight])

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    edge_weight = edge_weight / edge_weight.max() if edge_weight.max() > 0 else edge_weight

    # ---- Rótulo ----
    # Nota: is_near_river (feature 3) e y=1.0 usam o mesmo threshold,
    # o que gera circularidade intencional — o modelo aprende a generalizar
    # além do buffer para nós com elevação baixa e alta betweenness.
    elevation_risk = 1.0 - elevation_norm
    y_np = river_dist_norm * elevation_risk
    y_np = y_np / y_np.max() if y_np.max() > 0 else y_np

    for position, node in enumerate(nodes):
        if node in risk_nodes:
            y_np[position] = 1.0

    y = torch.tensor(y_np, dtype=torch.float)

    # ---- Dados brutos para depuração ----
    raw_data = {
        "degrees": degrees,
        "river_dist": river_dist,
        "elevation": raw_elevation,
        "betweenness": betweenness,
    }

    data = tg.data.Data(x=features, edge_index=edge_index, edge_weight=edge_weight, y=y)

    print("\n--- Dataset ---")
    print(f"Nós: {n_nodes}")
    print(f"Arestas (bidirecional): {edge_index.shape[1]}")
    print(f"Nós críticos (buffer): {len(risk_nodes)}")

    return data, nodes, raw_data, graph, risk_nodes, rivers, threshold


def evaluate(predictions, targets, mask):
    """MSE, MAE e Spearman para os nós do mask."""
    pred = predictions.squeeze()

    mse = torch.mean((pred[mask] - targets[mask]) ** 2).item()
    mae = torch.mean(torch.abs(pred[mask] - targets[mask])).item()

    pred_np = pred[mask].detach().numpy()
    y_np = targets[mask].detach().numpy()
    rho, _ = spearmanr(pred_np, y_np)

    return mse, mae, rho


def print_neighbors(node_id: int, data, raw_data, predictions):
    """Exibe um nó e todos os seus vizinhos de primeiro nível (inspeção manual)."""
    predictions = predictions.squeeze()

    mask = data.edge_index[0] == node_id
    neighbors = data.edge_index[1][mask].tolist()

    def format_line(index):
        return (
            f"  Nó {index:>4} | "
            f"pred: {predictions[index].item():.4f} | "
            f"y: {data.y[index].item():.4f} | "
            f"dist_rio: {raw_data['river_dist'][index]:.1f}m | "
            f"elevação: {raw_data['elevation'][index]:.1f}m | "
            f"grau: {int(raw_data['degrees'][index])} | "
            f"betweenness: {raw_data['betweenness'][index]:.6f}"
        )

    print(f"\n--- Nó {node_id} e seus vizinhos ---")
    print(f"[alvo] {format_line(node_id)}")

    if not neighbors:
        print("  Nenhum vizinho encontrado.")
        return

    for neighbor in sorted(set(neighbors)):
        print(f"[viz]  {format_line(neighbor)}")


def train_and_evaluate(data, nodes, raw_data, graph, risk_nodes, rivers, threshold):
    """Treina GCN + MLP baseline e retorna modelos e predições.

    Retorna também gcn_pred e mlp_pred para uso externo (ex.: visualização).
    """
    n_nodes = data.num_nodes
    edge_index = data.edge_index
    edge_weight = data.edge_weight

    # ---- Split 70/15/15 ----
    perm = torch.randperm(n_nodes)
    train_end = int(0.70 * n_nodes)
    val_end = int(0.85 * n_nodes)

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    for position in perm[:train_end]:
        train_mask[position] = True
    for position in perm[train_end:val_end]:
        val_mask[position] = True
    for position in perm[val_end:]:
        test_mask[position] = True

    # ---- GCN ----
    gcn = GCNRisk()
    optimizer = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=1e-4)

    gcn.train()
    loss_history = []

    for epoch in range(1000):
        optimizer.zero_grad()
        output = gcn(data.x, edge_index, edge_weight)
        loss = torch.nn.functional.mse_loss(output.squeeze()[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if epoch % 100 == 0:
            print(f"Época {epoch:>4} | Loss: {loss.item():.6f}")

    # ---- Avaliação GCN ----
    gcn.eval()
    with torch.no_grad():
        gcn_pred = gcn(data.x, edge_index, edge_weight)

    train_mse, train_mae, train_rho = evaluate(gcn_pred, data.y, train_mask)
    val_mse, val_mae, val_rho = evaluate(gcn_pred, data.y, val_mask)
    test_mse, test_mae, test_rho = evaluate(gcn_pred, data.y, test_mask)

    gcn_pred = gcn_pred.squeeze()

    # ---- Top 20 GCN ----
    sorted_vals, sorted_indices = torch.sort(gcn_pred, descending=True)
    top20 = sorted_indices[:20]

    print("\nTop 20 nós por risco previsto (GCN):")
    for position in top20:
        node = position.item()
        print(
            f"Nó {node:>4} | "
            f"pred: {gcn_pred[node].item():.4f} | "
            f"y: {data.y[node].item():.4f} | "
            f"rio: {raw_data['river_dist'][node]:.1f}m | "
            f"elev: {raw_data['elevation'][node]:.1f}m | "
            f"grau: {int(raw_data['degrees'][node])} | "
            f"bet: {raw_data['betweenness'][node]:.6f}"
        )

    print("\n--- Avaliação GCN ---")
    print(f"{'':6} {'MSE':>8} {'MAE':>8} {'Spearman ρ':>12}")
    print(f"{'Train':6} {train_mse:>8.4f} {train_mae:>8.4f} {train_rho:>12.4f}")
    print(f"{'Val':6}   {val_mse:>8.4f} {val_mae:>8.4f} {val_rho:>12.4f}")
    print(f"{'Test':6}  {test_mse:>8.4f} {test_mae:>8.4f} {test_rho:>12.4f}")

    # ---- MLP ----
    print("\n--- MLP (sem grafo) ---")
    mlp = MLP()
    mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01, weight_decay=1e-4)

    mlp.train()
    for epoch in range(1000):
        mlp_optimizer.zero_grad()
        output = mlp(data.x)
        loss = torch.nn.functional.mse_loss(output.squeeze()[train_mask], data.y[train_mask])
        loss.backward()
        mlp_optimizer.step()

    mlp.eval()
    with torch.no_grad():
        mlp_pred = mlp(data.x).squeeze()

    # ---- Top 20 MLP ----
    top20_mlp = torch.argsort(mlp_pred, descending=True)[:20]

    print("\nTop 20 nós por risco previsto (MLP):")
    for position in top20_mlp:
        node = position.item()
        print(
            f"Nó {node:>4} | "
            f"pred: {mlp_pred[node].item():.4f} | "
            f"y: {data.y[node].item():.4f} | "
            f"rio: {raw_data['river_dist'][node]:.1f}m | "
            f"elev: {raw_data['elevation'][node]:.1f}m | "
            f"grau: {int(raw_data['degrees'][node])} | "
            f"bet: {raw_data['betweenness'][node]:.6f}"
        )

    mlp_train = evaluate(mlp_pred.unsqueeze(1), data.y, train_mask)
    mlp_val = evaluate(mlp_pred.unsqueeze(1), data.y, val_mask)
    mlp_test = evaluate(mlp_pred.unsqueeze(1), data.y, test_mask)

    print(f"\n{'':6} {'MSE':>8} {'MAE':>8} {'Spearman ρ':>12}")
    print(f"{'Train':6} {mlp_train[0]:>8.4f} {mlp_train[1]:>8.4f} {mlp_train[2]:>12.4f}")
    print(f"{'Val':6}   {mlp_val[0]:>8.4f} {mlp_val[1]:>8.4f} {mlp_val[2]:>12.4f}")
    print(f"{'Test':6}  {mlp_test[0]:>8.4f} {mlp_test[1]:>8.4f} {mlp_test[2]:>12.4f}")

    return gcn, mlp, gcn_pred, mlp_pred, loss_history
