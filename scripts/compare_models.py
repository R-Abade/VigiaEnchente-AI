"""
compare_models.py

Comparação sistemática para a Tabela 4 do artigo:
  - Heurística contínua (sem saturação)
  - MLP completo (6 features)
  - MLP sem is_near_river (5 features)
  - GCN sem is_near_river (5 features)
  - GCN completo (6 features)

Uso:
    python -m scripts.compare_models
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch_geometric as tg
import numpy as np
from scipy.stats import spearmanr

from src.graph.gcn import build_pyg_data, GCNRisk, MLP

SEED = 42
EPOCHS = 1000
LR = 0.01
WEIGHT_DECAY = 1e-4

# Índices das features na matriz x (ordem definida em build_pyg_data):
#   0 degrees_norm | 1 betweenness | 2 elevation_norm
#   3 is_near_river | 4 bridge | 5 highway
ABLATED_FEATURE = [0, 1, 2, 4, 5]  # sem is_near_river


# ── Métricas ──────────────────────────────────────────────────────────────────

def precision_recall_at_k(predictions, risk_node_positions, k=8):
    """P@k e R@k sobre o ranking de todos os nós."""
    topk = torch.topk(predictions, k).indices.tolist()
    hits = len(set(topk) & risk_node_positions)
    precision = hits / k
    recall = hits / len(risk_node_positions)
    return precision, recall


def full_eval(predictions, y_true, mask, risk_node_positions, k=8):
    """MSE, MAE, Spearman, P@k e R@k."""
    predictions = predictions.squeeze()

    mse = torch.mean((predictions[mask] - y_true[mask]) ** 2).item()
    mae = torch.mean(torch.abs(predictions[mask] - y_true[mask])).item()

    pred_np = predictions[mask].detach().numpy()
    y_np = y_true[mask].detach().numpy()
    rho, _ = spearmanr(pred_np, y_np)

    p_at_k, r_at_k = precision_recall_at_k(predictions.detach(), risk_node_positions, k=k)

    return {"mse": mse, "mae": mae, "rho": rho, "p_at_k": p_at_k, "r_at_k": r_at_k}


def format_row(name, metrics):
    return (f"{name:<30} MSE={metrics['mse']:.4f}  MAE={metrics['mae']:.4f}  "
            f"rho={metrics['rho']:.4f}  P@8={metrics['p_at_k']:.2f}  R@8={metrics['r_at_k']:.2f}")


# ── Treino ────────────────────────────────────────────────────────────────────

def train_gcn(features, edge_index, edge_weight, y_true, train_mask, in_channels, hidden=36):
    """Treina um GCN com in_channels variável."""
    torch.manual_seed(SEED)

    if in_channels == 6:
        model = GCNRisk()
    else:
        model = _make_gcn(in_channels, hidden)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        output = model(features, edge_index, edge_weight)
        loss = torch.nn.functional.mse_loss(output.squeeze()[train_mask], y_true[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(features, edge_index, edge_weight)

    return predictions


def _make_gcn(in_channels, hidden):
    """GCN com in_channels variável (mesma arquitetura do GCNRisk)."""
    class _GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = tg.nn.GCNConv(in_channels, hidden)
            self.conv2 = tg.nn.GCNConv(hidden, 1)

        def forward(self, features, edge_index, edge_weight):
            features = torch.relu(self.conv1(features, edge_index, edge_weight))
            return self.conv2(features, edge_index, edge_weight)

    return _GCN()


def train_mlp(features, y_true, train_mask, in_channels, hidden=36):
    """Treina um MLP com in_channels variável."""
    torch.manual_seed(SEED)

    class _MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_channels, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, 1),
            )

        def forward(self, features):
            return self.net(features)

    model = _MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        output = model(features)
        loss = torch.nn.functional.mse_loss(output.squeeze()[train_mask], y_true[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(features)

    return predictions


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ---- 1. Dataset ----
    data, nodes, raw_data, graph, risk_nodes, rivers, threshold = build_pyg_data()

    n_nodes = data.num_nodes
    node_index = {node: position for position, node in enumerate(nodes)}
    risk_node_positions = {node_index[node] for node in risk_nodes if node in node_index}

    assert len(risk_node_positions) == 8, (
        f"Esperado 8 nós no buffer, encontrado {len(risk_node_positions)} — "
        "verifique identify_flood_risk_nodes / threshold."
    )

    # ---- 2. Split (mesma seed para todos) ----
    torch.manual_seed(SEED)
    perm = torch.randperm(n_nodes)
    train_end = int(0.70 * n_nodes)
    val_end = int(0.85 * n_nodes)

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    train_mask[perm[:train_end]] = True
    val_mask[perm[train_end:val_end]] = True
    test_mask[perm[val_end:]] = True

    print(f"Split: train={train_mask.sum().item()} "
          f"val={val_mask.sum().item()} test={test_mask.sum().item()}")

    n_critical_train = sum(1 for p in risk_node_positions if train_mask[p])
    n_critical_val = sum(1 for p in risk_node_positions if val_mask[p])
    n_critical_test = sum(1 for p in risk_node_positions if test_mask[p])

    print(f"Nós críticos por split: train={n_critical_train} "
          f"val={n_critical_val} test={n_critical_test} (de 8 totais)")

    x_full = data.x
    y_true = data.y
    edge_index = data.edge_index
    edge_weight = data.edge_weight
    x_ablated = x_full[:, ABLATED_FEATURE]

    results = {}

    # ---- 3. Heurística contínua ----
    river_dist = torch.tensor(raw_data["river_dist"], dtype=torch.float32)
    river_dist_norm = torch.exp(-river_dist / threshold)
    elevation_norm = x_full[:, 2]
    elevation_risk = 1.0 - elevation_norm

    heuristic_raw = river_dist_norm * elevation_risk
    heuristic_pred = heuristic_raw / heuristic_raw.max()

    results["Heurística contínua"] = full_eval(
        heuristic_pred, y_true, test_mask, risk_node_positions
    )

    # ---- 4. MLP completo (6 features) ----
    mlp_full_pred = train_mlp(x_full, y_true, train_mask, in_channels=6)
    results["MLP completo (6 feat.)"] = full_eval(
        mlp_full_pred, y_true, test_mask, risk_node_positions
    )

    # ---- 5. MLP sem is_near_river (5 features) ----
    mlp_ablated_pred = train_mlp(x_ablated, y_true, train_mask, in_channels=5)
    results["MLP sem is_near_river (5 feat.)"] = full_eval(
        mlp_ablated_pred, y_true, test_mask, risk_node_positions
    )

    # ---- 6. GCN sem is_near_river (5 features) ----
    gcn_ablated_pred = train_gcn(
        x_ablated, edge_index, edge_weight, y_true, train_mask, in_channels=5
    )
    results["GCN sem is_near_river (5 feat.)"] = full_eval(
        gcn_ablated_pred, y_true, test_mask, risk_node_positions
    )

    # ---- 7. GCN completo (6 features) ----
    gcn_full_pred = train_gcn(
        x_full, edge_index, edge_weight, y_true, train_mask, in_channels=6
    )
    results["GCN completo (6 feat.)"] = full_eval(
        gcn_full_pred, y_true, test_mask, risk_node_positions
    )

    # ---- Relatório ----
    print("\n" + "=" * 78)
    print("RESULTADOS — Tabela 4 (mesmo split, mesma seed)")
    print("=" * 78)

    for name, metrics in results.items():
        print(format_row(name, metrics))


if __name__ == "__main__":
    main()
