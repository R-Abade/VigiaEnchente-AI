"""
train_gcn.py

CLI para treinar a GCN de risco de enchente e salvar os pesos.

Uso:
    python -m scripts.train_gcn
    python -m scripts.train_gcn --place "Sabará, Minas Gerais, Brazil" --threshold 50 --epochs 1000

Saída:
    models/gcn_risk.pt          — state_dict do modelo treinado
    data/output/baseline.txt    — métricas e top-20 para referência

Nota: o treinamento usa semente fixa (42) para reprodutibilidade.
"""

import argparse
import sys
from pathlib import Path

# Garante que o pacote src é encontrado ao rodar como `python -m scripts.train_gcn`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.graph.gcn import build_pyg_data, train_and_evaluate
from src.graph.visualization import (
    plot_graph_on_map,
    plot_top_risk_nodes,
    plot_training_curve,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treina a GCN de risco de enchente para Sabará/MG."
    )
    parser.add_argument(
        "--place",
        default="Sabará, Minas Gerais, Brazil",
        help="Localidade para download do grafo OSM (padrão: Sabará, MG).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="Buffer de risco em metros ao Rio das Velhas (padrão: 50m).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Número de épocas de treino (padrão: 1000).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Treinando GCN — place={args.place!r}  threshold={args.threshold}m  epochs={args.epochs}")

    # ---- Construção do dataset ----
    data, nodes, raw_data, graph, risk_nodes, rivers, threshold = build_pyg_data(
        place=args.place, threshold=args.threshold
    )

    # ---- Treino ----
    gcn, mlp, gcn_pred, mlp_pred, loss_history = train_and_evaluate(
        data, nodes, raw_data, graph, risk_nodes, rivers, threshold
    )

    # ---- Salvar pesos ----
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "gcn_risk.pt"
    torch.save(gcn.state_dict(), model_path)
    print(f"\nModelo salvo em: {model_path}")

    # ---- Visualizações ----
    gcn_pred_np = gcn_pred.detach().numpy()
    y_np = data.y.numpy()

    plot_graph_on_map(
        graph=graph,
        nodes=nodes,
        risk_scores=gcn_pred_np,
        risk_nodes=risk_nodes,
        rivers=rivers,
        threshold=threshold,
    )

    plot_top_risk_nodes(
        graph=graph,
        nodes=nodes,
        risk_scores=gcn_pred_np,
        y_true=y_np,
        risk_nodes=risk_nodes,
        rivers=rivers,
        top_n=20,
        threshold=threshold,
    )

    plot_training_curve(
        loss_history=loss_history,
        filename="gcn_training_curve.png",
    )

    # ---- Inspecionar vizinhos (descomentado para debug) ----
    # from src.graph.gcn import print_neighbors
    # print_neighbors(919, data, raw_data, gcn_pred)
    # print_neighbors(11, data, raw_data, gcn_pred)


if __name__ == "__main__":
    main()
