"""
vigiaEnchente.py

Orquestrador das análises Q4/Q5/Q6 do artigo:
  Q4 — Penalização de arestas próximas ao Rio das Velhas e comparação de rotas
  Q5 — Identificação dos nós mais isolados (Floyd-Warshall)
  Q6 — Análise de clusters / sub-regiões

Este script é exclusivamente para uso offline (análise para artigo).
Não deve ser importado pelo microsserviço FastAPI.

Uso:
    python -m scripts.vigiaEnchente
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.graph.loader import build_graph
from src.graph.flood import identify_flood_risk_nodes
from src.graph.routing import compare_routes, run_algorithms
from src.graph.features import utm_to_latlon
from scripts.legacy_metrics import (
    compute_network_metrics,
    compute_fw_matrix,
    compute_diameter,
    find_isolated_nodes,
    compute_hierarchical_clusters,
    report_clusters,
    geocode_nodes,
)
from src.graph.visualization import (
    plot_route_comparison,
    plot_fw_heatmap,
    plot_fw_heatmap_clustered,
    plot_isolated_nodes,
    plot_degree_distribution,
)


def main(
    place: str = "Sabará, Minas Gerais, Brazil",
    threshold: float = 50.0,
    cluster_threshold_pct: float = 0.80,
    top_isolated: int = 8,
):
    """
    Parâmetros
    ──────────
    place                 : município para download do grafo OSM
    threshold             : distância de risco ao rio em metros
    cluster_threshold_pct : percentil para classificar nós como isolados
    top_isolated          : número de nós isolados a exibir
    """

    # Grafo
    G = build_graph(place)
    compute_network_metrics(G)

    # Floyd-Warshall
    matrix, nodes = compute_fw_matrix(G)
    source, target, _, matrix, nodes = compute_diameter(G, matrix=matrix, nodes=nodes)

    # Risco + penalização + comparação de rotas
    risk_nodes, risk_info, rivers = identify_flood_risk_nodes(G, place, threshold)

    cmp = compare_routes(G, source, target, risk_nodes)

    plot_route_comparison(
        G, cmp, risk_nodes, risk_info, rivers,
        threshold=threshold,
        filename="route_comparison.png",
    )

    # Nós mais isolados
    isolated_nodes, avg_dist = find_isolated_nodes(
        matrix, nodes,
        top_n=top_isolated,
        threshold_pct=cluster_threshold_pct,
    )

    # Geocoding reverso dos nós isolados
    print("\n  Geocoding reverso dos nós isolados ")
    iso_ids = [n for n, _ in isolated_nodes]
    locations = geocode_nodes(G, iso_ids)
    loc_by_node = {entry["node"]: entry for entry in locations}

    print(f"\n{'─' * 78}")
    print(f" TOP {top_isolated} NÓS MAIS ISOLADOS")
    print(f"{'─' * 78}")
    print(f"  {'Rank':>4}  {'Nó':>10}  {'Dist. média (km)':>16}  {'Lat':>10}  {'Lon':>11}  Endereço")
    for rank, (node, d) in enumerate(isolated_nodes, 1):
        lat, lon = utm_to_latlon(G, node)
        address = loc_by_node.get(node, {}).get("address", "—")
        print(f"  {rank:>4}  {node:>10}  {d / 1000:>16.3f}  {lat:>10.5f}  {lon:>11.5f}  {address}")

    # Clustering hierárquico
    finite_max = float(np.nanmax(matrix[matrix != np.inf]))
    matrix_vis = np.where(matrix == np.inf, finite_max * 1.2, matrix)

    labels, order, n_clusters = compute_hierarchical_clusters(matrix_vis, nodes)
    report_clusters(matrix_vis, nodes, labels, n_clusters, G=G)

    # Heatmaps
    plot_fw_heatmap(matrix, place_name=place, filename="fw_heatmap.png")

    plot_fw_heatmap_clustered(
        matrix_vis, nodes, isolated_nodes, labels, order, n_clusters,
        cluster_threshold_pct=cluster_threshold_pct,
        filename="fw_heatmap.png",
    )

    plot_isolated_nodes(
        G, matrix, nodes, isolated_nodes,
        rivers=rivers,
        filename="nos_isolados.png",
    )

    # Demais análises
    run_algorithms(G, source, target)

    plot_degree_distribution(G, filename="degree_distribution.png")


if __name__ == "__main__":
    main()
