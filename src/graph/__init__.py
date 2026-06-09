"""
src/graph
─────────
Pacote de modelagem de grafos viários para o VigiaEnchente.

Módulos
  loader         — carregamento e pré-processamento do grafo (OSMnx)
  flood          — risco de enchente: nós de risco, penalização, grafo bloqueado
  routing        — algoritmos de caminho mínimo e comparação de rotas
  metrics        — Floyd-Warshall, betweenness, clustering hierárquico
  visualization  — todas as funções de plot (saída em assets/images/)

Scripts orquestradores (não importar diretamente)
  vigiaEnchente   — pipeline Q4 / Q5 / Q6
  analysis_blocked — análise do grafo bloqueado
  gcn             — construção do objeto PyG Data para GNN
"""

from .loader import load_graph, preprocess_graph, add_weights, build_graph
from .flood import (
    fetch_river_geometry,
    identify_flood_risk_nodes,
    find_nearest_river_nodes,
    apply_flood_penalty,
    reset_weights,
    make_blocked_graph,
)
from .routing import run_algorithms, compare_routes
from .metrics import (
    compute_network_metrics,
    compute_fw_matrix,
    compute_diameter,
    find_isolated_nodes,
    compute_hierarchical_clusters,
    report_clusters,
    compute_betweenness,
    geocode_nodes,
)
from .visualization import (
    plot_route_comparison,
    plot_fw_heatmap,
    plot_fw_heatmap_clustered,
    plot_isolated_nodes,
    plot_nearest_river_nodes,
    plot_blocked_comparison,
    plot_degree_distribution,
)

__all__ = [
    # loader
    "load_graph", "preprocess_graph", "add_weights", "build_graph",
    # flood
    "fetch_river_geometry", "identify_flood_risk_nodes", "find_nearest_river_nodes",
    "apply_flood_penalty", "reset_weights", "make_blocked_graph",
    # routing
    "run_algorithms", "compare_routes",
    # metrics
    "compute_network_metrics", "compute_fw_matrix", "compute_diameter",
    "find_isolated_nodes", "compute_hierarchical_clusters", "report_clusters",
    "compute_betweenness", "geocode_nodes",
    # visualization
    "plot_route_comparison", "plot_fw_heatmap", "plot_fw_heatmap_clustered",
    "plot_isolated_nodes", "plot_nearest_river_nodes",
    "plot_blocked_comparison", "plot_degree_distribution",
]
