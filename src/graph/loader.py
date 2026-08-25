"""
loader.py

Carregamento e pré-processamento do grafo viário via OSMnx.
"""

import warnings
import networkx as nx
import osmnx as ox

warnings.filterwarnings("ignore")


def load_graph(place_name: str):
    """Baixa o grafo viário (modo drive) para place_name e projeta em UTM."""
    ox.settings.all_oneway = True
    G = ox.graph_from_place(place_name, network_type="drive")
    G = ox.project_graph(G)
    return G


def preprocess_graph(G):
    """Consolida interseções, converte para não-direcionado e mantém o maior componente."""
    G = ox.consolidate_intersections(G, tolerance=15, rebuild_graph=True, dead_ends=False)
    G = G.to_undirected()
    largest = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest).copy()
    print(f"Nodes: {len(G.nodes())}  |  Edges: {len(G.edges())}")
    return G


def add_weights(G):
    """Inicializa weight = length para todas as arestas."""
    for _, _, edge_data in G.edges(data=True):
        edge_data["weight"] = edge_data.get("length", 1.0)
    return G


def build_graph(place_name: str):
    """Pipeline completo: load → preprocess → add_weights."""
    return add_weights(preprocess_graph(load_graph(place_name)))