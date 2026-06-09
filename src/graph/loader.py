"""
loader.py
─────────
Carregamento e pré-processamento do grafo viário via OSMnx.

Responsabilidades
  • Baixar o grafo de uma cidade/região pelo nome
  • Consolidar interseções, converter para não-direcionado e extrair o maior componente
  • Inicializar o atributo 'weight' de cada aresta com o comprimento real (metros)

Uso típico
    from loader import load_graph, preprocess_graph, add_weights
    G = add_weights(preprocess_graph(load_graph("Sabará, Minas Gerais, Brazil")))
"""

import warnings

import networkx as nx
import osmnx as ox

warnings.filterwarnings("ignore")


def load_graph(place_name: str):
    """
    Baixa o grafo viário (modo 'drive') para *place_name* e o projeta em
    coordenadas métricas (UTM).

    Parâmetros
    ──────────
    place_name : nome do lugar reconhecido pelo Nominatim (ex: "Sabará, MG, Brazil")

    Retorna
    ───────
    G : MultiDiGraph projetado (CRS métrico)
    """
    print(f"Downloading graph for {place_name}...")
    ox.settings.all_oneway = True
    G = ox.graph_from_place(place_name, network_type="drive")
    G = ox.project_graph(G)
    return G


def preprocess_graph(G):
    """
    Consolida interseções próximas, converte para grafo não-direcionado e
    mantém apenas o maior componente conectado.

    Parâmetros
    ──────────
    G : MultiDiGraph projetado retornado por load_graph()

    Retorna
    ───────
    G : Graph (não-direcionado, maior componente)
    """
    G_clean = ox.consolidate_intersections(
        G, tolerance=15, rebuild_graph=True, dead_ends=False
    )
    G = G_clean.to_undirected()
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    print(f"Nodes: {len(G.nodes())}  |  Edges: {len(G.edges())}")
    return G


def add_weights(G):
    """
    Inicializa weight = length para todas as arestas.
    Funciona tanto para Graph quanto para MultiGraph (usa keys=True).

    Parâmetros
    ──────────
    G : grafo NetworkX (modificado in-place e retornado)

    Retorna
    ───────
    G : mesmo grafo com atributo 'weight' definido em todas as arestas
    """
    for *_, data in G.edges(keys=True, data=True):
        data["weight"] = data.get("length", 1.0)
    return G


def build_graph(place_name: str):
    """
    Atalho que executa o pipeline completo: load → preprocess → add_weights.

    Parâmetros
    ──────────
    place_name : nome do lugar

    Retorna
    ───────
    G : grafo pronto para análise
    """
    return add_weights(preprocess_graph(load_graph(place_name)))
