"""
features.py

Funções de feature engineering usadas no pipeline de treino da GCN
e potencialmente em inferência (runtime do serviço).

Contém apenas o que é necessário para construir o vetor de features
por nó — sem análises offline, sem Floyd-Warshall, sem geocoding.
"""

import warnings
import networkx as nx
from pyproj import Transformer

warnings.filterwarnings("ignore")


def compute_betweenness(G, top_n=10):
    """Centralidade de intermediação de todos os nós.

    Retorna o dicionário completo (necessário como feature de treino)
    e os top_n nós para inspeção rápida.
    """
    G_simple = nx.Graph(G)
    bet = nx.betweenness_centrality(G_simple, k=len(G_simple), weight="weight", seed=42)
    top = sorted(bet.items(), key=lambda x: x[1], reverse=True)[:top_n]

    print(f"\nTop {top_n} betweenness:")
    for node, value in top:
        print(f"  Nó {node:>6}: {value:.6f}")
    return bet, top


def utm_to_latlon(G, node):
    """Converte coordenadas UTM de um nó do grafo para lat/lon (EPSG:4326).

    Lê o CRS do grafo — funciona para qualquer zona UTM armazenada
    no atributo graph['crs'].
    """
    crs = G.graph.get("crs", "EPSG:32723")
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(G.nodes[node]["x"], G.nodes[node]["y"])
    return lat, lon
