"""
metrics.py

Métricas de rede para o grafo viário.
"""

import time
import warnings
import networkx as nx
import numpy as np

warnings.filterwarnings("ignore")


def compute_network_metrics(G):
    """Estatísticas básicas da rede."""
    G_simple = nx.Graph(G)
    degrees = [d for _, d in G_simple.degree()]
    stats = {
        "num_nodes": len(G_simple.nodes()),
        "num_edges": len(G_simple.edges()),
        "avg_degree": sum(degrees) / len(degrees),
        "density": nx.density(G_simple),
        "avg_clustering": nx.average_clustering(G_simple),
        "num_components": nx.number_connected_components(G_simple),
    }
    print("\n=== Métricas da Rede ===")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    return stats


def compute_fw_matrix(G):
    """Matriz de distâncias Floyd-Warshall."""
    nodes = list(G.nodes())
    print(f"Calculando Floyd-Warshall para {len(nodes)} nós...")
    t0 = time.time()
    matrix = nx.floyd_warshall_numpy(G, weight="weight")
    np.fill_diagonal(matrix, 0)
    print(f"  Tempo: {time.time() - t0:.2f}s")
    return matrix, nodes


def compute_diameter(G, matrix=None, nodes=None):
    """Par de nós mais distantes (diâmetro)."""
    if matrix is None:
        matrix, nodes = compute_fw_matrix(G)

    finite = matrix != np.inf
    max_dist = float(np.nanmax(matrix[finite]))
    u_idx, v_idx = np.unravel_index(np.where(finite, matrix, -1).argmax(), matrix.shape)

    print(f"  Diâmetro: {max_dist:.2f} m  |  {nodes[u_idx]} → {nodes[v_idx]}")
    return nodes[u_idx], nodes[v_idx], max_dist, matrix, nodes


def find_isolated_nodes(matrix, nodes, top_n=8, threshold_pct=0.80):
    """Nós com maior distância média a todos os demais."""
    avg_dist = matrix.mean(axis=1)
    threshold = np.percentile(avg_dist, threshold_pct * 100)
    top_indices = np.argsort(avg_dist)[::-1][:top_n]
    isolated = [(nodes[i], float(avg_dist[i])) for i in top_indices]

    print(f"\nQ5 — Top {top_n} nós isolados (média > {threshold/1000:.2f} km)")
    for rank, (node, d) in enumerate(isolated, 1):
        print(f"  #{rank}: nó {node}  |  {d/1000:.3f} km")

    return isolated, avg_dist


def compute_hierarchical_clusters(matrix, nodes, n_clusters=None):
    """Clustering hierárquico de Ward sobre a matriz FW."""
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial.distance import squareform

    N = len(nodes)
    if n_clusters is None:
        n_clusters = max(2, int(np.sqrt(N) / 2))

    condensed = squareform(matrix, checks=False)
    Z = linkage(condensed, method="ward")
    dend = dendrogram(Z, no_plot=True)
    labels = fcluster(Z, n_clusters, criterion="maxclust")

    print(f"\nQ6 — Clusters: {n_clusters}")
    return labels, dend["leaves"], n_clusters


def report_clusters(matrix, nodes, labels, n_clusters, G=None):
    """Relatório de clusters: distâncias intra/inter e razão de isolamento."""
    node_index = {n: i for i, n in enumerate(nodes)}
    unique = np.unique(labels)

    print(f"\n{'─'*60}")
    print(f"  ANÁLISE DE CLUSTERS ({n_clusters} clusters)")
    print(f"{'─'*60}")

    for c in unique:
        members = [nodes[i] for i, lbl in enumerate(labels) if lbl == c]
        m_idx = [node_index[m] for m in members]
        e_idx = [node_index[m] for m in nodes if m not in set(members)]

        sub = matrix[np.ix_(m_idx, m_idx)]
        ext = matrix[np.ix_(m_idx, e_idx)]

        intra = float(sub[sub > 0].mean()) / 1000 if sub[sub > 0].size else 0.0
        inter = float(ext.mean()) / 1000 if ext.size else 0.0
        ratio = intra / inter if inter > 0 else float("nan")

        intra_means = sub.mean(axis=1)
        centroid = members[int(np.argmin(intra_means))]

        flag = " ← ALTO ISOLAMENTO" if ratio < 0.6 else ""
        print(f"\n  Cluster {c}: {len(members)} nós")
        print(f"    Intra: {intra:.3f} km  |  Inter: {inter:.3f} km  |  Razão: {ratio:.3f}{flag}")
        print(f"    Centroide: {centroid}")


def compute_betweenness(G, top_n=10):
    """Centralidade de intermediação."""
    G_simple = nx.Graph(G)
    bet = nx.betweenness_centrality(G_simple, k=len(G_simple), weight="weight", seed=42)
    top = sorted(bet.items(), key=lambda x: x[1], reverse=True)[:top_n]

    print(f"\nTop {top_n} betweenness:")
    for node, value in top:
        print(f"  Nó {node:>6}: {value:.6f}")
    return bet, top


def _utm_to_latlon(G, node):
    from pyproj import Transformer
    crs = G.graph.get("crs", "EPSG:32723")
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(G.nodes[node]["x"], G.nodes[node]["y"])
    return lat, lon


def geocode_nodes(G, node_ids):
    """Geocoding reverso via Nominatim."""
    import time
    from geopy.geocoders import Nominatim
    from pyproj import Transformer

    crs = G.graph.get("crs", "EPSG:32723")
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    geolocator = Nominatim(user_agent="vigiaEnchente")
    locations = []

    for node in node_ids:
        lon, lat = transformer.transform(G.nodes[node]["x"], G.nodes[node]["y"])
        try:
            result = geolocator.reverse((lat, lon), language="pt", timeout=10)
            address = result.address if result else f"({lat:.5f}, {lon:.5f})"
        except Exception:
            address = f"({lat:.5f}, {lon:.5f})"
        locations.append({"node": node, "address": address, "lat": lat, "lon": lon})
        time.sleep(1.1)

    return locations