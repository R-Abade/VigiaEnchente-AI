"""
metrics.py
──────────
Métricas de rede para o grafo viário de Sabará.

Responsabilidades
  • Estatísticas gerais (grau, densidade, clustering, componentes)
  • Floyd-Warshall: matriz de distâncias, diâmetro, nós mais isolados
  • Centralidade de intermediação (betweenness)
  • Clustering hierárquico (Ward) para análise de sub-regiões (Q6)
  • Geocoding reverso para localizar nós isolados

Nenhuma variável global — todos os parâmetros são explícitos.
"""

import time
import warnings

import networkx as nx
import numpy as np

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Estatísticas gerais
# ─────────────────────────────────────────────────────────────────────────────

def compute_network_metrics(G):
    """
    Imprime e retorna estatísticas básicas da rede.

    Parâmetros
    ──────────
    G : grafo NetworkX

    Retorna
    ───────
    stats : dict com num_nodes, num_edges, avg_degree, density,
                       avg_clustering, num_components
    """
    G_simple = nx.Graph(G)
    num_nodes = len(G_simple.nodes())
    degrees   = dict(G_simple.degree()).values()

    stats = {
        "num_nodes":       num_nodes,
        "num_edges":       len(G_simple.edges()),
        "avg_degree":      sum(degrees) / num_nodes,
        "density":         nx.density(G_simple),
        "avg_clustering":  nx.average_clustering(G_simple),
        "num_components":  nx.number_connected_components(G_simple),
    }

    print("\n=== Network Metrics ===")
    print(f"  Nodes                : {stats['num_nodes']}")
    print(f"  Edges                : {stats['num_edges']}")
    print(f"  Average degree       : {stats['avg_degree']:.4f}")
    print(f"  Density              : {stats['density']:.6f}")
    print(f"  Average clustering   : {stats['avg_clustering']:.4f}")
    print(f"  Connected components : {stats['num_components']}")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Floyd-Warshall
# ─────────────────────────────────────────────────────────────────────────────

def compute_fw_matrix(G):
    """
    Calcula a matriz de distâncias Floyd-Warshall.

    Parâmetros
    ──────────
    G : grafo NetworkX com atributo 'weight'

    Retorna
    ───────
    matrix : ndarray (N×N) com distâncias em metros (inf para pares desconectados)
    nodes  : lista de nós na mesma ordem das linhas/colunas da matriz
    """
    nodes = list(G.nodes())
    print(f"\nComputing Floyd-Warshall for {len(nodes)} nodes...")
    t0 = time.time()
    matrix = nx.floyd_warshall_numpy(G, weight="weight")
    np.fill_diagonal(matrix, 0)
    print(f"  Computation time : {time.time() - t0:.2f}s")
    return matrix, nodes


def compute_diameter(G, matrix=None, nodes=None):
    """
    Encontra o par de nós mais distantes (diâmetro do grafo).

    Parâmetros
    ──────────
    G      : grafo NetworkX
    matrix : matriz FW pré-calculada (opcional — calculada internamente se None)
    nodes  : lista de nós correspondente à matrix (obrigatório se matrix != None)

    Retorna
    ───────
    source    : nó de origem do diâmetro
    target    : nó de destino do diâmetro
    max_dist  : distância máxima em metros
    matrix    : matriz FW (para reuso)
    nodes     : lista de nós (para reuso)
    """
    if matrix is None:
        matrix, nodes = compute_fw_matrix(G)

    finite_mask = matrix != np.inf
    max_dist    = float(np.nanmax(matrix[finite_mask]))

    u_idx, v_idx = np.unravel_index(
        np.where(finite_mask, matrix, -1).argmax(), matrix.shape
    )
    source, target = nodes[u_idx], nodes[v_idx]

    print(f"  Graph diameter  : {max_dist:.2f} m")
    print(f"  Source node     : {source}")
    print(f"  Target node     : {target}")

    return source, target, max_dist, matrix, nodes


# ─────────────────────────────────────────────────────────────────────────────
# Nós mais isolados (Q5)
# ─────────────────────────────────────────────────────────────────────────────

def find_isolated_nodes(matrix, nodes, top_n: int = 8, threshold_pct: float = 0.80):
    """
    Identifica os nós com maior distância média a todos os demais (Q5).

    Parâmetros
    ──────────
    matrix        : ndarray FW
    nodes         : lista de nós (mesma ordem da matrix)
    top_n         : quantos nós retornar
    threshold_pct : percentil para classificar como "isolado" (só para logging)

    Retorna
    ───────
    isolated : lista de (node_id, avg_dist_m) ordenada de mais para menos isolado
    avg_dist : vetor completo de distâncias médias por nó
    """
    avg_dist      = matrix.mean(axis=1)
    threshold_val = np.percentile(avg_dist, threshold_pct * 100)
    top_idx       = np.argsort(avg_dist)[::-1][:top_n]
    isolated      = [(nodes[i], float(avg_dist[i])) for i in top_idx]

    print(
        f"\nQ5 — Top {top_n} isolated nodes "
        f"(avg dist > {threshold_val / 1000:.2f} km = p{int(threshold_pct * 100)})"
    )
    for rank, (node, d) in enumerate(isolated, 1):
        print(f"  #{rank:>2}  node={node}  avg_dist={d / 1000:.3f} km")

    return isolated, avg_dist


# ─────────────────────────────────────────────────────────────────────────────
# Clustering hierárquico (Q6)
# ─────────────────────────────────────────────────────────────────────────────

def compute_hierarchical_clusters(matrix, nodes, n_clusters: int = None):
    """
    Agrupa os nós por clustering hierárquico de Ward sobre a matriz FW.

    Parâmetros
    ──────────
    matrix     : ndarray FW (valores finitos substituídos por finite_max × 1.2)
    nodes      : lista de nós
    n_clusters : número de clusters; se None, usa heurística sqrt(N)/2

    Retorna
    ───────
    labels      : array de rótulos de cluster (tamanho N, indexado por nodes)
    order       : ordem de folhas do dendrograma (para reordenação do heatmap)
    n_clusters  : número de clusters usado
    """
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial.distance import squareform

    N = len(nodes)
    if n_clusters is None:
        n_clusters = max(2, int(np.sqrt(N) / 2))

    print(f"\nQ6 — Hierarchical clustering (Ward, {n_clusters} clusters)...")

    condensed = squareform(matrix, checks=False)
    Z         = linkage(condensed, method="ward")
    dend      = dendrogram(Z, no_plot=True)
    order     = dend["leaves"]
    labels    = fcluster(Z, n_clusters, criterion="maxclust")

    return labels, order, n_clusters


def report_clusters(matrix, nodes, labels, n_clusters: int, G=None):
    """
    Imprime o relatório de clusters (Q6): intra/inter distâncias e razão de isolamento.

    Parâmetros
    ──────────
    matrix     : ndarray FW
    nodes      : lista de nós
    labels     : array de rótulos (saída de compute_hierarchical_clusters)
    n_clusters : número de clusters
    G          : grafo NetworkX (opcional, para converter coordenadas do centroide)
    """
    from metrics import _utm_to_latlon  # import interno para evitar circular

    node_index    = {n: i for i, n in enumerate(nodes)}
    unique_labels = np.unique(labels)

    print(f"\n{'─' * 78}")
    print(f"  Q6 — ANÁLISE DE CLUSTERS ({n_clusters} clusters — Ward linkage)")
    print(f"  Razão de isolamento = dist. intra / dist. inter")
    print(f"{'─' * 78}")

    for c in unique_labels:
        members = [nodes[i] for i, lbl in enumerate(labels) if lbl == c]
        m_idx   = [node_index[m] for m in members]
        e_idx   = [node_index[m] for m in nodes if m not in set(members)]

        sub = matrix[np.ix_(m_idx, m_idx)]
        ext = matrix[np.ix_(m_idx, e_idx)]

        intra_avg = float(sub[sub > 0].mean()) / 1000 if sub[sub > 0].size else 0.0
        inter_avg = float(ext.mean()) / 1000            if ext.size         else 0.0
        ratio     = intra_avg / inter_avg if inter_avg > 0 else float("nan")

        intra_means   = sub.mean(axis=1)
        centroid_node = members[int(np.argmin(intra_means))]

        flag = "← ISOLAMENTO ALTO" if ratio < 0.6 else ""
        print(f"\n  Cluster {c:>2}  |  {len(members):>4} nós")
        print(f"    Dist. intra média  : {intra_avg:.3f} km")
        print(f"    Dist. inter média  : {inter_avg:.3f} km")
        print(f"    Razão de isolamento: {ratio:.3f}  {flag}")

        if G is not None:
            lat_c, lon_c = _utm_to_latlon(G, centroid_node)
            print(f"    Nó centroide       : {centroid_node}  ({lat_c:.5f}, {lon_c:.5f})")


# ─────────────────────────────────────────────────────────────────────────────
# Centralidade de intermediação
# ─────────────────────────────────────────────────────────────────────────────

def compute_betweenness(G, top_n: int = 10):
    """
    Calcula a centralidade de intermediação e retorna os top_n nós.

    Parâmetros
    ──────────
    G     : grafo NetworkX
    top_n : quantos nós reportar (padrão 10)

    Retorna
    ───────
    bet : dict {node: score} para todos os nós
    top : lista dos top_n (node, score) ordenada decrescentemente
    """
    print(f"\nComputing betweenness centrality ({len(G)} nodes)...")
    G_simple = nx.Graph(G)
    bet = nx.betweenness_centrality(G_simple, k=len(G_simple), weight="weight", seed=42)
    top = sorted(bet.items(), key=lambda x: x[1], reverse=True)[:top_n]

    print(f"  Top {top_n} nodes by betweenness:")
    for node, val in top:
        print(f"    Node {node:>8}  score={val:.6f}")

    return bet, top


# ─────────────────────────────────────────────────────────────────────────────
# Geocoding reverso
# ─────────────────────────────────────────────────────────────────────────────

def _utm_to_latlon(G, node):
    """Converte coordenadas UTM do nó para (lat, lon) WGS-84."""
    from pyproj import Transformer
    crs = G.graph.get("crs", "EPSG:32723")
    tf  = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    x, y = G.nodes[node]["x"], G.nodes[node]["y"]
    lon, lat = tf.transform(x, y)
    return lat, lon


def geocode_nodes(G, node_ids: list):
    """
    Faz geocoding reverso via Nominatim para uma lista de nós.
    Respeita o limite de 1 req/s da API.

    Parâmetros
    ──────────
    G        : grafo NetworkX projetado
    node_ids : lista de IDs de nós

    Retorna
    ───────
    locations : lista de dicts {node, address, lat, lon}
    """
    import time
    from geopy.geocoders import Nominatim
    from pyproj import Transformer

    crs         = G.graph.get("crs", "EPSG:32723")
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    geolocator  = Nominatim(user_agent="vigiaEnchente_graph_analysis")
    locations   = []

    for node in node_ids:
        x = G.nodes[node]["x"]
        y = G.nodes[node]["y"]
        lon, lat = transformer.transform(x, y)

        try:
            result  = geolocator.reverse((lat, lon), language="pt", timeout=10)
            address = result.address if result else f"({lat:.5f}, {lon:.5f})"
        except Exception:
            address = f"({lat:.5f}, {lon:.5f})"

        locations.append({"node": node, "address": address, "lat": lat, "lon": lon})
        time.sleep(1.1)

    return locations
