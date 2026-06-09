"""
visualization.py
────────────────
Funções de visualização para o grafo viário de Sabará.

Todas as imagens são salvas em assets/images/ (relativo à raiz do projeto).
O diretório é criado automaticamente se não existir.

Responsabilidades
  • plot_route_comparison      — Q4: rota original vs penalizada
  • plot_fw_heatmap            — heatmap principal Floyd-Warshall (replica notebook)
  • plot_fw_heatmap_clustered  — Q5+Q6: heatmap reordenado + clusters + isolados
  • plot_isolated_nodes        — mapa dos N nós mais isolados sobre a rede
  • plot_nearest_river_nodes   — gradiente de proximidade ao rio
  • plot_blocked_comparison    — grafo original vs grafo bloqueado (lado a lado)
  • plot_degree_distribution   — histograma de graus
  • plot_graph_on_map          — GCN: grafo sobreposto ao mapa de Sabará com scores de risco
  • plot_top_risk_nodes        — GCN: Top-N nós de risco destacados sobre a rede

Nenhuma variável global — todos os parâmetros são explícitos.
"""

import warnings
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox

warnings.filterwarnings("ignore")

# Diretório de saída padrão (relativo à raiz do projeto)
IMAGES_DIR = Path(__file__).resolve().parents[2] / "assets" / "images"


def _ensure_output_dir(path: Path) -> Path:
    """Cria o diretório pai se não existir e retorna o path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _img(filename: str) -> Path:
    """Resolve e garante o diretório de saída para um nome de arquivo."""
    return _ensure_output_dir(IMAGES_DIR / filename)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos de desenho
# ─────────────────────────────────────────────────────────────────────────────

def _draw_base_graph(ax, G, rivers, graph_crs, risk_info=None, show_risk: bool = True):
    """Renderiza arestas do grafo, o rio e os nós de risco num eixo existente."""
    ox.plot_graph(
        G, ax=ax,
        node_size=0, edge_color="#2a2a2a", edge_linewidth=0.3,
        bgcolor="#0F0F0F", show=False, close=False,
    )
    if rivers is not None and not rivers.empty:
        rivers.to_crs(graph_crs).plot(
            ax=ax, color="#4FC3F7", linewidth=2.5, alpha=0.85, zorder=4
        )
    if show_risk and risk_info:
        rx = [r["x"] for r in risk_info]
        ry = [r["y"] for r in risk_info]
        ax.scatter(rx, ry, c="#00E5FF", s=28, zorder=5,
                   edgecolors="#003F5C", linewidths=0.4, alpha=0.85)


def _draw_route(ax, G, route, color, lw: float = 2.4, ls: str = "-", zorder: int = 6):
    """Traça uma rota segmento a segmento sobre um eixo."""
    if not route:
        return
    for u, v in zip(route[:-1], route[1:]):
        if u in G.nodes() and v in G.nodes():
            x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
            x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw,
                    alpha=0.9, zorder=zorder, linestyle=ls)


# ─────────────────────────────────────────────────────────────────────────────
# Q4 — Comparação de rotas (original vs penalizada)
# ─────────────────────────────────────────────────────────────────────────────

def plot_route_comparison(
    G,
    cmp: dict,
    risk_nodes: set,
    risk_info: list,
    rivers,
    threshold: float = 50.0,
    filename: str = "q4_route_comparison.png",
):
    """
    Q4 — Plota a rota original e a rota penalizada sobre o grafo.

    Parâmetros
    ──────────
    G          : grafo NetworkX projetado
    cmp        : dict retornado por routing.compare_routes()
    risk_nodes : conjunto de nós de risco
    risk_info  : lista de dicts com coordenadas dos nós de risco
    rivers     : GeoDataFrame do rio (pode ser None)
    threshold  : distância de risco usada (para legenda)
    filename   : nome do arquivo PNG de saída (salvo em assets/images/)
    """
    print("\nPlotting route comparison (Q4)...")
    graph_crs = G.graph.get("crs", "EPSG:32723")

    fig, ax = ox.plot_graph(
        G, node_size=0, edge_color="#333333", edge_linewidth=0.3,
        bgcolor="#0F0F0F", show=False, close=False, figsize=(14, 11),
    )

    if rivers is not None and not rivers.empty:
        rivers.to_crs(graph_crs).plot(
            ax=ax, color="#4FC3F7", linewidth=2.5, alpha=0.9, zorder=4
        )

    # Nós de risco
    for r in risk_info:
        ax.scatter(r["x"], r["y"], c="#00E5FF", s=45, zorder=5,
                   edgecolors="#004D6E", linewidths=0.6)

    # Rota original
    _draw_route(ax, G, cmp["route_orig"], "#E53935", lw=2.5, zorder=6)

    # Rota penalizada
    pen_color = "#FFD600" if not cmp["same_route"] else "#FFB300"
    _draw_route(ax, G, cmp["route_pen"], pen_color, lw=1.8, ls="--", zorder=7)

    # Origem / destino
    src, tgt = cmp["route_orig"][0], cmp["route_orig"][-1]
    ax.scatter(G.nodes[src]["x"], G.nodes[src]["y"],
               c="#00C853", s=150, zorder=9, edgecolors="white", linewidths=1.2)
    ax.scatter(G.nodes[tgt]["x"], G.nodes[tgt]["y"],
               c="#FFB300", s=150, zorder=9, edgecolors="white", linewidths=1.2)

    pen_label = ("Rota penalizada (diferente)" if not cmp["same_route"]
                 else "Rota penalizada (idêntica)")
    legend_handles = [
        mpatches.Patch(color="#E53935",
                       label=f"Rota original — {cmp['dist_orig']:.0f} m | "
                             f"{len(cmp['risk_orig'])} nós de risco"),
        mpatches.Patch(color=pen_color,
                       label=f"{pen_label} — {cmp['dist_pen']:.0f} m | "
                             f"{len(cmp['risk_pen'])} nós de risco"),
        mpatches.Patch(color="#4FC3F7", label="Rio das Velhas"),
        mpatches.Patch(color="#00E5FF", label=f"Nós de risco (≤ {threshold} m)"),
        mpatches.Patch(color="#00C853", label="Origem"),
        mpatches.Patch(color="#FFB300", label="Destino"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8.5,
              framealpha=0.88, facecolor="#1A1A1A", labelcolor="white")

    delta  = cmp["dist_pen"] - cmp["dist_orig"]
    status = "ROTA ALTERADA" if not cmp["same_route"] else "ROTA INALTERADA — sem alternativa"
    ax.set_title(
        f"Q4 — Comparação de Rotas: Original vs Penalizada (×2 em nós de risco)\n"
        f"{status}  |  Δ distância real = {delta:+.1f} m",
        fontsize=11, color="white", pad=10,
    )

    out = _img(filename)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap Floyd-Warshall — fiel ao notebook (sem reordenação)
# ─────────────────────────────────────────────────────────────────────────────

def plot_fw_heatmap(
    matrix: np.ndarray,
    place_name: str,
    filename: str = "fw_heatmap.png",
):
    """
    Replica o heatmap do notebook de referência (caminhos.ipynb):
    cmap='magma', colorbar em metros, sem reordenação de nós.

    Parâmetros
    ──────────
    matrix     : ndarray N×N retornado por metrics.compute_fw_matrix()
    place_name : nome do lugar (usado no título)
    filename   : nome do arquivo de saída
    """
    finite_max = float(np.nanmax(matrix[matrix != np.inf]))
    display    = np.where(matrix == np.inf, finite_max * 1.2, matrix)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(display, cmap="magma", interpolation="nearest", aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Shortest Path Distance (meters)")
    ax.set_title(f"Floyd-Warshall Distance Matrix: {place_name}")
    ax.set_xlabel("Target Node Index")
    ax.set_ylabel("Source Node Index")

    out = _img(filename)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Q5 + Q6 — Heatmap reordenado por Ward + isolados + fronteiras de cluster
# ─────────────────────────────────────────────────────────────────────────────

def plot_fw_heatmap_clustered(
    matrix: np.ndarray,
    nodes: list,
    isolated_nodes: list,
    labels: np.ndarray,
    order: list,
    n_clusters: int,
    cluster_threshold_pct: float = 0.80,
    filename: str = "fw_heatmap_q5q6.png",
):
    """
    Q5+Q6 — Heatmap reordenado pelo dendrograma de Ward.
    Marca nós isolados (Q5) com linhas ciano e fronteiras de cluster (Q6) em branco.

    Parâmetros
    ──────────
    matrix                : ndarray FW
    nodes                 : lista de nós (mesma ordem da matrix)
    isolated_nodes        : lista de (node, avg_dist) retornada por metrics.find_isolated_nodes()
    labels                : array de rótulos de cluster
    order                 : ordem de folhas do dendrograma
    n_clusters            : número de clusters
    cluster_threshold_pct : percentil usado em Q5 (para a legenda)
    filename              : nome do arquivo de saída
    """
    nodes_ord  = [nodes[i] for i in order]
    labels_ord = labels[order]
    unique_clusters = np.unique(labels)

    finite_max = float(np.nanmax(matrix[matrix != np.inf]))
    matrix_vis = np.where(matrix == np.inf, finite_max * 1.2, matrix)
    matrix_ord = matrix_vis[np.ix_(order, order)]

    cluster_colors    = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    color_map_cluster = {c: cluster_colors[i] for i, c in enumerate(unique_clusters)}

    fig = plt.figure(figsize=(14, 12), facecolor="#0F0F0F")
    fig.suptitle(
        "Floyd-Warshall Distance Matrix — Sabará\n"
        "Nós reordenados por Ward clustering  |  Q5: nós isolados  |  Q6: fronteiras de cluster",
        color="white", fontsize=11, y=0.99,
    )

    gs      = fig.add_gridspec(2, 2, width_ratios=[30, 1],
                               height_ratios=[1, 30], hspace=0.02, wspace=0.02)
    ax_top  = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_cbar = fig.add_subplot(gs[1, 1])

    norm2 = mcolors.PowerNorm(gamma=0.5, vmin=matrix_ord.min(), vmax=matrix_ord.max())
    im2   = ax_heat.imshow(matrix_ord / 1000, cmap="magma",
                           norm=norm2, aspect="auto", interpolation="nearest")
    ax_heat.set_xlabel("Índice do nó (reordenado)", color="white", fontsize=9)
    ax_heat.set_ylabel("Índice do nó (reordenado)", color="white", fontsize=9)
    ax_heat.tick_params(colors="white", labelsize=7)

    cb2 = plt.colorbar(im2, cax=ax_cbar)
    cb2.set_label("Menor caminho (km)", color="white", fontsize=9)
    cb2.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=7)

    # Barra de clusters no topo
    cluster_bar  = np.array([labels_ord]).astype(float)
    cluster_cmap = mcolors.ListedColormap([color_map_cluster[c] for c in unique_clusters])
    ax_top.imshow(cluster_bar, aspect="auto", cmap=cluster_cmap,
                  vmin=unique_clusters.min(), vmax=unique_clusters.max())
    ax_top.set_yticks([]); ax_top.set_xticks([])
    ax_top.set_title("Cluster membership", color="white", fontsize=8, pad=3)
    for spine in ax_top.spines.values():
        spine.set_edgecolor("#444444")

    # Linhas dos nós isolados (Q5)
    isolated_set  = {n for n, _ in isolated_nodes}
    iso_positions = [nodes_ord.index(n) for n in isolated_set if n in nodes_ord]
    for pos in iso_positions:
        ax_heat.axvline(pos, color="#00E5FF", linewidth=0.7, alpha=0.7)
        ax_heat.axhline(pos, color="#00E5FF", linewidth=0.7, alpha=0.7)

    # Fronteiras de cluster (Q6)
    prev = 0
    for c in unique_clusters[:-1]:
        count    = int((labels_ord == c).sum())
        boundary = prev + count
        ax_heat.axvline(boundary - 0.5, color="white", linewidth=0.9, alpha=0.45)
        ax_heat.axhline(boundary - 0.5, color="white", linewidth=0.9, alpha=0.45)
        prev = boundary

    top_n = len(isolated_nodes)
    legend_handles = [
        mpatches.Patch(color="#00E5FF",
                       label=f"Nós isolados — top {top_n} "
                             f"(Q5, p{int(cluster_threshold_pct * 100)})"),
    ] + [
        mpatches.Patch(color=cluster_colors[i], label=f"Cluster {c}")
        for i, c in enumerate(unique_clusters)
    ]
    ax_heat.legend(handles=legend_handles, loc="lower right", fontsize=7,
                   framealpha=0.5, facecolor="#1A1A1A", labelcolor="white")

    out = _img(filename)
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Mapa dos nós mais isolados
# ─────────────────────────────────────────────────────────────────────────────

def plot_isolated_nodes(
    G,
    matrix: np.ndarray,
    nodes: list,
    isolated_nodes: list,
    rivers=None,
    filename: str = "nos_isolados.png",
):
    """
    Mapa com todos os nós coloridos por distância média (background magma)
    e os mais isolados destacados com rótulo de ranking.

    Parâmetros
    ──────────
    G              : grafo NetworkX projetado
    matrix         : ndarray FW
    nodes          : lista de nós
    isolated_nodes : lista de (node, avg_dist) — saída de metrics.find_isolated_nodes()
    rivers         : GeoDataFrame do rio (pode ser None)
    filename       : nome do arquivo de saída
    """
    graph_crs = G.graph.get("crs", "EPSG:32723")
    avg_dist  = matrix.mean(axis=1)

    dist_vals = np.array([d for _, d in isolated_nodes])
    cmap_iso  = plt.cm.YlOrRd_r
    norm_iso  = plt.Normalize(vmin=dist_vals.min(), vmax=dist_vals.max())
    all_norm  = plt.Normalize(vmin=avg_dist.min(), vmax=avg_dist.max())
    cmap_bg   = plt.cm.magma

    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor("#0F0F0F")
    ox.plot_graph(G, ax=ax, node_size=0, edge_color="#1e1e1e",
                  edge_linewidth=0.28, bgcolor="#0F0F0F", show=False, close=False)

    # Background: todos os nós por distância média
    for i, node in enumerate(nodes):
        ax.scatter(G.nodes[node]["x"], G.nodes[node]["y"],
                   c=[cmap_bg(all_norm(avg_dist[i]))], s=5, zorder=2, alpha=0.4)

    if rivers is not None and not rivers.empty:
        rivers.to_crs(graph_crs).plot(ax=ax, color="#4FC3F7",
                                      linewidth=3, alpha=0.8, zorder=4)

    # Top-N isolados com rótulos
    for rank, (node, d) in enumerate(isolated_nodes):
        x, y  = G.nodes[node]["x"], G.nodes[node]["y"]
        color = cmap_iso(norm_iso(d))
        size  = 260 - rank * 8
        ax.scatter(x, y, c=[color], s=size, zorder=8,
                   edgecolors="white", linewidths=1.0, alpha=0.97)
        ax.annotate(
            f"#{rank + 1}  {d / 1000:.1f} km",
            xy=(x, y), xytext=(10, 5), textcoords="offset points",
            color="white", fontsize=7, zorder=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="#0F0F0F", ec="#444444", alpha=0.82),
        )

    sm_iso = plt.cm.ScalarMappable(cmap=cmap_iso, norm=norm_iso)
    sm_iso.set_array([])
    cb1 = plt.colorbar(sm_iso, ax=ax, fraction=0.022, pad=0.02, shrink=0.7)
    cb1.set_label(f"Dist. média — top {len(isolated_nodes)} isolados (m)",
                  color="white", fontsize=9)
    cb1.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=8)

    sm_bg = plt.cm.ScalarMappable(cmap=cmap_bg, norm=all_norm)
    sm_bg.set_array([])
    cb2 = plt.colorbar(sm_bg, ax=ax, fraction=0.022, pad=0.08, shrink=0.7)
    cb2.set_label("Dist. média — todos os nós (m)", color="white", fontsize=9)
    cb2.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=8)

    ax.set_title(
        f"Top {len(isolated_nodes)} nós mais isolados — rede viária de Sabará\n"
        "(Floyd-Warshall — dist. média; tamanho proporcional ao grau de isolamento)",
        fontsize=11, color="white", pad=10,
    )
    ax.set_axis_off()
    plt.tight_layout()

    out = _img(filename)
    plt.savefig(out, dpi=250, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Mapa de nós mais próximos do rio (gradiente de proximidade)
# ─────────────────────────────────────────────────────────────────────────────

def plot_nearest_river_nodes(
    G,
    nearby: list,
    rivers,
    threshold: float = 50.0,
    top_n: int = 20,
    filename: str = "nearest_river_nodes.png",
):
    """
    Plota todos os nós dentro da faixa de risco com gradiente de cor por distância.
    Os top_n mais próximos recebem borda branca e rótulo numérico.

    Parâmetros
    ──────────
    G         : grafo NetworkX projetado
    nearby    : lista de dicts retornada por flood.find_nearest_river_nodes()
    rivers    : GeoDataFrame do rio
    threshold : distância máxima usada (para legenda e colorbar)
    top_n     : quantos nós destacar com rótulo
    filename  : nome do arquivo de saída
    """
    import matplotlib.cm as cm

    graph_crs = G.graph.get("crs", "EPSG:32723")
    cmap      = cm.RdYlGn_r
    norm      = mcolors.Normalize(vmin=0, vmax=threshold)
    sm        = cm.ScalarMappable(cmap=cmap, norm=norm)

    fig, ax = ox.plot_graph(
        G, node_size=0, edge_color="#2a2a2a", edge_linewidth=0.25,
        bgcolor="#0d0d0d", show=False, close=False, figsize=(13, 10),
    )

    if rivers is not None and not rivers.empty:
        rivers.to_crs(graph_crs).plot(ax=ax, color="#4FC3F7",
                                      linewidth=2.8, alpha=0.9, zorder=4)

    for r in nearby:
        color = cmap(norm(r["dist_m"]))
        ax.scatter(r["x"], r["y"], c=[color], s=35, zorder=5,
                   edgecolors="none", alpha=0.75)

    for rank, r in enumerate(nearby[:top_n], 1):
        color = cmap(norm(r["dist_m"]))
        ax.scatter(r["x"], r["y"], c=[color], s=90, zorder=7,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(str(rank), xy=(r["x"], r["y"]), xytext=(4, 4),
                    textcoords="offset points", fontsize=6,
                    color="white", zorder=8)

    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Distância ao rio (m)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    ax.legend(
        handles=[
            mpatches.Patch(color="#4FC3F7", label="Rio das Velhas"),
            mpatches.Patch(color=cmap(0.0), label="Muito próximo (0 m)"),
            mpatches.Patch(color=cmap(0.5), label=f"~{threshold / 2:.0f} m"),
            mpatches.Patch(color=cmap(1.0), label=f"~{threshold:.0f} m"),
        ],
        loc="lower left", fontsize=8, framealpha=0.85,
        facecolor="#1a1a1a", labelcolor="white",
    )
    ax.set_title(
        f"Nós mais próximos do Rio das Velhas (≤ {threshold} m)\n"
        f"{len(nearby)} nós encontrados  |  Top {top_n} numerados  |  "
        "Gradiente: distância ao rio",
        color="white", fontsize=10, pad=8,
    )

    out = _img(filename)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Grafo original vs bloqueado (lado a lado)
# ─────────────────────────────────────────────────────────────────────────────

def plot_blocked_comparison(
    G, G_blocked,
    risk_nodes: set,
    risk_info: list,
    rivers,
    route_orig: list,
    route_alt: list,
    src_orig, tgt_orig,
    src_alt, tgt_alt,
    disconnected: bool,
    results_orig: dict,
    results_blocked: dict,
    threshold: float = 50.0,
    filename: str = "grafo_bloqueado.png",
):
    """
    Dois painéis lado a lado:
      Esquerdo: grafo completo + rota original
      Direito : grafo bloqueado + rota alternativa (ou desconexão)

    Parâmetros
    ──────────
    G, G_blocked      : grafos original e bloqueado
    risk_nodes        : conjunto de nós de risco
    risk_info         : coordenadas dos nós de risco
    rivers            : GeoDataFrame do rio
    route_orig        : rota original (lista de nós)
    route_alt         : rota alternativa ou nova rota no grafo bloqueado
    src_orig, tgt_orig: par original
    src_alt, tgt_alt  : par usado no grafo bloqueado
    disconnected      : True se o par original ficou desconectado
    results_orig      : dict retornado por routing.run_algorithms() no grafo original
    results_blocked   : dict retornado por routing.run_algorithms() no grafo bloqueado
    threshold         : distância de risco (para legenda)
    filename          : nome do arquivo de saída
    """
    graph_crs = G.graph.get("crs", "EPSG:32723")
    fig, axes = plt.subplots(1, 2, figsize=(20, 11))
    fig.patch.set_facecolor("#0F0F0F")

    d_o = results_orig["Dijkstra"]["dist"] or 0
    n_o = results_orig["Dijkstra"]["nodes"] or 0
    d_b = results_blocked["Dijkstra"]["dist"]
    n_b = results_blocked["Dijkstra"]["nodes"]
    d_str = f"{d_b / 1000:.2f} km" if d_b else "N/A"
    n_str = str(n_b) if n_b else "N/A"

    # Painel esquerdo
    ax_l = axes[0]
    _draw_base_graph(ax_l, G, rivers, graph_crs, risk_info, show_risk=True)
    _draw_route(ax_l, G, route_orig, "#E53935", lw=2.5)
    if route_orig:
        ax_l.scatter(G.nodes[src_orig]["x"], G.nodes[src_orig]["y"],
                     c="#00C853", s=200, zorder=9, edgecolors="white", linewidths=1.2)
        ax_l.scatter(G.nodes[tgt_orig]["x"], G.nodes[tgt_orig]["y"],
                     c="#FFB300", s=200, zorder=9, edgecolors="white", linewidths=1.2)

    risk_on_orig = sum(1 for n in (route_orig or []) if n in risk_nodes)
    ax_l.set_title(
        f"Grafo original — rota {src_orig}→{tgt_orig}\n"
        f"Distância: {d_o / 1000:.2f} km  |  Nós: {n_o}  |  "
        f"Nós de risco na rota: {risk_on_orig}",
        fontsize=10, color="white", pad=8,
    )
    ax_l.legend(
        handles=[
            mpatches.Patch(color="#E53935",
                           label=f"Rota original ({d_o / 1000:.2f} km, {n_o} nós)"),
            mpatches.Patch(color="#4FC3F7", label="Rio das Velhas"),
            mpatches.Patch(color="#00E5FF",
                           label=f"Nós de risco (≤ {threshold:.0f} m — n={len(risk_nodes)})"),
            mpatches.Patch(color="#00C853", label="Origem"),
            mpatches.Patch(color="#FFB300", label="Destino"),
        ],
        loc="lower left", fontsize=8.5,
        framealpha=0.88, facecolor="#1A1A1A", labelcolor="white",
    )
    ax_l.set_axis_off()

    # Painel direito
    ax_r = axes[1]
    _draw_base_graph(ax_r, G_blocked, rivers, graph_crs, risk_info=[], show_risk=False)

    if disconnected:
        for nid in [src_orig, tgt_orig]:
            ref = G if nid not in G_blocked.nodes() else G_blocked
            ax_r.scatter(ref.nodes[nid]["x"], ref.nodes[nid]["y"],
                         c="#E53935", s=300, marker="x", zorder=9,
                         linewidths=2.5, alpha=0.7)
        _draw_route(ax_r, G_blocked, route_alt, "#FFD600", lw=2.5, ls="--")
        if route_alt:
            if src_alt in G_blocked.nodes():
                ax_r.scatter(G_blocked.nodes[src_alt]["x"], G_blocked.nodes[src_alt]["y"],
                             c="#00C853", s=200, zorder=9, edgecolors="white", linewidths=1.2)
            if tgt_alt in G_blocked.nodes():
                ax_r.scatter(G_blocked.nodes[tgt_alt]["x"], G_blocked.nodes[tgt_alt]["y"],
                             c="#FFB300", s=200, zorder=9, edgecolors="white", linewidths=1.2)
        title = (
            f"Grafo bloqueado — PAR ORIGINAL DESCONECTADO\n"
            f"Novo diâmetro: {src_alt}→{tgt_alt}  |  Distância: {d_str}  |  Nós: {n_str}"
        )
        legend_handles = [
            mpatches.Patch(color="#FFD600",
                           label=f"Nova rota — novo par ({d_str}, {n_str} nós)"),
            mpatches.Patch(color="#E53935",
                           label=f"Par original desconectado ({src_orig}→{tgt_orig})"),
            mpatches.Patch(color="#4FC3F7", label="Rio das Velhas"),
            mpatches.Patch(color="#00C853", label=f"Nova origem ({src_alt})"),
            mpatches.Patch(color="#FFB300", label=f"Novo destino ({tgt_alt})"),
        ]
    else:
        _draw_route(ax_r, G_blocked, route_alt, "#FFD600", lw=2.5, ls="--")
        if route_alt:
            ax_r.scatter(G_blocked.nodes[src_alt]["x"], G_blocked.nodes[src_alt]["y"],
                         c="#00C853", s=200, zorder=9, edgecolors="white", linewidths=1.2)
            ax_r.scatter(G_blocked.nodes[tgt_alt]["x"], G_blocked.nodes[tgt_alt]["y"],
                         c="#FFB300", s=200, zorder=9, edgecolors="white", linewidths=1.2)
        d_delta = (d_b - d_o) if (d_b and d_o) else 0
        title = (
            f"Grafo bloqueado — rota alternativa {src_orig}→{tgt_orig}\n"
            f"Distância: {d_str} (Δ {d_delta / 1000:+.2f} km)  |  Nós: {n_str}"
        )
        legend_handles = [
            mpatches.Patch(color="#FFD600",
                           label=f"Rota alternativa ({d_str}, {n_str} nós)"),
            mpatches.Patch(color="#4FC3F7", label="Rio das Velhas"),
            mpatches.Patch(color="#00C853", label="Origem"),
            mpatches.Patch(color="#FFB300", label="Destino"),
        ]

    ax_r.set_title(title, fontsize=10, color="white", pad=8)
    ax_r.legend(handles=legend_handles, loc="lower left", fontsize=8.5,
                framealpha=0.88, facecolor="#1A1A1A", labelcolor="white")
    ax_r.set_axis_off()

    out = _img(filename)
    plt.tight_layout(pad=1.2)
    plt.savefig(out, dpi=250, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Distribuição de grau
# ─────────────────────────────────────────────────────────────────────────────

def plot_degree_distribution(G, filename: str = "degree_distribution.png"):
    """
    Histograma da distribuição de grau dos nós.

    Parâmetros
    ──────────
    G        : grafo NetworkX
    filename : nome do arquivo de saída
    """
    degrees = [d for _, d in G.degree()]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(degrees, bins=20, color="#2196F3", edgecolor="white")
    ax.set_title("Distribuição do Grau dos Nós — Sabará")
    ax.set_xlabel("Grau")
    ax.set_ylabel("Frequência")

    out = _img(filename)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# GCN — Grafo sobreposto ao mapa com scores de risco por nó
# ─────────────────────────────────────────────────────────────────────────────

def plot_graph_on_map(
    G,
    nodes: list,
    risk_scores: np.ndarray,
    risk_nodes: set,
    rivers,
    filename: str = "gcn_risk_map.png",
    threshold: float = 50.0,
):
    """
    Plota o grafo viário de Sabará com cada nó colorido pelo score de risco
    previsto pelo GCN. Nós críticos (APP <= threshold m) recebem borda branca.
    O Rio das Velhas é destacado em azul.

    Uso típico em gcn.py:
        from visualization import plot_graph_on_map
        plot_graph_on_map(G, nodes, pred.numpy(), risk_nodes, rivers)

    Parâmetros
    ──────────
    G           : grafo NetworkX projetado (com atributos x, y por nó)
    nodes       : lista de IDs de nós na mesma ordem de risk_scores
    risk_scores : array (N,) com scores previstos pelo modelo GCN, em [0, 1+]
    risk_nodes  : conjunto de IDs dos nós críticos (APP <= threshold m)
    rivers      : GeoDataFrame do rio (pode ser None)
    filename    : nome do arquivo de saída (salvo em assets/images/)
    threshold   : distância da APP em metros (para legenda)
    """
    import matplotlib.cm as cm

    print("\nPlotando grafo sobreposto ao mapa com scores GCN...")

    graph_crs = G.graph.get("crs", "EPSG:32723")
    cmap      = cm.RdYlGn_r          # verde (baixo risco) a vermelho (alto risco)
    vmin, vmax = 0.0, 1.0
    norm      = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm        = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, ax = ox.plot_graph(
        G,
        node_size=0,
        edge_color="#4A4A4A",
        edge_linewidth=0.25,
        bgcolor="#0d0d0d",
        show=False,
        close=False,
        figsize=(14, 11),
    )
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    if rivers is not None and not rivers.empty:
        rivers.to_crs(graph_crs).plot(
            ax=ax, color="#4FC3F7", linewidth=2.8, alpha=0.9, zorder=4
        )

    # Plota todos os nós com cor proporcional ao score de risco
    node_index = {n: i for i, n in enumerate(nodes)}
    for node in G.nodes():
        idx   = node_index.get(node)
        if idx is None:
            continue
        score = float(risk_scores[idx])
        color = cmap(norm(min(score, vmax)))
        x, y  = G.nodes[node]["x"], G.nodes[node]["y"]

        if node in risk_nodes:
            ax.scatter(x, y, c=[color], s=55, zorder=7,
                       edgecolors="white", linewidths=0.8, alpha=0.95)
        else:
            ax.scatter(x, y, c=[color], s=12, zorder=5,
                       edgecolors="none", alpha=0.80)

    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, shrink=0.75)
    cbar.set_label("Score de risco previsto (GCN)", color="white", fontsize=9)
    cbar.ax.set_facecolor("#0d0d0d")
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=8)

    ax.legend(
        handles=[
            mpatches.Patch(color="#4FC3F7", label="Rio das Velhas"),
            mpatches.Patch(color=cmap(1.0), label=f"Alto risco (nós críticos, APP ≤ {threshold:.0f} m)"),
            mpatches.Patch(color=cmap(0.0), label="Baixo risco"),
        ],
        loc="lower left", fontsize=9, framealpha=0.95,
        facecolor="#0d0d0d", labelcolor="white", edgecolor="#444444",
    )
    ax.set_title(
        "Scores de Risco de Enchente — GCN (VigiaEnchente)\n"
        "Rede viária de Sabará, MG  |  Cor: risco previsto por nó",
        color="white", fontsize=11, pad=10,
    )
    ax.set_axis_off()

    out = _img(filename)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="#0d0d0d", transparent=False)
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# GCN — Top-N nós de maior risco previstos pelo modelo
# ─────────────────────────────────────────────────────────────────────────────

def plot_top_risk_nodes(
    G,
    nodes: list,
    risk_scores: np.ndarray,
    y_true: np.ndarray,
    risk_nodes: set,
    rivers,
    top_n: int = 20,
    filename: str = "gcn_top_risk_nodes.png",
    threshold: float = 50.0,
):
    """
    Destaca os top_n nós com maior score de risco previsto pelo GCN.
    Cada nó recebe rótulo com ranking e score. Nós críticos reais (APP)
    são marcados com borda branca; falsos positivos com borda laranja.

    Uso típico em gcn.py:
        from visualization import plot_top_risk_nodes
        plot_top_risk_nodes(G, nodes, pred.numpy(), data.y.numpy(), risk_nodes, rivers)

    Parâmetros
    ──────────
    G           : grafo NetworkX projetado
    nodes       : lista de IDs de nós na mesma ordem de risk_scores
    risk_scores : array (N,) com scores previstos (output do modelo)
    y_true      : array (N,) com rótulos reais
    risk_nodes  : conjunto de IDs dos nós críticos reais (APP <= threshold m)
    rivers      : GeoDataFrame do rio (pode ser None)
    top_n       : quantos nós destacar (padrão 20)
    filename    : nome do arquivo de saída
    threshold   : distância da APP em metros (para legenda)
    """
    import matplotlib.cm as cm

    print(f"\nPlotando Top-{top_n} nós de risco previstos pelo GCN...")

    graph_crs = G.graph.get("crs", "EPSG:32723")

    # Índices dos top_n nós por score decrescente
    top_indices = np.argsort(risk_scores)[::-1][:top_n]

    # Colormap de fundo: todos os nós por score
    cmap_bg = cm.RdYlGn_r
    norm_bg = mcolors.Normalize(vmin=0.0, vmax=1.0)

    # Colormap dos top nós: score previsto
    scores_top = risk_scores[top_indices]
    cmap_top   = cm.hot_r
    norm_top   = mcolors.Normalize(vmin=scores_top.min(), vmax=scores_top.max())

    fig, ax = ox.plot_graph(
        G,
        node_size=0,
        edge_color="#222222",
        edge_linewidth=0.22,
        bgcolor="#0d0d0d",
        show=False,
        close=False,
        figsize=(14, 11),
    )
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    if rivers is not None and not rivers.empty:
        rivers.to_crs(graph_crs).plot(
            ax=ax, color="#4FC3F7", linewidth=2.5, alpha=0.9, zorder=4
        )

    # Plota todos os nós em baixa opacidade como referência
    node_index = {n: i for i, n in enumerate(nodes)}
    for node in G.nodes():
        idx = node_index.get(node)
        if idx is None:
            continue
        score = float(risk_scores[idx])
        color = cmap_bg(norm_bg(min(score, 1.0)))
        ax.scatter(G.nodes[node]["x"], G.nodes[node]["y"],
                   c=[color], s=6, zorder=3, edgecolors="none", alpha=0.35)

    # Plota os top_n nós com rótulo
    for rank, idx in enumerate(top_indices):
        node  = nodes[idx]
        score = float(risk_scores[idx])
        y_val = float(y_true[idx])
        x, y  = G.nodes[node]["x"], G.nodes[node]["y"]
        color = cmap_top(norm_top(score))

        is_critical = node in risk_nodes
        edge_col    = "white" if is_critical else "#FFB300"
        edge_lw     = 1.2     if is_critical else 0.9
        size        = max(200 - rank * 6, 60)

        ax.scatter(x, y, c=[color], s=size, zorder=9,
                   edgecolors=edge_col, linewidths=edge_lw, alpha=0.97)

        label = f"#{rank + 1} p={score:.2f}\ny={y_val:.2f}"
        ax.annotate(
            label,
            xy=(x, y), xytext=(8, 4), textcoords="offset points",
            color="white", fontsize=6.5, zorder=11,
            bbox=dict(boxstyle="round,pad=0.2", fc="#0d0d0d", ec="#555555", alpha=0.85),
        )

    sm = cm.ScalarMappable(cmap=cmap_top, norm=norm_top)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.022, pad=0.01, shrink=0.7)
    cbar.set_label(f"Score previsto — top {top_n} nós", color="white", fontsize=9)
    cbar.ax.set_facecolor("#0d0d0d")
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=8)

    ax.legend(
        handles=[
            mpatches.Patch(color="#4FC3F7", label="Rio das Velhas"),
            mpatches.Patch(facecolor="gray", edgecolor="white", linewidth=1.2,
                           label=f"Nó crítico real (APP ≤ {threshold:.0f} m)"),
            mpatches.Patch(facecolor="gray", edgecolor="#FFB300", linewidth=0.9,
                           label="Nó fora da APP (possível falso positivo)"),
        ],
        loc="lower left", fontsize=9, framealpha=0.95,
        facecolor="#0d0d0d", labelcolor="white", edgecolor="#444444",
    )
    ax.set_title(
        f"Top {top_n} nós de maior risco previsto — GCN (VigiaEnchente)\n"
        f"Borda branca: nó dentro da APP ≤ {threshold:.0f} m  |  "
        f"Borda laranja: possível falso positivo",
        color="white", fontsize=11, pad=10,
    )
    ax.set_axis_off()

    out = _img(filename)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="#0d0d0d", transparent=False)
    plt.close()
    print(f"Saved: {out}")


def plot_training_curve(
    loss_history,
    filename: str = "gcn_training_curve.png",
):
    """
    Gera a curva de convergência do treinamento da GCN.

    Parameters
    ----------
    loss_history : list[float]
        Histórico da loss por época.
    filename : str
        Nome do arquivo de saída.
    """

    print("\nPlotando curva de treinamento da GCN...")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        range(1, len(loss_history) + 1),
        loss_history,
        linewidth=2,
        label="Loss de treinamento"
    )

    ax.set_xlabel("Época")
    ax.set_ylabel("MSE")
    ax.set_title("Convergência da GCN durante o treinamento")

    ax.grid(True, alpha=0.3)

    ax.legend()

    out = _img(filename)

    plt.tight_layout()
    plt.savefig(
        out,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Salvo: {out}")