"""
visualization.py

Funções de visualização para o grafo viário de Sabará.
Todas as imagens são salvas em assets/images/.
"""

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
import osmnx as ox

warnings.filterwarnings("ignore")

IMAGES_DIR = Path(__file__).resolve().parents[2] / "assets" / "images"


def _ensure_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _img(filename: str) -> Path:
    return _ensure_dir(IMAGES_DIR / filename)


def draw_base_graph(ax, graph, rivers, graph_crs, risk_info=None, show_risk=True):
    """Renderiza arestas do grafo, rio e nós de risco num eixo."""
    ox.plot_graph(
        graph, ax=ax,
        node_size=0, edge_color="#2a2a2a", edge_linewidth=0.3,
        bgcolor="#0F0F0F", show=False, close=False,
    )
    if rivers is not None and not rivers.empty:
        rivers.to_crs(graph_crs).plot(
            ax=ax, color="#4FC3F7", linewidth=2.5, alpha=0.85, zorder=4
        )
    if show_risk and risk_info:
        xs = [node["x"] for node in risk_info]
        ys = [node["y"] for node in risk_info]
        ax.scatter(xs, ys, c="#00E5FF", s=28, zorder=5,
                   edgecolors="#003F5C", linewidths=0.4, alpha=0.85)


def draw_route(ax, graph, route, color, linewidth=2.4, linestyle="-", zorder=6):
    """Traça uma rota segmento a segmento sobre um eixo."""
    if not route:
        return
    for u, v in zip(route[:-1], route[1:]):
        if u in graph.nodes() and v in graph.nodes():
            x1, y1 = graph.nodes[u]["x"], graph.nodes[u]["y"]
            x2, y2 = graph.nodes[v]["x"], graph.nodes[v]["y"]
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth,
                    alpha=0.9, zorder=zorder, linestyle=linestyle)


def plot_route_comparison(
    graph,
    comparison: dict,
    risk_nodes: set,
    risk_info: list,
    rivers,
    threshold: float = 50.0,
    filename: str = "q4_route_comparison.png",
):
    """Plota rota original e penalizada sobre o grafo."""
    print("\nPlotando comparação de rotas...")
    graph_crs = graph.graph.get("crs", "EPSG:32723")

    fig, ax = ox.plot_graph(
        graph, node_size=0, edge_color="#333333", edge_linewidth=0.3,
        bgcolor="#0F0F0F", show=False, close=False, figsize=(14, 11),
    )

    if rivers is not None and not rivers.empty:
        rivers.to_crs(graph_crs).plot(
            ax=ax, color="#4FC3F7", linewidth=2.5, alpha=0.9, zorder=4
        )

    for node in risk_info:
        ax.scatter(node["x"], node["y"], c="#00E5FF", s=45, zorder=5,
                   edgecolors="#004D6E", linewidths=0.6)

    draw_route(ax, graph, comparison["route_orig"], "#E53935", linewidth=2.5, zorder=6)

    pen_color = "#FFD600" if not comparison["same_route"] else "#FFB300"
    draw_route(ax, graph, comparison["route_pen"], pen_color, linewidth=1.8, linestyle="--", zorder=7)

    src, tgt = comparison["route_orig"][0], comparison["route_orig"][-1]
    ax.scatter(graph.nodes[src]["x"], graph.nodes[src]["y"],
               c="#00C853", s=150, zorder=9, edgecolors="white", linewidths=1.2)
    ax.scatter(graph.nodes[tgt]["x"], graph.nodes[tgt]["y"],
               c="#FFB300", s=150, zorder=9, edgecolors="white", linewidths=1.2)

    pen_label = "Rota penalizada (diferente)" if not comparison["same_route"] else "Rota penalizada (idêntica)"
    legend = [
        mpatches.Patch(color="#E53935",
                       label=f"Rota original — {comparison['dist_orig']:.0f} m | {len(comparison['risk_orig'])} nós de risco"),
        mpatches.Patch(color=pen_color,
                       label=f"{pen_label} — {comparison['dist_pen']:.0f} m | {len(comparison['risk_pen'])} nós de risco"),
        mpatches.Patch(color="#4FC3F7", label="Rio das Velhas"),
        mpatches.Patch(color="#00E5FF", label=f"Nós de risco (≤ {threshold} m)"),
        mpatches.Patch(color="#00C853", label="Origem"),
        mpatches.Patch(color="#FFB300", label="Destino"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=8.5,
              framealpha=0.88, facecolor="#1A1A1A", labelcolor="white")

    delta = comparison["dist_pen"] - comparison["dist_orig"]
    status = "ROTA ALTERADA" if not comparison["same_route"] else "ROTA INALTERADA — sem alternativa"
    ax.set_title(
        f"Comparação de Rotas: Original vs Penalizada (×2 em nós de risco)\n"
        f"{status}  |  Δ distância = {delta:+.1f} m",
        fontsize=11, color="white", pad=10,
    )

    out = _img(filename)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Salvo: {out}")


def plot_fw_heatmap(
    matrix: np.ndarray,
    place_name: str,
    filename: str = "fw_heatmap.png",
):
    """Heatmap Floyd-Warshall com cmap='magma'."""
    finite_max = float(np.nanmax(matrix[matrix != np.inf]))
    display = np.where(matrix == np.inf, finite_max * 1.2, matrix)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(display, cmap="magma", interpolation="nearest", aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Distância do menor caminho (metros)")
    ax.set_title(f"Floyd-Warshall — {place_name}")
    ax.set_xlabel("Nó destino")
    ax.set_ylabel("Nó origem")

    out = _img(filename)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {out}")


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
    """Heatmap reordenado por Ward com nós isolados e fronteiras de cluster."""
    nodes_ord = [nodes[i] for i in order]
    labels_ord = labels[order]
    unique_clusters = np.unique(labels)

    finite_max = float(np.nanmax(matrix[matrix != np.inf]))
    matrix_vis = np.where(matrix == np.inf, finite_max * 1.2, matrix)
    matrix_ord = matrix_vis[np.ix_(order, order)]

    cluster_colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    color_map = {c: cluster_colors[i] for i, c in enumerate(unique_clusters)}

    fig = plt.figure(figsize=(14, 12), facecolor="#0F0F0F")
    fig.suptitle(
        "Floyd-Warshall — Sabará (Ward)\nNós isolados (Q5) | Fronteiras de cluster (Q6)",
        color="white", fontsize=11, y=0.99,
    )

    gs = fig.add_gridspec(2, 2, width_ratios=[30, 1],
                          height_ratios=[1, 30], hspace=0.02, wspace=0.02)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_cbar = fig.add_subplot(gs[1, 1])

    norm = mcolors.PowerNorm(gamma=0.5, vmin=matrix_ord.min(), vmax=matrix_ord.max())
    im = ax_heat.imshow(matrix_ord / 1000, cmap="magma", norm=norm,
                        aspect="auto", interpolation="nearest")
    ax_heat.set_xlabel("Índice do nó (reordenado)", color="white", fontsize=9)
    ax_heat.set_ylabel("Índice do nó (reordenado)", color="white", fontsize=9)
    ax_heat.tick_params(colors="white", labelsize=7)

    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label("Menor caminho (km)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=7)

    cluster_bar = np.array([labels_ord]).astype(float)
    cluster_cmap = mcolors.ListedColormap([color_map[c] for c in unique_clusters])
    ax_top.imshow(cluster_bar, aspect="auto", cmap=cluster_cmap,
                  vmin=unique_clusters.min(), vmax=unique_clusters.max())
    ax_top.set_yticks([])
    ax_top.set_xticks([])
    ax_top.set_title("Cluster", color="white", fontsize=8, pad=3)
    for spine in ax_top.spines.values():
        spine.set_edgecolor("#444444")

    isolated_set = {node for node, _ in isolated_nodes}
    for node in isolated_set:
        if node in nodes_ord:
            pos = nodes_ord.index(node)
            ax_heat.axvline(pos, color="#00E5FF", linewidth=0.7, alpha=0.7)
            ax_heat.axhline(pos, color="#00E5FF", linewidth=0.7, alpha=0.7)

    prev = 0
    for c in unique_clusters[:-1]:
        count = int((labels_ord == c).sum())
        boundary = prev + count
        ax_heat.axvline(boundary - 0.5, color="white", linewidth=0.9, alpha=0.45)
        ax_heat.axhline(boundary - 0.5, color="white", linewidth=0.9, alpha=0.45)
        prev = boundary

    legend = [
        mpatches.Patch(color="#00E5FF",
                       label=f"Nós isolados — top {len(isolated_nodes)} (Q5, p{int(cluster_threshold_pct * 100)})"),
    ] + [
        mpatches.Patch(color=cluster_colors[i], label=f"Cluster {c}")
        for i, c in enumerate(unique_clusters)
    ]
    ax_heat.legend(handles=legend, loc="lower right", fontsize=7,
                   framealpha=0.5, facecolor="#1A1A1A", labelcolor="white")

    out = _img(filename)
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Salvo: {out}")


def plot_isolated_nodes(
    graph,
    matrix: np.ndarray,
    nodes: list,
    isolated_nodes: list,
    rivers=None,
    filename: str = "nos_isolados.png",
):
    """Mapa com todos os nós coloridos por distância média e isolados destacados."""
    graph_crs = graph.graph.get("crs", "EPSG:32723")
    avg_dist = matrix.mean(axis=1)

    dist_vals = np.array([d for _, d in isolated_nodes])
    cmap_iso = plt.cm.YlOrRd_r
    norm_iso = plt.Normalize(vmin=dist_vals.min(), vmax=dist_vals.max())
    all_norm = plt.Normalize(vmin=avg_dist.min(), vmax=avg_dist.max())
    cmap_bg = plt.cm.magma

    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor("#0F0F0F")
    ox.plot_graph(graph, ax=ax, node_size=0, edge_color="#1e1e1e",
                  edge_linewidth=0.28, bgcolor="#0F0F0F", show=False, close=False)

    for position, node in enumerate(nodes):
        ax.scatter(graph.nodes[node]["x"], graph.nodes[node]["y"],
                   c=[cmap_bg(all_norm(avg_dist[position]))], s=5, zorder=2, alpha=0.4)

    if rivers is not None and not rivers.empty:
        rivers.to_crs(graph_crs).plot(ax=ax, color="#4FC3F7",
                                      linewidth=3, alpha=0.8, zorder=4)

    for rank, (node, dist) in enumerate(isolated_nodes):
        x, y = graph.nodes[node]["x"], graph.nodes[node]["y"]
        color = cmap_iso(norm_iso(dist))
        size = 260 - rank * 8
        ax.scatter(x, y, c=[color], s=size, zorder=8,
                   edgecolors="white", linewidths=1.0, alpha=0.97)
        ax.annotate(
            f"#{rank + 1}  {dist / 1000:.1f} km",
            xy=(x, y), xytext=(10, 5), textcoords="offset points",
            color="white", fontsize=7, zorder=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="#0F0F0F", ec="#444444", alpha=0.82),
        )

    sm_iso = plt.cm.ScalarMappable(cmap=cmap_iso, norm=norm_iso)
    sm_iso.set_array([])
    cbar1 = plt.colorbar(sm_iso, ax=ax, fraction=0.022, pad=0.02, shrink=0.7)
    cbar1.set_label(f"Dist. média — top {len(isolated_nodes)} isolados (m)",
                    color="white", fontsize=9)
    cbar1.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=8)

    sm_bg = plt.cm.ScalarMappable(cmap=cmap_bg, norm=all_norm)
    sm_bg.set_array([])
    cbar2 = plt.colorbar(sm_bg, ax=ax, fraction=0.022, pad=0.08, shrink=0.7)
    cbar2.set_label("Dist. média — todos os nós (m)", color="white", fontsize=9)
    cbar2.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=8)

    ax.set_title(
        f"Top {len(isolated_nodes)} nós mais isolados — Sabará\n"
        "(Floyd-Warshall — dist. média; tamanho proporcional ao isolamento)",
        fontsize=11, color="white", pad=10,
    )
    ax.set_axis_off()
    plt.tight_layout()

    out = _img(filename)
    plt.savefig(out, dpi=250, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Salvo: {out}")


def plot_nearest_river_nodes(
    graph,
    nearby: list,
    rivers,
    threshold: float = 50.0,
    top_n: int = 20,
    filename: str = "nearest_river_nodes.png",
):
    """Mapa com gradiente de proximidade ao rio."""
    graph_crs = graph.graph.get("crs", "EPSG:32723")
    cmap = cm.RdYlGn_r
    norm = mcolors.Normalize(vmin=0, vmax=threshold)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)

    fig, ax = ox.plot_graph(
        graph, node_size=0, edge_color="#2a2a2a", edge_linewidth=0.25,
        bgcolor="#0d0d0d", show=False, close=False, figsize=(13, 10),
    )

    if rivers is not None and not rivers.empty:
        rivers.to_crs(graph_crs).plot(ax=ax, color="#4FC3F7",
                                      linewidth=2.8, alpha=0.9, zorder=4)

    for node in nearby:
        color = cmap(norm(node["dist_m"]))
        ax.scatter(node["x"], node["y"], c=[color], s=35, zorder=5,
                   edgecolors="none", alpha=0.75)

    for rank, node in enumerate(nearby[:top_n], 1):
        color = cmap(norm(node["dist_m"]))
        ax.scatter(node["x"], node["y"], c=[color], s=90, zorder=7,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(str(rank), xy=(node["x"], node["y"]), xytext=(4, 4),
                    textcoords="offset points", fontsize=6,
                    color="white", zorder=8)

    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Distância ao rio (m)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    legend = [
        mpatches.Patch(color="#4FC3F7", label="Rio das Velhas"),
        mpatches.Patch(color=cmap(0.0), label="Muito próximo (0 m)"),
        mpatches.Patch(color=cmap(0.5), label=f"~{threshold / 2:.0f} m"),
        mpatches.Patch(color=cmap(1.0), label=f"~{threshold:.0f} m"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=8, framealpha=0.85,
              facecolor="#1a1a1a", labelcolor="white")
    ax.set_title(
        f"Nós mais próximos do Rio das Velhas (≤ {threshold} m)\n"
        f"{len(nearby)} nós encontrados | Top {top_n} numerados | Gradiente: distância ao rio",
        color="white", fontsize=10, pad=8,
    )

    out = _img(filename)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Salvo: {out}")


def plot_blocked_comparison(
    graph, graph_blocked,
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
    """Dois painéis: grafo completo vs bloqueado."""
    graph_crs = graph.graph.get("crs", "EPSG:32723")
    fig, axes = plt.subplots(1, 2, figsize=(20, 11))
    fig.patch.set_facecolor("#0F0F0F")

    d_o = results_orig["Dijkstra"]["dist"] or 0
    n_o = results_orig["Dijkstra"]["nodes"] or 0
    d_b = results_blocked["Dijkstra"]["dist"]
    n_b = results_blocked["Dijkstra"]["nodes"]
    d_str = f"{d_b / 1000:.2f} km" if d_b else "N/A"
    n_str = str(n_b) if n_b else "N/A"

    # Painel esquerdo: original
    ax_left = axes[0]
    draw_base_graph(ax_left, graph, rivers, graph_crs, risk_info, show_risk=True)
    draw_route(ax_left, graph, route_orig, "#E53935", linewidth=2.5)
    if route_orig:
        ax_left.scatter(graph.nodes[src_orig]["x"], graph.nodes[src_orig]["y"],
                        c="#00C853", s=200, zorder=9, edgecolors="white", linewidths=1.2)
        ax_left.scatter(graph.nodes[tgt_orig]["x"], graph.nodes[tgt_orig]["y"],
                        c="#FFB300", s=200, zorder=9, edgecolors="white", linewidths=1.2)

    risk_on_orig = sum(1 for node in (route_orig or []) if node in risk_nodes)
    ax_left.set_title(
        f"Grafo original — rota {src_orig}→{tgt_orig}\n"
        f"Distância: {d_o / 1000:.2f} km | Nós: {n_o} | Nós de risco: {risk_on_orig}",
        fontsize=10, color="white", pad=8,
    )
    legend = [
        mpatches.Patch(color="#E53935", label=f"Rota original ({d_o / 1000:.2f} km, {n_o} nós)"),
        mpatches.Patch(color="#4FC3F7", label="Rio das Velhas"),
        mpatches.Patch(color="#00E5FF", label=f"Nós de risco (≤ {threshold:.0f} m — n={len(risk_nodes)})"),
        mpatches.Patch(color="#00C853", label="Origem"),
        mpatches.Patch(color="#FFB300", label="Destino"),
    ]
    ax_left.legend(handles=legend, loc="lower left", fontsize=8.5,
                   framealpha=0.88, facecolor="#1A1A1A", labelcolor="white")
    ax_left.set_axis_off()

    # Painel direito: bloqueado
    ax_right = axes[1]
    draw_base_graph(ax_right, graph_blocked, rivers, graph_crs, risk_info=[], show_risk=False)

    if disconnected:
        for node_id in [src_orig, tgt_orig]:
            ref = graph if node_id not in graph_blocked.nodes() else graph_blocked
            ax_right.scatter(ref.nodes[node_id]["x"], ref.nodes[node_id]["y"],
                             c="#E53935", s=300, marker="x", zorder=9,
                             linewidths=2.5, alpha=0.7)
        draw_route(ax_right, graph_blocked, route_alt, "#FFD600", linewidth=2.5, linestyle="--")
        if route_alt:
            if src_alt in graph_blocked.nodes():
                ax_right.scatter(graph_blocked.nodes[src_alt]["x"], graph_blocked.nodes[src_alt]["y"],
                                 c="#00C853", s=200, zorder=9, edgecolors="white", linewidths=1.2)
            if tgt_alt in graph_blocked.nodes():
                ax_right.scatter(graph_blocked.nodes[tgt_alt]["x"], graph_blocked.nodes[tgt_alt]["y"],
                                 c="#FFB300", s=200, zorder=9, edgecolors="white", linewidths=1.2)
        title = f"Grafo bloqueado — PAR ORIGINAL DESCONECTADO\nNovo par: {src_alt}→{tgt_alt} | Distância: {d_str} | Nós: {n_str}"
        legend = [
            mpatches.Patch(color="#FFD600", label=f"Nova rota — novo par ({d_str}, {n_str} nós)"),
            mpatches.Patch(color="#E53935", label=f"Par original desconectado ({src_orig}→{tgt_orig})"),
            mpatches.Patch(color="#4FC3F7", label="Rio das Velhas"),
            mpatches.Patch(color="#00C853", label=f"Nova origem ({src_alt})"),
            mpatches.Patch(color="#FFB300", label=f"Novo destino ({tgt_alt})"),
        ]
    else:
        draw_route(ax_right, graph_blocked, route_alt, "#FFD600", linewidth=2.5, linestyle="--")
        if route_alt:
            ax_right.scatter(graph_blocked.nodes[src_alt]["x"], graph_blocked.nodes[src_alt]["y"],
                             c="#00C853", s=200, zorder=9, edgecolors="white", linewidths=1.2)
            ax_right.scatter(graph_blocked.nodes[tgt_alt]["x"], graph_blocked.nodes[tgt_alt]["y"],
                             c="#FFB300", s=200, zorder=9, edgecolors="white", linewidths=1.2)
        delta = (d_b - d_o) if (d_b and d_o) else 0
        title = f"Grafo bloqueado — rota alternativa {src_orig}→{tgt_orig}\nDistância: {d_str} (Δ {delta / 1000:+.2f} km) | Nós: {n_str}"
        legend = [
            mpatches.Patch(color="#FFD600", label=f"Rota alternativa ({d_str}, {n_str} nós)"),
            mpatches.Patch(color="#4FC3F7", label="Rio das Velhas"),
            mpatches.Patch(color="#00C853", label="Origem"),
            mpatches.Patch(color="#FFB300", label="Destino"),
        ]

    ax_right.set_title(title, fontsize=10, color="white", pad=8)
    ax_right.legend(handles=legend, loc="lower left", fontsize=8.5,
                    framealpha=0.88, facecolor="#1A1A1A", labelcolor="white")
    ax_right.set_axis_off()

    out = _img(filename)
    plt.tight_layout(pad=1.2)
    plt.savefig(out, dpi=250, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Salvo: {out}")


def plot_degree_distribution(graph, filename: str = "degree_distribution.png"):
    """Histograma da distribuição de grau dos nós."""
    degrees = [d for _, d in graph.degree()]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(degrees, bins=20, color="#2196F3", edgecolor="white")
    ax.set_title("Distribuição do Grau dos Nós — Sabará")
    ax.set_xlabel("Grau")
    ax.set_ylabel("Frequência")

    out = _img(filename)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {out}")


def draw_compass_rose(ax, center=(0.10, 0.90), size=0.06,
                      color="#111111", fontsize=12, bg_alpha=0.6):
    """Desenha uma rosa dos ventos (N, S, L, O) num eixo."""
    x0, y0 = center
    effective_radius = size * 1.25 + 0.03
    x0 = min(max(x0, effective_radius), 1 - effective_radius)
    y0 = min(max(y0, effective_radius), 1 - effective_radius)

    dirs = [("N", 90), ("L", 0), ("S", 270), ("O", 180)]

    circle = plt.Circle((x0, y0), radius=size * 0.9, facecolor="white",
                        edgecolor=color, linewidth=1.0,
                        alpha=bg_alpha, transform=ax.transAxes, zorder=10)
    ax.add_patch(circle)

    for angle in [0, 90, 180, 270]:
        rad = np.radians(angle)
        dx = size * np.cos(rad)
        dy = size * np.sin(rad)
        ax.plot([x0, x0 + dx], [y0, y0 + dy], color=color, linewidth=1.8,
                transform=ax.transAxes, zorder=11)

    for label, angle in dirs:
        rad = np.radians(angle)
        dx = 1.25 * size * np.cos(rad)
        dy = 1.25 * size * np.sin(rad)
        ax.text(x0 + dx, y0 + dy, label, color=color, fontsize=fontsize,
                ha="center", va="center", weight="bold",
                transform=ax.transAxes, zorder=12,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          alpha=0.8, edgecolor="none"))


def plot_graph_on_map(
    graph,
    nodes: list,
    risk_scores: np.ndarray,
    risk_nodes: set,
    rivers,
    filename: str = "gcn_risk_map.png",
    threshold: float = 50.0,
    figsize: tuple = (12, 9),
    also_save_pdf: bool = True,
    study_area=None,
    add_basemap: bool = True,
):
    """
    Plota o grafo com fundo claro, scores de risco, barra de escala e rosa dos ventos.
    """
    print("\nPlotando grafo com scores GCN (fundo claro)...")

    graph_crs = graph.graph.get("crs", "EPSG:32723")

    bg_color = "#ffffff"
    edge_color = "#4d4d4d"
    ink_color = "#111111"
    legend_face = "#ffffff"
    legend_edge = "#333333"

    mpl.rcParams["savefig.facecolor"] = bg_color
    mpl.rcParams["figure.facecolor"] = bg_color
    mpl.rcParams["axes.facecolor"] = bg_color
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["STIX", "DejaVu Serif", "Times New Roman"]
    mpl.rcParams["mathtext.fontset"] = "stix"

    fig, ax = ox.plot_graph(
        graph,
        node_size=0,
        edge_color=edge_color,
        edge_linewidth=0.4,
        bgcolor=bg_color,
        show=False,
        close=False,
        figsize=figsize,
    )
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    if add_basemap:
        try:
            import contextily as ctx
            ctx.add_basemap(ax, crs=graph_crs, source=ctx.providers.CartoDB.PositronNoLabels,
                            alpha=0.9, zorder=0)
        except Exception as e:
            print(f"Basemap não adicionado: {e}")

    if rivers is not None and not rivers.empty:
        rivers.to_crs(graph_crs).plot(ax=ax, color="#0077B6", linewidth=3.0, alpha=0.95, zorder=4)

    cmap_risk = cm.get_cmap("RdYlGn_r")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    sm = cm.ScalarMappable(cmap=cmap_risk, norm=norm)
    sm.set_array([])

    node_index = {node: position for position, node in enumerate(nodes)}
    low_x, low_y, low_c = [], [], []
    high_x, high_y, high_c = [], [], []

    for node in graph.nodes():
        position = node_index.get(node)
        if position is None:
            continue
        score = float(risk_scores[position])
        score_clipped = max(0.0, min(score, 1.0))
        color = cmap_risk(norm(score_clipped))
        x, y = graph.nodes[node]["x"], graph.nodes[node]["y"]

        if node in risk_nodes:
            high_x.append(x)
            high_y.append(y)
            high_c.append(color)
        else:
            low_x.append(x)
            low_y.append(y)
            low_c.append(color)

    if low_x:
        ax.scatter(low_x, low_y, c=low_c, s=22, zorder=5,
                   edgecolors="#333333", linewidths=0.3, alpha=0.90, rasterized=True)
    if high_x:
        ax.scatter(high_x, high_y, c=high_c, s=85, zorder=7,
                   edgecolors="#111111", linewidths=1.2, alpha=0.95, rasterized=True)

    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.01, shrink=0.75)
    cbar.set_label("Score de risco previsto (GCN)", color=ink_color, fontsize=15)
    cbar.ax.set_facecolor(bg_color)
    cbar.outline.set_edgecolor(ink_color)
    cbar.ax.yaxis.set_tick_params(color=ink_color, labelcolor=ink_color, labelsize=13)
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])

    xs = [graph.nodes[node]["x"] for node in graph.nodes()]
    ys = [graph.nodes[node]["y"] for node in graph.nodes()]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    if study_area is not None and not study_area.empty:
        study_area.to_crs(graph_crs).boundary.plot(
            ax=ax, color=ink_color, linewidth=2.5, linestyle="--",
            alpha=0.9, zorder=6,
        )
        bounds = study_area.to_crs(graph_crs).total_bounds
        minx, miny, maxx, maxy = bounds
        boundary_label = "Limite da área de estudo"
    else:
        margin = 0.02
        bx0 = minx - (maxx - minx) * margin
        bx1 = maxx + (maxx - minx) * margin
        by0 = miny - (maxy - miny) * margin
        by1 = maxy + (maxy - miny) * margin
        rect = plt.Rectangle(
            (bx0, by0), bx1 - bx0, by1 - by0,
            linewidth=2.5, edgecolor=ink_color, facecolor="none",
            linestyle="--", alpha=0.85, zorder=6,
        )
        ax.add_patch(rect)
        boundary_label = "Área de estudo"

    handles = [
        mpatches.Patch(color="#0077B6", label="Rio das Velhas"),
        mpatches.Patch(color=cmap_risk(1.0), label=f"Alto risco (APP ≤ {threshold:.0f} m)"),
        mpatches.Patch(color=cmap_risk(0.0), label="Baixo risco (seguro)"),
        mpatches.Patch(facecolor="none", edgecolor=ink_color, linestyle="--",
                       label=boundary_label),
    ]

    ax.set_title(
        "Scores de Risco de Enchente - Rede viária de Sabará, MG",
        color=ink_color,
        fontsize=18,
        pad=4,
    )

    # Barra de escala
    width_deg = maxx - minx
    scale_km = max(1.0, width_deg * 0.05 / 1000.0)
    scale_m = scale_km * 1000.0

    fontprops = fm.FontProperties(size=13, weight="bold")
    scalebar = AnchoredSizeBar(
        ax.transData,
        scale_m,
        f"{scale_km:.1f} km",
        loc="lower right",
        pad=0.3,
        borderpad=0.3,
        color=ink_color,
        frameon=False,
        size_vertical=2.0,
        fontproperties=fontprops,
        sep=3,
    )
    ax.add_artist(scalebar)

    # Rosa dos ventos
    draw_compass_rose(
        ax,
        center=(0.10, 0.90),
        size=0.06,
        color=ink_color,
        fontsize=12,
        bg_alpha=0.6,
    )

    ax.set_axis_off()
    ax.set_xlim(minx - (maxx - minx) * 0.03, maxx + (maxx - minx) * 0.03)
    ax.set_ylim(miny - (maxy - miny) * 0.03, maxy + (maxy - miny) * 0.03)

    legend = ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        fontsize=13,
        framealpha=0.95,
        facecolor=legend_face,
        labelcolor=ink_color,
        edgecolor=legend_edge,
        handlelength=1.5,
        handletextpad=0.6,
        columnspacing=1.0,
        borderpad=0.4,
    )

    n_legend_rows = -(-len(handles) // 2)
    bottom_margin = 0.03 + n_legend_rows * 0.045
    fig.subplots_adjust(bottom=bottom_margin, top=0.97, left=0.0, right=1.0)

    out = _img(filename)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.0,
                facecolor=fig.get_facecolor(), transparent=False)

    if also_save_pdf:
        out_pdf = out.with_suffix(".pdf")
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.0,
                    facecolor=fig.get_facecolor(), transparent=False)
        print(f"Salvo (vetor): {out_pdf}")

    plt.close(fig)
    print(f"Salvo: {out}")


def plot_top_risk_nodes(
    graph,
    nodes: list,
    risk_scores: np.ndarray,
    y_true: np.ndarray,
    risk_nodes: set,
    rivers,
    top_n: int = 20,
    filename: str = "gcn_top_risk_nodes.png",
    threshold: float = 50.0,
):
    """Destaca os top_n nós com maior score de risco previsto pelo GCN."""
    print(f"\nPlotando Top-{top_n} nós de risco previstos pelo GCN...")

    graph_crs = graph.graph.get("crs", "EPSG:32723")
    top_indices = np.argsort(risk_scores)[::-1][:top_n]

    cmap_bg = cm.RdYlGn_r
    norm_bg = mcolors.Normalize(vmin=0.0, vmax=1.0)

    scores_top = risk_scores[top_indices]
    cmap_top = cm.hot_r
    norm_top = mcolors.Normalize(vmin=scores_top.min(), vmax=scores_top.max())

    fig, ax = ox.plot_graph(
        graph,
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

    node_index = {node: position for position, node in enumerate(nodes)}

    for node in graph.nodes():
        position = node_index.get(node)
        if position is None:
            continue
        score = float(risk_scores[position])
        color = cmap_bg(norm_bg(min(score, 1.0)))
        ax.scatter(graph.nodes[node]["x"], graph.nodes[node]["y"],
                   c=[color], s=6, zorder=3, edgecolors="none", alpha=0.35)

    for rank, position in enumerate(top_indices):
        node = nodes[position]
        score = float(risk_scores[position])
        y_val = float(y_true[position])
        x, y = graph.nodes[node]["x"], graph.nodes[node]["y"]
        color = cmap_top(norm_top(score))

        is_critical = node in risk_nodes
        edge_color = "white" if is_critical else "#FFB300"
        edge_width = 1.2 if is_critical else 0.9
        size = max(200 - rank * 6, 60)

        ax.scatter(x, y, c=[color], s=size, zorder=9,
                   edgecolors=edge_color, linewidths=edge_width, alpha=0.97)

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

    legend = [
        mpatches.Patch(color="#4FC3F7", label="Rio das Velhas"),
        mpatches.Patch(facecolor="gray", edgecolor="white", linewidth=1.2,
                       label=f"Nó crítico real (APP ≤ {threshold:.0f} m)"),
        mpatches.Patch(facecolor="gray", edgecolor="#FFB300", linewidth=0.9,
                       label="Nó fora da APP (possível falso positivo)"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=9, framealpha=0.95,
              facecolor="#0d0d0d", labelcolor="white", edgecolor="#444444")
    ax.set_title(
        f"Top {top_n} nós de maior risco previsto — GCN (VigiaEnchente)\n"
        f"Borda branca: nó dentro da APP ≤ {threshold:.0f} m | "
        f"Borda laranja: possível falso positivo",
        color="white", fontsize=11, pad=10,
    )
    ax.set_axis_off()

    out = _img(filename)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="#0d0d0d", transparent=False)
    plt.close()
    print(f"Salvo: {out}")


def plot_training_curve(
    loss_history: list,
    filename: str = "gcn_training_curve.png",
):
    """Curva de convergência do treinamento da GCN."""
    print("\nPlotando curva de treinamento da GCN...")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(loss_history) + 1), loss_history, linewidth=2, label="Loss de treinamento")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE")
    ax.set_title("Convergência da GCN")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out = _img(filename)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {out}")