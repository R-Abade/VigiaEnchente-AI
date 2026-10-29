"""
analysis_blocked.py

Orquestrador offline: baixa o grafo de Sabará, escolhe o par de nós mais
distantes, calcula a rota original e uma rota segura que evita (ou minimiza)
a passagem por nós de risco próximos ao Rio das Velhas.

Estratégia: em vez de remover os nós de risco (o que pode desconectar o
grafo), aplica-se uma penalização alta às arestas incidentes a eles via
flood.apply_flood_penalty. O caminho mais curto então:
  • evita 100% dos nós de risco quando existe alternativa;
  • atravessa o mínimo possível quando for inevitável.

Uso:
    python -m scripts.analysis_blocked
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx

from src.graph.loader import build_graph
from src.graph.flood import identify_flood_risk_nodes, apply_flood_penalty, reset_weights
from src.graph.routing import run_algorithms
from src.graph.visualization import plot_blocked_comparison
from scripts.legacy_metrics import compute_fw_matrix, compute_diameter


PENALTY_FACTOR = 5


# ── Par mais distante ─────────────────────────────────────────────────────────

def find_distant_pair(G):
    print("Calculando par mais distante (Floyd–Warshall)...")
    matrix, nodes = compute_fw_matrix(G)
    source, target, dist, _, _ = compute_diameter(G, matrix=matrix, nodes=nodes)
    print(f"  Par escolhido: {source} → {target}  ({dist:.1f} m)")
    return source, target


# ── Rotas ─────────────────────────────────────────────────────────────────────

def compute_original_route(G, source, target):
    print(f"\nCalculando rota original {source}→{target}...")
    return run_algorithms(G, source, target)


def compute_safe_route(G, source, target, risk_nodes, penalty_factor):
    print(f"\nCalculando rota segura {source}→{target} (evitando risco)...")
    apply_flood_penalty(G, risk_nodes, penalty_factor)
    path = nx.dijkstra_path(G, source, target, weight="weight")
    reset_weights(G)
    dist = nx.path_weight(G, path, weight="weight")
    return path, dist


def build_safe_results(route_safe, dist_safe):
    return {
        "Dijkstra": {
            "dist":  dist_safe,
            "nodes": len(route_safe),
            "time":  0.0,
            "path":  route_safe,
        },
    }


def build_penalized_graph(G, risk_nodes, penalty_factor):
    G_safe = G.copy()
    apply_flood_penalty(G_safe, risk_nodes, penalty_factor)
    return G_safe


# ── Nós de risco na rota ──────────────────────────────────────────────────────

def risk_nodes_in_route(route, risk_nodes):
    return [node for node in route if node in risk_nodes]


# ── Relatório ─────────────────────────────────────────────────────────────────

def print_report(results_orig, route_orig, route_safe, dist_safe, source, target, risk_nodes):
    SEP_DOUBLE = "═" * 68
    SEP_SINGLE = "─" * 68

    d_orig    = results_orig["Dijkstra"]["dist"]
    risk_orig = risk_nodes_in_route(route_orig, risk_nodes)
    risk_safe = risk_nodes_in_route(route_safe, risk_nodes)

    delta = dist_safe - d_orig
    pct   = 100 * delta / d_orig

    print(f"\n{SEP_DOUBLE}")
    print(f"  RESULTADOS — par {source}→{target}")
    print(SEP_SINGLE)

    print(f"  {'Algoritmo':<12} {'Dist. (m)':>12} {'Nós':>6} {'Tempo (s)':>10}")
    for algo, r in results_orig.items():
        dist_str  = f"{r['dist']:.2f}" if r['dist'] is not None else "N/A"
        nodes_str = str(r['nodes']) if r['nodes'] is not None else "N/A"
        print(f"  {algo:<12} {dist_str:>12} {nodes_str:>6} {r['time']:>10.5f}")

    print(f"\n  Comparação:")
    print(f"    Rota original : {len(route_orig):>3} nós | "
          f"{d_orig:>10.2f} m | risco: {len(risk_orig)} nó(s)")
    print(f"    Rota segura   : {len(route_safe):>3} nós | "
          f"{dist_safe:>10.2f} m | risco: {len(risk_safe)} nó(s)")
    print(f"    Δ             : {delta:+.2f} m ({pct:+.2f} %)")

    print()
    if risk_safe:
        print(f"  ⚠  {len(risk_safe)} nó(s) de risco inevitáveis: {risk_safe}")
        print(f"     (não há caminho que evite 100% dos riscos)")
    else:
        print(f"  ✓  Rota 100% livre de nós de risco.")

    print(SEP_DOUBLE)


# ── Orquestrador ──────────────────────────────────────────────────────────────

def run(place: str = "Sabará, Minas Gerais, Brazil", threshold: float = 50.0):
    G = build_graph(place)

    source, target = find_distant_pair(G)

    results_orig = compute_original_route(G, source, target)
    route_orig   = results_orig["Dijkstra"]["path"]

    risk_nodes, risk_info, rivers = identify_flood_risk_nodes(G, place, threshold)

    route_safe, dist_safe = compute_safe_route(G, source, target, risk_nodes, PENALTY_FACTOR)

    print_report(results_orig, route_orig, route_safe, dist_safe, source, target, risk_nodes)

    G_safe       = build_penalized_graph(G, risk_nodes, PENALTY_FACTOR)
    results_safe = build_safe_results(route_safe, dist_safe)

    plot_blocked_comparison(
        G, G_safe, risk_nodes, risk_info, rivers,
        route_orig, route_safe, source, target,
        source, target,
        False,
        results_orig, results_safe,
        threshold=threshold,
        filename="grafo_bloqueado.png",
    )


if __name__ == "__main__":
    run()
