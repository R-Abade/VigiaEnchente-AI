"""
routing.py
──────────
Algoritmos de caminho mínimo e comparação de rotas para o grafo viário.

Responsabilidades
  • Executar Dijkstra, Bellman-Ford e A* com heurística euclidiana
  • Comparar rota original vs rota penalizada por enchente (Q4)
  • Tratar pares desconectados (NodeNotFound / NetworkXNoPath)

Nenhuma variável global — todos os parâmetros são explícitos.
"""

import time
import warnings

import networkx as nx

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Heurística euclidiana para A*
# ─────────────────────────────────────────────────────────────────────────────

def _euclidean_heuristic(G):
    """Retorna uma função heurística euclidiana fechada sobre o grafo G."""
    def heuristic(u, v):
        xu, yu = G.nodes[u].get("x", 0), G.nodes[u].get("y", 0)
        xv, yv = G.nodes[v].get("x", 0), G.nodes[v].get("y", 0)
        return ((xu - xv) ** 2 + (yu - yv) ** 2) ** 0.5
    return heuristic


# ─────────────────────────────────────────────────────────────────────────────
# Execução dos algoritmos
# ─────────────────────────────────────────────────────────────────────────────

def run_algorithms(G, source, target):
    """
    Executa Dijkstra, Bellman-Ford e A* de source a target usando weight='weight'.
    Trata graciosamente pares desconectados (retorna None nos campos do resultado).

    Parâmetros
    ──────────
    G      : grafo NetworkX com atributo 'weight' nas arestas
    source : nó de origem
    target : nó de destino

    Retorna
    ───────
    results : dict[str, dict] com chaves "Dijkstra", "Bellman-Ford", "A*".
              Cada valor contém: path, dist, nodes, time
              (path e dist são None se não houver caminho)
    """
    heuristic = _euclidean_heuristic(G)

    algorithms = {
        "Dijkstra":     lambda s, t: nx.dijkstra_path(G, s, t, weight="weight"),
        "Bellman-Ford": lambda s, t: nx.bellman_ford_path(G, s, t, weight="weight"),
        "A*":           lambda s, t: nx.astar_path(G, s, t, heuristic=heuristic, weight="weight"),
    }

    results = {}
    for name, fn in algorithms.items():
        t0 = time.time()
        try:
            path = fn(source, target)
            dist = nx.path_weight(G, path, weight="weight")
            results[name] = {
                "path": path,
                "dist": dist,
                "nodes": len(path),
                "time": time.time() - t0,
            }
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            results[name] = {"path": None, "dist": None, "nodes": None, "time": time.time() - t0}

    print("\n=== Algorithm Performance ===")
    for algo, data in results.items():
        d = f"{data['dist']:.2f} m" if data["dist"] is not None else "N/A"
        n = str(data["nodes"]) if data["nodes"] is not None else "N/A"
        print(f"  {algo:15s}  time={data['time']:.4f}s  nodes={n}  dist={d}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Comparação de rotas (Q4)
# ─────────────────────────────────────────────────────────────────────────────

def compare_routes(G, source, target, risk_nodes: set, penalty_factor: float = 2.0):
    """
    Q4 — Compara a rota original (weight = length) com a rota penalizada
    (weight = length × penalty_factor para arestas adjacentes a nós de risco).

    O grafo é restaurado para weight = length ao final da função.

    Parâmetros
    ──────────
    G              : grafo NetworkX
    source         : nó de origem
    target         : nó de destino
    risk_nodes     : conjunto de nós de risco
    penalty_factor : fator de penalização (padrão 2.0)

    Retorna
    ───────
    cmp : dict com chaves:
          route_orig, dist_orig, risk_orig,
          route_pen,  dist_pen,  risk_pen,
          same_route
    """
    from flood import apply_flood_penalty, reset_weights

    print("\n=== Q4 — Route Comparison: Original vs Flood-Penalized ===")

    # ── Rota original ─────────────────────────────────────────────────────────
    reset_weights(G)
    route_orig = nx.dijkstra_path(G, source, target, weight="weight")
    dist_orig  = nx.path_weight(G, route_orig, weight="weight")
    risk_orig  = [n for n in route_orig if n in risk_nodes]

    # ── Rota penalizada ───────────────────────────────────────────────────────
    apply_flood_penalty(G, risk_nodes, penalty_factor)
    route_pen = nx.dijkstra_path(G, source, target, weight="weight")

    # Distância real da rota penalizada (recalcula com length)
    reset_weights(G)
    dist_pen = nx.path_weight(G, route_pen, weight="weight")
    risk_pen = [n for n in route_pen if n in risk_nodes]

    same_route = route_orig == route_pen

    # ── Relatório ─────────────────────────────────────────────────────────────
    print(f"\n  {'Metric':<30} {'Original':>12} {'Penalized':>12}")
    print(f"  {'─'*30} {'─'*12} {'─'*12}")
    print(f"  {'Nodes in path':<30} {len(route_orig):>12} {len(route_pen):>12}")
    print(f"  {'Real distance (m)':<30} {dist_orig:>12.2f} {dist_pen:>12.2f}")
    print(f"  {'Risk nodes crossed':<30} {len(risk_orig):>12} {len(risk_pen):>12}")
    print(f"  {'Route changed?':<30} {'—':>12} {str(not same_route):>12}")
    print(f"  {'Δ distance (m)':<30} {'—':>12} {dist_pen - dist_orig:>+12.2f}")

    if same_route:
        print(
            "\n  ► Rota inalterada: a rede não oferece alternativa viável — "
            "gargalo topológico."
        )
    else:
        delta_risk = len(risk_orig) - len(risk_pen)
        delta_dist = dist_pen - dist_orig
        print(
            f"\n  ► Rota alterada: desviou {delta_risk} nó(s) de risco, "
            f"acrescentando {delta_dist:+.1f} m."
        )

    return {
        "route_orig": route_orig, "dist_orig": dist_orig, "risk_orig": risk_orig,
        "route_pen":  route_pen,  "dist_pen":  dist_pen,  "risk_pen":  risk_pen,
        "same_route": same_route,
    }
