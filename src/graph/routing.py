"""
routing.py

Algoritmos de caminho mínimo e comparação de rotas.
"""

import time
import warnings
import networkx as nx

warnings.filterwarnings("ignore")


def euclidean_heuristic(G):
    def heuristic(u, v):
        xu, yu = G.nodes[u].get("x", 0), G.nodes[u].get("y", 0)
        xv, yv = G.nodes[v].get("x", 0), G.nodes[v].get("y", 0)
        return ((xu - xv) ** 2 + (yu - yv) ** 2) ** 0.5
    return heuristic


def run_algorithms(G, source, target):
    """Executa Dijkstra, Bellman-Ford e A* com heurística euclidiana."""
    heuristic = euclidean_heuristic(G)
    algorithms = {
        "Dijkstra": lambda s, t: nx.dijkstra_path(G, s, t, weight="weight"),
        "Bellman-Ford": lambda s, t: nx.bellman_ford_path(G, s, t, weight="weight"),
        "A*": lambda s, t: nx.astar_path(G, s, t, heuristic=heuristic, weight="weight"),
    }

    results = {}
    for name, fn in algorithms.items():
        t0 = time.time()
        try:
            path = fn(source, target)
            dist = nx.path_weight(G, path, weight="weight")
            results[name] = {"path": path, "dist": dist, "nodes": len(path), "time": time.time() - t0}
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            results[name] = {"path": None, "dist": None, "nodes": None, "time": time.time() - t0}

    print("\n=== Algoritmos ===")
    for algo, data in results.items():
        dist_str = f"{data['dist']:.2f} m" if data["dist"] is not None else "N/A"
        nodes_str = str(data["nodes"]) if data["nodes"] is not None else "N/A"
        print(f"  {algo:12}  tempo={data['time']:.4f}s  nós={nodes_str}  dist={dist_str}")

    return results


def compare_routes(G, source, target, risk_nodes: set, penalty_factor: float = 2.0):
    """Compara rota original com rota penalizada por risco."""
    from flood import apply_flood_penalty, reset_weights

    reset_weights(G)
    route_orig = nx.dijkstra_path(G, source, target, weight="weight")
    dist_orig = nx.path_weight(G, route_orig, weight="weight")
    risk_orig = [n for n in route_orig if n in risk_nodes]

    apply_flood_penalty(G, risk_nodes, penalty_factor)
    route_pen = nx.dijkstra_path(G, source, target, weight="weight")
    reset_weights(G)
    dist_pen = nx.path_weight(G, route_pen, weight="weight")
    risk_pen = [n for n in route_pen if n in risk_nodes]

    same_route = route_orig == route_pen

    print("\n=== Comparação de Rotas ===")
    print(f"  Rota original      : {len(route_orig)} nós, {dist_orig:.2f} m, {len(risk_orig)} nós de risco")
    print(f"  Rota penalizada    : {len(route_pen)} nós, {dist_pen:.2f} m, {len(risk_pen)} nós de risco")
    print(f"  Rotas diferentes   : {not same_route}")
    print(f"  Δ distância        : {dist_pen - dist_orig:+.2f} m")

    return {
        "route_orig": route_orig, "dist_orig": dist_orig, "risk_orig": risk_orig,
        "route_pen": route_pen, "dist_pen": dist_pen, "risk_pen": risk_pen,
        "same_route": same_route,
    }