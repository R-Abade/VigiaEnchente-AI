"""
analysis_blocked.py
───────────────────
Script orquestrador — análise do grafo bloqueado (remoção dos nós de risco).

Etapas
  1. Carrega e pré-processa o grafo de Sabará
  2. Identifica os nós de risco (≤ threshold m do Rio das Velhas)
  3. Calcula a rota de referência no grafo original (par src → tgt)
  4. Constrói o grafo bloqueado (sem nós de risco)
  5. Verifica a conectividade do par original no grafo bloqueado
     • Se conectado : roda os três algoritmos no mesmo par
     • Se desconectado : encontra novo diâmetro via Floyd-Warshall
  6. Gera figuras comparativas salvas em assets/images/
  7. Exibe relatório consolidado no console

Importa exclusivamente dos módulos locais; não duplica lógica.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import networkx as nx

from loader import build_graph
from flood import (
    identify_flood_risk_nodes,
    find_nearest_river_nodes,
    make_blocked_graph,
)
from routing import run_algorithms
from metrics import compute_fw_matrix, compute_diameter, find_isolated_nodes
from visualization import (
    plot_blocked_comparison,
    plot_isolated_nodes,
    plot_nearest_river_nodes,
)


# ─────────────────────────────────────────────────────────────────────────────
# Relatório consolidado
# ─────────────────────────────────────────────────────────────────────────────

def _print_report(
    results_orig: dict,
    results_blocked: dict,
    disconnected: bool,
    src_orig, tgt_orig,
    src_alt, tgt_alt,
    iso_nodes: list,
    risk_nodes: set,
    src_in: bool,
    tgt_in: bool,
    top_iso: int,
):
    SEP = "═" * 72

    print(f"\n{SEP}")
    print(f"  RESULTADOS — GRAFO ORIGINAL (par {src_orig}→{tgt_orig})")
    print("─" * 72)
    print(f"  {'Algoritmo':<15} {'Distância (m)':>14} {'Nós':>6} {'Tempo (s)':>10}")
    print(f"  {'─'*15} {'─'*14} {'─'*6} {'─'*10}")
    for algo, r in results_orig.items():
        d = f"{r['dist']:.2f}" if r["dist"] is not None else "N/A"
        n = str(r["nodes"])    if r["nodes"] is not None else "N/A"
        print(f"  {algo:<15} {d:>14} {n:>6} {r['time']:>10.5f}")

    label = f"par original" if not disconnected else f"novo par {src_alt}→{tgt_alt}"
    print(f"\n  RESULTADOS — GRAFO BLOQUEADO ({label})")
    print("─" * 72)
    print(f"  {'Algoritmo':<15} {'Distância (m)':>14} {'Nós':>6} {'Tempo (s)':>10}")
    print(f"  {'─'*15} {'─'*14} {'─'*6} {'─'*10}")
    for algo, r in results_blocked.items():
        d = f"{r['dist']:.2f}" if r["dist"] is not None else "N/A"
        n = str(r["nodes"])    if r["nodes"] is not None else "N/A"
        print(f"  {algo:<15} {d:>14} {n:>6} {r['time']:>10.5f}")

    print()
    if disconnected:
        print(f"  ► Par original {src_orig}→{tgt_orig} DESCONECTADO no grafo bloqueado.")
        print(f"    Nó {src_orig} presente: {src_in}  |  Nó {tgt_orig} presente: {tgt_in}")
        print(f"    Novo par de diâmetro : {src_alt}→{tgt_alt}")
    else:
        d_o = results_orig["Dijkstra"]["dist"]
        d_b = results_blocked["Dijkstra"]["dist"]
        n_o = results_orig["Dijkstra"]["nodes"]
        n_b = results_blocked["Dijkstra"]["nodes"]
        if d_o and d_b:
            print(f"  ► Par original ainda conectado no grafo bloqueado.")
            print(f"    Δ distância real : {d_b - d_o:+.2f} m  ({(d_b / d_o - 1) * 100:+.2f} %)")
            print(f"    Δ nós na rota    : {n_b - n_o:+d}")

    if iso_nodes:
        print(f"\n  TOP {top_iso} NÓS MAIS ISOLADOS (grafo original, Floyd-Warshall)")
        print("─" * 72)
        print(f"  {'Rank':>4}  {'Nó':>8}  {'Dist. média (km)':>18}")
        for rank, (node, d) in enumerate(iso_nodes, 1):
            print(f"  {rank:>4}  {node:>8}  {d / 1000:>18.3f}")

    print(SEP)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(
    place: str = "Sabará, Minas Gerais, Brazil",
    threshold: float = 50.0,
    source: int = 1602,
    target: int = 927,
    top_iso: int = 20,
):
    """
    Parâmetros
    ──────────
    place     : nome do lugar (Nominatim)
    threshold : distância de risco ao rio em metros
    source    : nó de origem do par de referência
    target    : nó de destino do par de referência
    top_iso   : quantos nós isolados destacar
    """

    # ── 1. Grafo original ──────────────────────────────────────────────────────
    G = build_graph(place)

    # ── 2. Nós de risco ────────────────────────────────────────────────────────
    risk_nodes, risk_info, rivers = identify_flood_risk_nodes(G, place, threshold)

    # ── 3. Nós mais próximos do rio (relatório + mapa) ─────────────────────────
    nearby = find_nearest_river_nodes(G, place, threshold=threshold)
    plot_nearest_river_nodes(
        G, nearby, rivers,
        threshold=threshold,
        top_n=top_iso,
        filename="nearest_river_nodes.png",
    )

    # ── 4. Rota de referência no grafo original ────────────────────────────────
    print(f"\nCalculando rota original {source}→{target}...")
    results_orig = run_algorithms(G, source, target)
    route_orig   = results_orig["Dijkstra"]["path"]

    # ── 5. Grafo bloqueado ────────────────────────────────────────────────────
    G_blocked = make_blocked_graph(G, risk_nodes)

    src_in = source in G_blocked.nodes()
    tgt_in = target in G_blocked.nodes()
    pair_connected = (
        src_in and tgt_in
        and nx.has_path(G_blocked, source, target)
    )

    print(f"\n  Origem  {source} no grafo bloqueado : {src_in}")
    print(f"  Destino {target} no grafo bloqueado : {tgt_in}")
    print(f"  Par conectado                      : {pair_connected}")

    disconnected = not pair_connected

    # ── 6. Algoritmos no grafo bloqueado ──────────────────────────────────────
    if pair_connected:
        print(f"\nRodando algoritmos no grafo bloqueado (par original)...")
        results_blocked = run_algorithms(G_blocked, source, target)
        src_alt, tgt_alt = source, target
        route_alt = results_blocked["Dijkstra"]["path"]
    else:
        print(f"\nPar original desconectado — buscando novo diâmetro...")
        matrix_b, nodes_b = compute_fw_matrix(G_blocked)
        src_alt, tgt_alt, max_d_b, _, _ = compute_diameter(
            G_blocked, matrix=matrix_b, nodes=nodes_b
        )
        print(f"  Novo par: {src_alt}→{tgt_alt}  ({max_d_b / 1000:.2f} km)")
        results_blocked = run_algorithms(G_blocked, src_alt, tgt_alt)
        route_alt = results_blocked["Dijkstra"]["path"]

    # ── 7. Floyd-Warshall no grafo original (nós isolados) ────────────────────
    matrix_orig, nodes_orig = compute_fw_matrix(G)
    iso_nodes, _ = find_isolated_nodes(matrix_orig, nodes_orig, top_n=top_iso)

    # ── 8. Relatório ──────────────────────────────────────────────────────────
    _print_report(
        results_orig, results_blocked, disconnected,
        source, target, src_alt, tgt_alt,
        iso_nodes, risk_nodes,
        src_in, tgt_in, top_iso,
    )

    # ── 9. Figuras ────────────────────────────────────────────────────────────
    print("\nGerando figura comparativa (grafo original vs bloqueado)...")
    plot_blocked_comparison(
        G, G_blocked,
        risk_nodes, risk_info, rivers,
        route_orig, route_alt,
        source, target,
        src_alt, tgt_alt,
        disconnected,
        results_orig, results_blocked,
        threshold=threshold,
        filename="grafo_bloqueado.png",
    )

    print("Gerando figura dos nós mais isolados...")
    plot_isolated_nodes(
        G, matrix_orig, nodes_orig, iso_nodes,
        rivers=rivers,
        filename="nos_isolados.png",
    )

    print("\nConcluído.")


if __name__ == "__main__":
    main()
