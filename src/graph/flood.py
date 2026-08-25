"""
flood.py

Lógica de risco de inundação aplicada ao grafo viário.
"""

import warnings
import networkx as nx
import osmnx as ox
import srtm
from pyproj import Transformer
from shapely.geometry import Point

warnings.filterwarnings("ignore")


def fetch_river_geometry(place_name, graph_crs, river_name="Velhas"):
    """Obtém a geometria do rio via OSMnx e reprojeta para o CRS do grafo."""
    tags = {"waterway": "river"}
    rivers = ox.features_from_place(place_name, tags)
    if rivers.empty:
        raise ValueError(f"Nenhum curso d'água encontrado em '{place_name}'.")

    mask = rivers["name"].str.contains(river_name, case=False, na=False)
    rivers = rivers[mask] if mask.any() else rivers
    rivers = rivers.to_crs(graph_crs)
    return rivers, rivers.geometry.unary_union


def identify_flood_risk_nodes(graph, place_name, threshold=150.0, river_name="Velhas"):
    """Classifica nós cuja distância ao rio é ≤ threshold metros."""
    graph_crs = graph.graph.get("crs", "EPSG:32723")
    rivers, river_geom = fetch_river_geometry(place_name, graph_crs, river_name)
    transformer = Transformer.from_crs(graph_crs, "EPSG:4326", always_xy=True)

    risk_nodes = set()
    risk_info = []

    for node in graph.nodes():
        point = Point(graph.nodes[node]["x"], graph.nodes[node]["y"])
        distance = point.distance(river_geom)
        if distance <= threshold:
            lon, lat = transformer.transform(graph.nodes[node]["x"], graph.nodes[node]["y"])
            risk_nodes.add(node)
            risk_info.append({
                "node": node,
                "dist_m": distance,
                "lat": lat,
                "lon": lon,
                "x": graph.nodes[node]["x"],
                "y": graph.nodes[node]["y"],
            })

    print(f"Nós de risco encontrados: {len(risk_nodes)}")
    return risk_nodes, risk_info, rivers


def find_nearest_river_nodes(graph, place_name, threshold=50.0, river_name="Velhas"):
    """Retorna todos os nós dentro de threshold metros do rio, ordenados por distância."""
    graph_crs = graph.graph.get("crs", "EPSG:32723")
    _, river_geom = fetch_river_geometry(place_name, graph_crs, river_name)
    transformer = Transformer.from_crs(graph_crs, "EPSG:4326", always_xy=True)

    nearby = []
    for node in graph.nodes():
        point = Point(graph.nodes[node]["x"], graph.nodes[node]["y"])
        distance = point.distance(river_geom)
        if distance <= threshold:
            lon, lat = transformer.transform(graph.nodes[node]["x"], graph.nodes[node]["y"])
            nearby.append({
                "node": node,
                "dist_m": distance,
                "lat": lat,
                "lon": lon,
                "x": graph.nodes[node]["x"],
                "y": graph.nodes[node]["y"],
                "degree": graph.degree(node)
            })

    nearby.sort(key=lambda n: n["dist_m"])
    return nearby


def apply_flood_penalty(graph, risk_nodes, penalty_factor=2.0):
    """Multiplica o peso de arestas adjacentes a nós de risco por penalty_factor."""
    penalized = 0
    for u, v, edge_data in graph.edges(data=True):
        length = edge_data.get("length", 1.0)
        if u in risk_nodes or v in risk_nodes:
            edge_data["weight"] = length * penalty_factor
            penalized += 1
        else:
            edge_data["weight"] = length
    return graph, penalized


def reset_weights(graph):
    """Restaura weight = length em todas as arestas."""
    for _, _, edge_data in graph.edges(data=True):
        edge_data["weight"] = edge_data.get("length", 1.0)
    return graph


def make_blocked_graph(graph, risk_nodes):
    """Remove nós de risco e mantém apenas o maior componente conectado."""
    blocked = graph.copy()
    for node in risk_nodes:
        if node in blocked.nodes():
            blocked.remove_node(node)

    if not nx.is_connected(blocked):
        largest = max(nx.connected_components(blocked), key=len)
        blocked = blocked.subgraph(largest).copy()

    print(f"Grafo bloqueado — Nós: {len(blocked.nodes())}  |  Arestas: {len(blocked.edges())}")
    return blocked


def compute_river_distance(graph, place_name):
    """Calcula distância ao rio e elevação SRTM para cada nó."""
    graph_crs = graph.graph.get("crs", "EPSG:32723")
    _, river_geom = fetch_river_geometry(place_name, graph_crs)
    transformer = Transformer.from_crs(graph_crs, "EPSG:4326", always_xy=True)
    elevation_data = srtm.get_data()

    result = {}
    for node in graph.nodes():
        point = Point(graph.nodes[node]["x"], graph.nodes[node]["y"])
        dist = point.distance(river_geom)
        lon, lat = transformer.transform(graph.nodes[node]["x"], graph.nodes[node]["y"])
        elev = elevation_data.get_elevation(lat, lon) or 0.0
        result[node] = {"node": node, "dist_m": dist, "elevation": elev}

    return result