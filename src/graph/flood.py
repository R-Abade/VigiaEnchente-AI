"""
flood.py
────────
Flood risk logic applied to the road graph.

Responsibilities
  • Fetch the geometry of the Rio das Velhas (or other watercourse) via OSMnx
  • Classify nodes within a risk radius (meters)
  • Penalize edges adjacent to risk nodes (multiplicative factor)
  • Build a blocked graph (removal of risk nodes)
  • Find nodes closest to the river in ascending order of distance
"""

import warnings
import networkx as nx
import osmnx as ox
import srtm

from pyproj import Transformer
from shapely.geometry import Point

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# River geometry
# ------------------------------------------------------------
def fetch_river_geometry(place_name, graph_crs, river_name="Velhas"):
    """
    Download river geometry via OSMnx and reproject to graph CRS.
    """
    tags = {"waterway": "river"}
    rivers_raw = ox.features_from_place(place_name, tags)

    if rivers_raw.empty:
        raise ValueError(f"No watercourse found in '{place_name}'.")

    # Filter by river name
    name_mask = rivers_raw["name"].str.contains(river_name, case=False, na=False)
    rivers = rivers_raw[name_mask]
    if rivers.empty:
        print(f"  [WARNING] '{river_name}' not found; using all watercourses.")
        rivers = rivers_raw

    rivers = rivers.to_crs(graph_crs)
    return rivers, rivers.geometry.unary_union


# ------------------------------------------------------------
# Identify risk nodes
# ------------------------------------------------------------
def identify_flood_risk_nodes(graph, place_name, threshold=150.0, river_name="Velhas"):
    """
    Classify graph nodes whose distance to the river is ≤ threshold meters.
    """
    print(f"\nIdentifying flood-risk nodes (≤ {threshold} m of Rio das Velhas)...")

    graph_crs = graph.graph.get("crs", "EPSG:32723")
    rivers, river_geom = fetch_river_geometry(place_name, graph_crs, river_name)

    transformer = Transformer.from_crs(graph_crs, "EPSG:4326", always_xy=True)

    risk_nodes = set()
    risk_info = []

    for node in graph.nodes():
        x = graph.nodes[node]["x"]
        y = graph.nodes[node]["y"]
        distance = Point(x, y).distance(river_geom)

        if distance <= threshold:
            lon, lat = transformer.transform(x, y)
            risk_nodes.add(node)
            risk_info.append({
                "node": node,
                "dist_m": distance,
                "lat": lat,
                "lon": lon,
                "x": x,
                "y": y
            })

    print(f"  Risk nodes found: {len(risk_nodes)}")

    for info in risk_info:
        print(f"  Node: {info['node']}, Distance: {info['dist_m']}")

    return risk_nodes, risk_info, rivers


def find_nearest_river_nodes(graph, place_name, threshold=50.0, river_name="Velhas"):
    """
    Return all nodes within ≤ threshold meters of the river,
    sorted from closest to farthest. Includes node degree.
    """
    print(f"\nFinding nodes within {threshold} m of {river_name}...")

    graph_crs = graph.graph.get("crs", "EPSG:32723")
    _, river_geom = fetch_river_geometry(place_name, graph_crs, river_name)
    transformer = Transformer.from_crs(graph_crs, "EPSG:4326", always_xy=True)

    nearby = []

    for node in graph.nodes():
        x = graph.nodes[node]["x"]
        y = graph.nodes[node]["y"]
        distance = Point(x, y).distance(river_geom)

        if distance <= threshold:
            lon, lat = transformer.transform(x, y)
            nearby.append({
                "node": node,
                "dist_m": distance,
                "lat": lat,
                "lon": lon,
                "x": x,
                "y": y,
                "degree": graph.degree(node)
            })

    nearby.sort(key=lambda r: r["dist_m"])
    print(f"  Nodes found (≤ {threshold} m): {len(nearby)}")
    return nearby


# ------------------------------------------------------------
# Edge penalty
# ------------------------------------------------------------
def apply_flood_penalty(graph, risk_nodes, penalty_factor=2.0):
    """
    Multiply the weight of edges adjacent to risk nodes by penalty_factor.
    Reset weight = length for others. In-place operation.
    """
    penalized = 0

    for u, v, edge_data in graph.edges(data=True):
        length = edge_data.get("length", 1.0)

        if (u in risk_nodes) or (v in risk_nodes):
            edge_data["weight"] = length * penalty_factor
            penalized = penalized + 1
        else:
            edge_data["weight"] = length

    print(f"  Edges penalized: {penalized} / {graph.number_of_edges()}")
    return graph, penalized


def reset_weights(graph):
    """Restore weight = length on all edges (undo penalty)."""
    for u, v, edge_data in graph.edges(data=True):
        edge_data["weight"] = edge_data.get("length", 1.0)
    return graph


# ------------------------------------------------------------
# Blocked graph (remove risk nodes)
# ------------------------------------------------------------
def make_blocked_graph(graph, risk_nodes):
    """
    Remove risk nodes from the graph and keep only the largest connected component.
    """
    blocked = graph.copy()
    # Remove all risk nodes that exist in the graph
    for node in list(risk_nodes):

        if node in blocked.nodes():
            blocked.remove_node(node)

    if not nx.is_connected(blocked):
        components = list(nx.connected_components(blocked))
        largest = max(components, key=len)
        blocked = blocked.subgraph(largest).copy()

    print(
        f"\nBlocked graph — "
        f"Nodes: {len(blocked.nodes())}  |  Edges: {len(blocked.edges())}"
    )
    return blocked


# ------------------------------------------------------------
# River distance + elevation
# ------------------------------------------------------------
def compute_river_distance(graph, place_name):
    graph_crs = graph.graph.get("crs", "EPSG:32723")
    _, river_geom = fetch_river_geometry(place_name, graph_crs)
    transformer = Transformer.from_crs(graph_crs, "EPSG:4326", always_xy=True)

    elevation_data = srtm.get_data()
    result = {}

    for node in graph.nodes():
        x = graph.nodes[node]["x"]
        y = graph.nodes[node]["y"]
        dist = Point(x, y).distance(river_geom)
        lon, lat = transformer.transform(x, y)

        try:
            elevation = elevation_data.get_elevation(lat, lon)
            if elevation is None:
                elevation = 0.0

        except Exception:
            elevation = 0.0

        result[node] = {"node": node, "dist_m": dist, "elevation": elevation}

    return result