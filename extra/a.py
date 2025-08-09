import matplotlib.pyplot as plt
import networkx as nx
import queue
import time

# --- BFS with Path Cost and Goal Node ---
def order_bfs(graph, costs, start_node, goal_node):
    visited = set()
    q = queue.Queue()
    q.put((start_node, [start_node], 0))  # node, path, cost

    while not q.empty():
        current, path, total_cost = q.get()
        if current == goal_node:
            return path, total_cost

        if current not in visited:
            visited.add(current)
            for neighbor in graph[current]:
                if neighbor not in visited:
                    edge_cost = costs.get((current, neighbor), 1)
                    q.put((neighbor, path + [neighbor], total_cost + edge_cost))

    return [], float('inf')

# --- DFS with Path Cost and Goal Node ---
def order_dfs(graph, costs, current, goal_node, visited=None, path=None, cost=0):
    if visited is None:
        visited = set()
    if path is None:
        path = [current]

    if current == goal_node:
        return path, cost

    visited.add(current)

    for neighbor in graph[current]:
        if neighbor not in visited:
            edge_cost = costs.get((current, neighbor), 1)
            new_path, new_cost = order_dfs(
                graph, costs, neighbor, goal_node, visited.copy(), path + [neighbor], cost + edge_cost)
            if new_path:
                return new_path, new_cost

    return [], float('inf')

# --- Visualization ---
def visualize(path, title, G, pos, cost):
    plt.ion()
    plt.figure()

    for i, node in enumerate(path, start=1):
        plt.clf()
        plt.title(f"{title} - Step {i}/{len(path)}\nTotal Cost: {cost}")
        colors = ['r' if n == node else 'g' if n in path[:i] else 'lightgray' for n in G.nodes]
        edge_colors = ['red' if (u, v) in list(zip(path, path[1:])) or (v, u) in list(zip(path, path[1:])) else 'gray'
                       for u, v in G.edges]
        nx.draw(G, pos, with_labels=True, node_color=colors, node_size=500, font_weight='bold', edge_color=edge_colors)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.pause(0.5)

    plt.ioff()
    plt.show()
    time.sleep(0.5)

# --- Sample Weighted Graph ---
G = nx.Graph()
edges_with_weights = [
    ('A', 'B', 1),
    ('A', 'C', 4),
    ('B', 'D', 2),
    ('B', 'E', 5),
    ('C', 'F', 3),
    ('E', 'F', 1),
]

G.add_weighted_edges_from(edges_with_weights)
graph = {node: list(G.neighbors(node)) for node in G.nodes}
costs = {(u, v): w for u, v, w in edges_with_weights}
costs.update({(v, u): w for u, v, w in edges_with_weights})  # make undirected

pos = nx.spring_layout(G)

# --- Run Traversals ---
start_node = 'A'
goal_node = 'F'

bfs_path, bfs_cost = order_bfs(graph, costs, start_node, goal_node)
dfs_path, dfs_cost = order_dfs(graph, costs, start_node, goal_node)

# --- Visualize Results ---
visualize(bfs_path, "BFS Traversal", G, pos, bfs_cost)
visualize(dfs_path, "DFS Traversal", G, pos, dfs_cost)
