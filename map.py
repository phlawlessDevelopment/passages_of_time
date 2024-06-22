from collections import deque, defaultdict
import random
import networkx as nx
from numpy import append
import plotly.graph_objects as go
# Function to create the NetworkX graph from the custom Graph class
class Graph:
    def __init__(self):
        self.adj = {}

    def add_vertex(self, vertex):
        if vertex not in self.adj:
            self.adj[vertex] = []

    def add_edge(self, vertex1, vertex2):
        val = random.choice([-1,1])
        if vertex1 in self.adj:
            self.adj[vertex1].append((vertex2, val ))
        else:
            self.adj[vertex1] = [(vertex2, val)]

    def remove_vertex(self, vertex):
        if vertex in self.adj:
            del self.adj[vertex]
        for vertices in [a[1] for a in self.adj.values()]:
            if vertex in vertices:
                vertices.remove(vertex)

    def remove_edge(self, vertex1, vertex2):
        if vertex1 in self.adj and vertex2 in self.adj[vertex1]:
            self.adj[vertex1].remove(vertex2)

    def get_vertices(self):
        return list(self.adj.keys())

    def get_edges(self):
        edges = []
        for vertex, neighbors in self.adj.items():
            for neighbor in neighbors:
                edges.append((vertex, neighbor))
        return edges

    def tuple_to_str(self, tu):
        return "".join([str(k) for k in tu])
    
    def print_graph(self)->None:
        for k, v in self.adj.items():
            code = self.tuple_to_str(k)
            print(code, end=" ->")
            print()
            for r in v:
                row_code = self.tuple_to_str(r[0])
                print("\t -> " + str((row_code, r[1])))
        print()
    
    def print_path(self):
        def dfs(vertex, path, visited):
            if vertex in visited:
                # Detect a loop and print with fancy ASCII
                print(" -> ".join((path[0], path[1][0])) + " -> " + vertex + " (loop detected)")
                return
            
            visited.add(vertex)
            path.append(vertex)
            
            if vertex not in self.adj or len(self.adj[vertex]) == 0:
                # Print the full path
                print(" -> ".join((path[0], path[1][0])))
            else:
                for neighbor in self.adj[vertex]:
                    dfs(neighbor, path.copy(), visited.copy())
            
        for vertex in self.adj:
            dfs(vertex, [], set())

      
def make_map_graph(dimensions = (2,2)) -> Graph:
    graph = Graph()
    grid = []
    letters_ = [chr(i) for i in range(ord('A'), ord('Z'))]
    letters = []
    for i in range(1, 4):
        letters += [l*i for l in letters_]
    i = 0
    letter_count = 1    
    for x in range(dimensions[0]):
        grid.append([])
        for y in range(dimensions[1]):
            l = (letters[i] * letter_count)
            grid[x].append(l)
            graph.add_vertex(l)
            i = (i + 1) % len(letters)
            if i == 0:
                letter_count +=1
    
    
    for x in range(dimensions[0]):
        for y in range(dimensions[1]):
            room = grid[x][y]
            if x > 0:
                graph.add_edge(room,grid[x-1][y])
            if x < dimensions[0] -1:
                graph.add_edge(room, grid[x+1][y])
            if y < dimensions[1]-1:
                graph.add_edge(room, grid[x][y+1])
            if y > 0:
                graph.add_edge(room, grid[x][y-1])

    return graph


def generate_new_state(state, char_index, new_room, new_age):
    new_state = list(state)
    new_state[char_index * 2] = new_room
    new_state[char_index * 2 + 1] = str(new_age)
    return ''.join(new_state)

def create_game_state_graph_for_first_character(map, initial_state):
    game_state_graph = Graph()
    queue = deque([initial_state])
    visited = set([initial_state])
    
    game_state_graph.add_vertex(initial_state)
    
    while queue:
        current_state = queue.popleft()
        
        current_room = ""
        i = 0
        for c in current_state:
                if c.isdigit(): 
                    break
                current_room += c
                i += 1
        current_age = int(current_state[i])
        
        for neighbor, weight in map.adj[current_room]:
            new_age = current_age + weight
            
            if 1 <= new_age <= 3:
                new_state = generate_new_state(current_state, 0, neighbor, new_age)
                
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append(new_state)
                    game_state_graph.add_vertex(new_state)
                
                game_state_graph.add_edge(current_state, new_state)
    
    return game_state_graph

def create_networkx_graph(game_state_graph):
    G = nx.DiGraph()
    for vertex in game_state_graph.get_vertices():
        G.add_node(vertex)
    for edge in game_state_graph.get_edges():
        G.add_edge(edge[0], edge[1][0])
    return G


map = make_map_graph((4,2))
map.print_graph()
game_state_graph = create_game_state_graph_for_first_character(map, "A1A2A3")
game_state_graph.print_graph()

G = create_networkx_graph(game_state_graph)

pos = nx.spring_layout(G, dim=3)

edge_trace = go.Scatter3d(
    x=[],
    y=[],
    z=[],
    line=dict(width=2, color='#888'),
    hoverinfo='none',
    mode='lines')

for edge in G.edges():
    x0, y0, z0 = pos[edge[0]]
    x1, y1, z1 = pos[edge[1]]
    edge_trace['x'] += (x0, x1, None)
    edge_trace['y'] += (y0, y1, None)
    edge_trace['z'] += (z0, z1, None)

node_trace = go.Scatter3d(
    x=[],
    y=[],
    z=[],
    text=[],
    mode='markers+text',
    textposition='top center',
    hoverinfo='text',
    marker=dict(
        showscale=False,
        color=[],
        size=10,
        line_width=2))

for node in G.nodes():
    x, y, z = pos[node]
    node_trace['x'] += (x,)
    node_trace['y'] += (y,)
    node_trace['z'] += (z,)
    node_trace['text'] += (node,)
    node_trace['marker']['color'] += ('skyblue',)

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='3D Network Graph',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0, l=0, r=0, t=0),
                    scene=dict(
                        xaxis=dict(showbackground=False),
                        yaxis=dict(showbackground=False),
                        zaxis=dict(showbackground=False)),
                    ))

fig.show()

