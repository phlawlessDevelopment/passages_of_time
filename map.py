import psutil
from collections import deque 
import random
import networkx as nx
import plotly.graph_objects as go
import json
import  re

class Graph:
    def __init__(self):
        self.adj = {}

    def add_vertex(self, vertex):
        if vertex not in self.adj:
            self.adj[vertex] = []

    def add_edge(self, vertex1, vertex2, val=0):
        if val == 0:
            val = random.choice([-1,1])
        if vertex1 in self.adj:
            if not ((vertex2, val) in self.adj[vertex1]):
                self.adj[vertex1].append((vertex2, val))
        else:
            self.adj[vertex1] = [(vertex2, val)]
        if vertex2 in self.adj:
            if not ((vertex1, val *-1) in self.adj[vertex2]):
                self.adj[vertex2].append((vertex1, val *-1))
        else:
            self.adj[vertex2] = [(vertex1, val * -1)]


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
            visited.add(vertex)
            path.append(vertex)
            
            if not (vertex not in self.adj or len(self.adj[vertex]) == 0):
                print("".join(vertex) + " -> [" + ", ".join([str(x) + '(' + str(y) + ')' for x, y in self.adj[vertex]]) + "]")
                for neighbor in self.adj[vertex]:
                    dfs(neighbor, path.copy(), visited.copy())
            

        for vertex in self.adj:
            dfs(vertex, [], set())

      
def make_map_graph(input_graph = None, dimensions = (2,2)) -> Graph:
    graph = Graph()

    if input_graph:
        for k,v in input_graph.items():
            graph.add_vertex(k)
            for e in v:
                graph.add_edge(k,e[0], e[1])
        return graph

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
    new_state[char_index-1] = new_room
    new_state[char_index] = str(new_age)
    return ''.join(new_state)

def create_game_state_graph(map, character, initial_state):
    game_state_graph = Graph()
    queue = deque([initial_state])
    visited = set([initial_state])
    
    game_state_graph.add_vertex(initial_state)
    
    while queue:
        current_state = queue.popleft()
    
        thief, knight, wizard = re.findall(r'([A-Za-z]+[1-3])', current_state)
        current_room = ""
        char_index = 1

        if character == "T":
            current_room = thief[:-1]
        if character == "K":
            current_room = knight[:-1]
            char_index += len(current_room) + 1
        if character == "W":
            current_room = wizard[:-1]
            char_index += (len(current_room) + 1) * 2
        
        current_age = int(current_state[char_index])
        
        for neighbor, weight in map.adj[current_room]:
            new_age = current_age + weight
            
            if 1 <= new_age <= 3:
                new_state = generate_new_state(current_state, char_index, neighbor, new_age)
                
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


def create_plot(state_graph):

    G = create_networkx_graph(state_graph)
    grid_size = int(len(G.nodes)**(1/2)) + 1

    initial_positions = {}
    node_list = list(G.nodes)
    node_index = 0
    
    for x in range(grid_size):
        for y in range(grid_size):
            if node_index < len(node_list):
                initial_positions[node_list[node_index]] = (x, y)
                node_index += 1
            else:
                break
        if node_index >= len(node_list):
            break

    pos = nx.fruchterman_reingold_layout(G,pos=initial_positions, dim=2)
    #pos = nx.spring_layout(G,pos=initial_positions, dim=2)
    #pos = nx.kamada_kawai_layout(G,pos=initial_positions,dim=2)

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=2, color='#888'), hoverinfo='none',
        mode='lines+markers')
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)
    
    node_trace = go.Scatter(
        x=[],
        y=[],
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
        x, y  = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['text'] += (node,)
        node_trace['marker']['color'] += ('skyblue',)

    return go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="",
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        scene=dict(
                            xaxis=dict(showbackground=False),
                            yaxis=dict(showbackground=False),
                            ),
                        ))


def show_graphs(map_graph):
    thief_state_graph = create_game_state_graph(map_graph, "T", "A1A2A3")
    knight_state_graph = create_game_state_graph(map_graph, "K", "A1A2A3")
    wizard_state_graph = create_game_state_graph(map_graph, "W", "A1A2A3")
    
    #map_plot = create_plot(map)
    thief_plot = create_plot(thief_state_graph)
    knight_plot = create_plot(knight_state_graph)
    wizard_plot = create_plot(wizard_state_graph)


    print("-----------------------------")
    map_graph.print_path()
    print("-----------------------------")
    
    #map_plot.show()
    thief_plot.show()
    knight_plot.show()
    wizard_plot.show()
    print(f"used virtual memmory percentage : {psutil.virtual_memory().percent}")



print(f"used virtual memmory percentage : {psutil.virtual_memory().percent}")

for file in ["2x2.json","2x3.json"]:
    with open(file) as f:
        map_graph = make_map_graph(json.load(f))
        show_graphs(map_graph)

