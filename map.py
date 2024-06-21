from pprint import pprint
import random


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
            for r in v:
                row_code = self.tuple_to_str(r[0])
                print(code +  " -> " + str((row_code, r[1])))
        print()

      
def make_map_graph() -> Graph:
    graph = Graph()
    grid = []
    letters = [chr(i) for i in range(ord('A'), ord('Z'))]
    letters.reverse()
    dimensions = (2,2)
    
    for x in range(dimensions[0]):
        grid.append([])
        for y in range(dimensions[1]):
            l = letters.pop()
            grid[x].append(l)
            graph.add_vertex(l)
    
    
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

def make_path(map:Graph) -> Graph:
    path = Graph()

    visited = set()
    stack = ["A"]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            path.add_vertex((vertex, 0, vertex, 0, vertex, 0))
            for neighbor in reversed(map.adj.get(vertex, [])):
                if neighbor and neighbor not in visited:
                    stack.append(neighbor)
                    
                    print(neighbor)
                    from_room, t_age, k_age, w_age = neighbor[0], neighbor[1], neighbor[3], neighbor[5]
                    for to_room, age_change in map.adj[from_room]:
                        char_ages = [0, 0, 0]
                        if 0 < t_age + age_change < 4:
                            char_ages[0] = t_age + age_change
                        if 0 < k_age + age_change < 4:
                            char_ages[1] = k_age + age_change
                        if 0 < w_age + age_change < 4:
                            char_ages[2] = w_age + age_change
                        if 1 in char_ages or 2 in char_ages or 3 in char_ages:
                            new_v = tuple(sum(([to_room, age] for age in char_ages), []))
                            path.add_vertex(neighbor)
                            path.add_vertex(new_v)
                            path.add_edge(neighbor, new_v)
           

    return path

map = make_map_graph()
map.print_graph()
path = make_path(map)
path.print_graph()
