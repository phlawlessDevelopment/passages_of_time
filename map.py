import random


class Graph:
    def __init__(self):
        # Initialize the adjacency list
        self.adj = {}

    def add_vertex(self, vertex):
        # Add a vertex to the graph
        if vertex not in self.adj:
            self.adj[vertex] = []

    def add_edge(self, vertex1, vertex2):
        val = random.choice([-1,1])
        # Add an edge to the graph (directed graph)
        if vertex1 in self.adj:
            self.adj[vertex1].append((vertex2, val ))
        else:
            self.adj[vertex1] = [(vertex2, val)]

    def remove_vertex(self, vertex):
        # Remove a vertex from the graph
        if vertex in self.adj:
            del self.adj[vertex]
        for vertices in [a[1] for a in self.adj.values()]:
            if vertex in vertices:
                vertices.remove(vertex)

    def remove_edge(self, vertex1, vertex2):
        # Remove an edge from the graph
        if vertex1 in self.adj and vertex2 in self.adj[vertex1]:
            self.adj[vertex1].remove(vertex2)

    def get_vertices(self):
        # Return all vertices in the graph
        return list(self.adj.keys())

    def get_edges(self):
        # Return all edges in the graph
        edges = []
        for vertex, neighbors in self.adj.items():
            for neighbor in neighbors:
                edges.append((vertex, neighbor))
        return edges

    def __str__(self):
        # String representation of the graph
        return str(self.adj)

class Node:
    def __init__(self, room, ages):
        self.room = room
        self.ages = ages
        self.next = None

    def __str__(self) -> str:
        return f"{self.room}{"".join(self.ages)}"

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, room, ages):
        new_node = Node(room, ages)
        if self.head is None:
            self.head = new_node
            return
        last_node = self.head
        while last_node.next:
            last_node = last_node.next
        last_node.next = new_node


    def delete_with_value(self, room, ages):
        if self.head is None:
            return
        if self.head.room == room and self.head.ages == ages:
            self.head = self.head.next
            return
        current_node = self.head
        while current_node.next and current_node.next.room != room and current_node.next.ages != ages:
            current_node = current_node.next
        if current_node.next:
            current_node.next = current_node.next.next

    def print_list(self):
        current_node = self.head
        while current_node:
            print(str(current_node), end=" -> " if current_node.next else "")
            current_node = current_node.next

      
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

def make_path(map_graph:Graph) -> LinkedList:
    path = LinkedList()
    
        
    return path

map = make_map_graph()
_2d = make_path(map)

