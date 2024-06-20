import random

class Edge:
    def __init__(self, to_node, val) -> None:
        self.to_node = to_node
        self.val = val

    def __str__(self) -> str:
        return f"{self.val}"
        
class Node:
    def __init__(self, letter) -> None:
        self.letter = letter
        self.ages = {"T" : set(), "K" : set(), "W" : set()}
        self.edges = []

    def add_age(self, character, age )->None:
        self.ages[character].add(age)

    def add_edge(self, node_b)-> None:
        val = random.choice([1,0,-1])
        self.edges.append(Edge(node_b, val))
        node_b.edges.append(Edge(self, val * -1)) 

    def __str__(self) -> str:
        return f"""{self.letter} 
        [T:{"".join([str(age) for age in self.ages["T"]])}] 
        [K:{"".join([str(age) for age in self.ages["K"]])}] 
        [W:{"".join([str(age) for age in self.ages["W"]])}] 
        [{','.join([str(edge) + '->' + edge.to_node.letter for edge in self.edges])}]"""

grid = []
letters = [chr(i) for i in range(ord('A'), ord('Z'))]
letters.reverse()
dimensions = (3,3)

for x in range(dimensions[0]):
    grid.append([])
    for y in range(dimensions[1]):
        grid[x].append(Node(letters.pop()))


for x in range(dimensions[0]):
    for y in range(dimensions[1]):
        if x > 0:
            grid[x][y].add_edge(grid[x-1][y])
        if x < dimensions[0] -1:
            grid[x][y].add_edge(grid[x+1][y])
        if y < dimensions[1]-1:
            grid[x][y].add_edge(grid[x][y+1])
        if y > 0:
            grid[x][y].add_edge(grid[x][y-1])

grid[0][0].add_age("T", 1)
grid[0][0].add_age("K", 1)
grid[0][0].add_age("W", 1)

for x in range(dimensions[0]):
    for y in range(dimensions[1]):
        print(grid[x][y])

