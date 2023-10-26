import copy

start_matrix = [[7, 2, 4], 
               [5, 0, 6], 
               [8, 3, 1]]

final_matrix = [[0, 1, 2], 
               [3, 4, 5], 
               [6, 7, 8]]

goal = {
    1: (0, 1),
    2: (0, 2),
    3: (1, 0),
    4: (1, 1),
    5: (1, 2),
    6: (2, 0),
    7: (2, 1),
    8: (2, 2)
}

def heuristic(matrix, goal):
    total = 0
    for i in range(3):
        for j in range(3):
            value = matrix[i][j]
            if goal.get(value):
                pos = goal[value]
                manh = (abs(i - pos[0])) + (abs(j - pos[1]))
                total += manh
    return total

class node():
    def __init__(self, cost=0, heuristic=0, total=0, matrix=None, position=(0, 0)):
        self.cost = cost
        self.heuristic = heuristic
        self.total = total
        self.matrix = matrix
        self.position = position

    def __eq__(self, other):
        return self.matrix == other.matrix

start = node(0, 18, 18, start_matrix, (1, 1))

moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]

def highestTotal(e):
    return (e.total, e.cost)  

def astar(start, goal):
    openNodes = []
    allNodes = []
    current = start
    while heuristic(current.matrix, goal) != 0:
        for move in moves:
            next_position = (current.position[0] + move[0], current.position[1] + move[1])

            if (
                0 <= next_position[0] < 3
                and 0 <= next_position[1] < 3
            ):
                new_matrix = copy.deepcopy(current.matrix)
                aux = new_matrix[current.position[0]][current.position[1]]
                new_matrix[current.position[0]][current.position[1]] = new_matrix[next_position[0]][next_position[1]]
                new_matrix[next_position[0]][next_position[1]] = aux
                manh = heuristic(new_matrix, goal)
                new_node = node(current.cost + 1, manh, current.cost + 1 + manh, new_matrix, next_position)
                if new_node not in allNodes:
                    openNodes.append(new_node)
                    allNodes.append(new_node)
        openNodes.sort(key=highestTotal)
        current = openNodes[0]
        openNodes.pop(0)
        
        print(current.matrix)
        print("Custo atual:",current.cost, "\nHeuristica atual:",current.heuristic, "\nTotal:",current.total)
        print()
    print("Solução encontrada!")
    print("Foi necessário ", current.cost, "movimentações para chegar na solução")
astar(start, goal)