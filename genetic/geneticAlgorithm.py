import numpy as np
import random

class chromosome():
    def __init__(self, x, y, bird, rank, pocket):
        self.x = x
        self.y = y
        self.bird = bird
        self.rank = rank
        self.pocket = pocket

# Função Bird
def bird_function(x, y):
    return np.sin(x) * np.exp((1 - np.cos(y))**2) + np.cos(y) * np.exp((1 - np.sin(x))**2) + (x - y)**2

def linear_value(index):
    min = 0
    max = 100
    linearResult=(min+ (max-min) * ((100-index)/(100-1)))
    return linearResult

def linear_function(list):
    linearResult = []
    for i in range(len(list)):
        min = 0
        max = 100
        linearResult.append(min+ (max-min) * ((list[i]-i)/(list[i]-1)))
    return list

def crossover(chromossome1, chromossome2):
    r1 = random.random() # Random entre 0 e 1
    newX1 = (r1*chromossome1.x + (1-r1)*chromossome2.x)
    newY1 = (r1*chromossome1.y + (1-r1)*chromossome2.y)
    newBird1 = bird_function(newX1, newY1)
    newChromosome1 = chromosome(newX1, newY1, newBird1, None, None)

    r2 = 1-r1
    newX2 = (r2*chromossome1.x + (1-r2)*chromossome2.x)
    newY2 = (r2*chromossome1.y + (1-r2)*chromossome2.y)
    newBird2 = bird_function(newX2, newY2)
    newChromosome2 = chromosome(newX2, newY2, newBird2, None, None)
    descendants = (newChromosome1, newChromosome2)

    return descendants



    
# Parâmetros do algoritmo genético
pop_size = 100  # Tamanho da população
num_genes = 2   # Número de genes em cada cromossomo
num_generations = 100  # Número de gerações
mutation_rate = 0.1  # Taxa de mutação

# Inicialização da população
population = np.random.uniform(low=-10, high=10, size=(pop_size, num_genes))
lineage = []
for i, n in enumerate(population):    
    x = n[0]
    y = n[1]
    current_bird = bird_function(x,y)
    new_chromosome = chromosome(x,y, current_bird, None, None)
    lineage.append(new_chromosome)

# 100 gerações
for i in range(0,num_generations):
    # Sort pela função objetiva
    sorted_lineage = sorted(lineage, key=lambda x: x.bird)

    # Somas do ranks
    pocket_total = 0
    for i, n in enumerate(sorted_lineage):
        n.rank = linear_value(i)
        pocket_total+= n.rank
        n.pocket = pocket_total

    # Selecionar 100 cromossomos através da roleta
    parents = []
    for i in range(0,100):
        r1 = random.randint(1,5101)
        for n in sorted_lineage:
            if n.pocket >= r1:
                parents.append(n)



    # Selecionar os 100 pais a partir dos 100 selecionados e realizar o crossover
    lineage.clear()
    for i in range(0,100):
        r1 = random.randint(1,100)
        parent1 = parents[r1]
        r1 = random.randint(1,100)
        parent2 = parents[r1]
        childs = crossover(parent1, parent2)
        lineage.append(childs[0])
        lineage.append(childs[1])

