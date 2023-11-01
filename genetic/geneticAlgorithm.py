from math import trunc
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

def crossover(chromossome1, chromossome2, crossover_tax):
    r1 = crossover_tax
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

def mutate(chromossomeInput):
    x = chromossomeInput.x 
    y = chromossomeInput.y    
    x += np.random.uniform(-5,5)
    y += np.random.uniform(-5,5)
    bird = bird_function(x, y)
    new_chromosome = chromosome(x,y,bird,None,None)
    return new_chromosome

    
# Parâmetros do algoritmo genético
pop_size = 100  # Tamanho da população
num_genes = 2   # Número de genes em cada cromossomo
num_generations = 400  # Número de gerações
mutation_rate = 0.1  # Taxa de mutação
crossover_tax = 0.2 # Taxa de crossover
best_chromosome = None

# Inicialização da população
population = np.random.uniform(low=-10, high=10, size=(pop_size, num_genes))
lineage = []
for i, n in enumerate(population):
    x = n[0]
    y = n[1]
    current_bird = bird_function(x,y)
    new_chromosome = chromosome(x,y, current_bird, None, None)
    lineage.append(new_chromosome)
sorted_lineage = sorted(lineage, key=lambda x: x.bird)
best_chromosome = sorted_lineage[0]
print("melhor: ",best_chromosome.bird)


# 100 gerações
for gen in range(0,num_generations):
    # Sort pela função objetiva
    sorted_lineage = sorted(lineage, key=lambda x: x.bird)

    # Rank linear e soma dos ranks
    pocket_total = 0
    for i, n in enumerate(sorted_lineage):
        n.rank = linear_value(i+1)
        pocket_total+= n.rank
        n.pocket = pocket_total  

    #Print dos cromossomos
    if(gen == 0 or gen == num_generations//4 or gen == num_generations//2 or gen == num_generations-1):
        print("Geração: ", gen)
        print(best_chromosome.bird)
        print()        

    # Selecionar 100 cromossomos através da roleta
    parents = []    
    for i in range(0,100):
        r1 = random.randint(int(sorted_lineage[0].pocket),int(sorted_lineage[-1].pocket))
        for n in sorted_lineage:
            if n.pocket >= r1:
                parents.append(n)

    # Selecionar os 100 pais a partir dos 100 selecionados e realizar o crossover
    lineage.clear()
    for i in range(0,50):
        r1 = random.randint(0,99)
        parent1 = parents[r1]
        r1 = random.randint(0,99)
        parent2 = parents[r1]
        childs = crossover(parent1, parent2, crossover_tax)
        lineage.append(childs[0])
        lineage.append(childs[1])

    # Mutação
    for n in lineage:
        rMutation = np.random.uniform(0,100)
        if (rMutation <= mutation_rate):
            mutatedChromosome = mutate(n)
            n = mutatedChromosome
        if(n.bird<best_chromosome.bird):         
            best_chromosome=n


 