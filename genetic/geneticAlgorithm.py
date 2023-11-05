from math import trunc
from matplotlib import pyplot as plt
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
    new_X_1 = (r1*chromossome1.x + (1-r1)*chromossome2.x)
    new_Y_1 = (r1*chromossome1.y + (1-r1)*chromossome2.y)
    new_bird_1 = bird_function(new_X_1, new_Y_1)
    new_chromosome_1 = chromosome(new_X_1, new_Y_1, new_bird_1, None, None)

    r2 = 1-r1
    new_X_2 = (r2*chromossome1.x + (1-r2)*chromossome2.x)
    new_Y_2 = (r2*chromossome1.y + (1-r2)*chromossome2.y)
    new_bird_2 = bird_function(new_X_2, new_Y_2)
    new_chromosome_2 = chromosome(new_X_2, new_Y_2, new_bird_2, None, None)
    descendants = (new_chromosome_1, new_chromosome_2)

    return descendants

def mutate(chromossomeInput):
    x = chromossomeInput.x 
    y = chromossomeInput.y    
    x += np.random.uniform(-20,20)
    y += np.random.uniform(-20,20)
    bird = bird_function(x, y)
    new_chromosome = chromosome(x,y,bird,None,None)
    return new_chromosome

    
# Parâmetros do algoritmo genético
pop_size = 100  # Tamanho da população
num_genes = 2   # Número de genes em cada cromossomo
num_generations = 100  # Número de gerações
mutation_rate = 0.1  # Taxa de mutação
crossover_tax = 0.2 # Taxa de crossover
best_chromosome = None
best_values=[]
average_values=[]
worst_values=[]

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
    if best_chromosome.bird > sorted_lineage[0].bird:
        sorted_lineage[0] = best_chromosome


    # Rank linear e soma dos ranks
    pocket_total = 0
    for i, n in enumerate(sorted_lineage):
        n.rank = linear_value(i+1)
        pocket_total+= n.rank
        n.pocket = pocket_total  
    
    #Print dos cromossomos
    if(gen == 0 or gen == num_generations//4 or gen == num_generations//2 or gen == num_generations-1):
        print("Geração: ", gen)
        for n in sorted_lineage:
            print(n.bird, n.pocket)
        print(best_chromosome.bird)
        print()        

    # Selecionar 100 cromossomos através da roleta
    parents = []    
    for i in range(0,100):
        r1 = random.randint(0,int(sorted_lineage[-1].pocket))
        for n in sorted_lineage:
            if n.pocket >= r1:
                parents.append(n)
                break            
    
    # Calculate the average and worst values for the current generation
    generation_birds = [n.bird for n in sorted_lineage]
    average_bird = sum(generation_birds) / len(generation_birds)
    worst_bird = sorted_lineage[-1].bird

    # Update the best, average, and worst values lists
    best_values.append(best_chromosome.bird)
    average_values.append(average_bird)
    worst_values.append(worst_bird)

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
            print("mutação ocorreu")
            mutatedChromosome = mutate(n)
            n = mutatedChromosome
        if(n.bird<best_chromosome.bird):         
            best_chromosome=n

# Create a graphic showing the best, average, and worst values for each generation
generations = list(range(num_generations))
plt.plot(generations, best_values, label='Best Chromosome')
plt.plot(generations, average_values, label='Average Chromosome')
plt.plot(generations, worst_values, label='Worst Chromosome')
plt.xlabel('Generation')
plt.ylabel('Bird Function Value')
plt.legend()
plt.title('Genetic Algorithm Performance')
plt.show()
 