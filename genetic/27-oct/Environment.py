import random, math
from Chromosome import Chromosome

class Environment:

    def __init__(self, population_size, crossover_rate, mutation_rate):
        # Stores an array of chromosomes (samples, individuals)
        self.population = []

        # Environmental rates
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        # Initializes the population
        for i in range(population_size):
            x = random.uniform(-10, 10)
            y = random.uniform(-10, 10)
            aux = Chromosome(x, y, 0, 0, 0)
            self.population.append(aux)

        # Maximum and minimum index values
        self.max_index = population_size
        self.min_index = 0

        # Top 5 solutions/individuals
        self.top_ranking = []

    # Prints the top chromosomes for every iteration
    def print_top_ranking(self, generation):
        print(("=" * 10) + str(generation + 1) + ("=" * 10))
        aux = 1
        for chromosome in self.top_ranking:
            print(f"Chromosome {aux} - ({chromosome.rank}): [x={chromosome.x}; y={chromosome.y}; z={chromosome.bird}]")
            aux += 1
        print("=" * 20)

    # Is the standard for how well a sample is adapted
    def fitness_function(self, sample):
        return ((math.sin(sample.x) * math.exp((1 - math.cos(sample.y))**2)) +  \
                (math.cos(sample.y) * math.exp((1 - math.sin(sample.x))**2)) +  \
                ((sample.x - sample.y)**2))
    
    # Calculates the linear index value
    def linear_index(self, index):
        return self.min_index + (self.max_index - self.min_index) * \
            ((len(self.population) - index) / (len(self.population) - 1))
    
    # Tests the population and orders them according to their bird value
    def fitness_test(self):
        # Array of phenotypes
        results = []

        # Populates the array of phenotypes
        for i in range(len(self.population)):
            # Calculate the fitness value for each individual
            fitness_value = self.fitness_function(self.population[i])
            self.population[i].bird = fitness_value
            results.append(self.population[i])

        # Sorts the population according to their bird (target) value
        sorted_results = sorted(results, key=lambda x: x.bird)

        # Reorders the population
        self.population = sorted_results

    # Calculates the population's linear ranking and reorders it
    def linear_ranking(self):
        # Ranks each sample in the population
        count, total_aptitude = 0, 0
        for sample in self.population:
            sample.rank = self.linear_index(count + 1)
            total_aptitude += sample.rank
            sample.aptitude = total_aptitude
            count += 1

        # Reorders the population according to the linear ranking
        self.population = sorted(self.population, key=lambda x: x.rank, reverse=True)

    # Produces the offspring of two chromosomes according to the environmental crossover rate
    def chromosome_crossover(self, chromosome_a, chromosome_b):
        # First crossover rate
        r1 = self.crossover_rate
        
        # New data calculation for first chromosome
        new_x_1 = (r1*chromosome_a.x + (1-r1)*chromosome_b.x)
        new_y_1 = (r1*chromosome_a.y + (1-r1)*chromosome_b.y)
        
        # First descendant chromosome
        new_chromosome_a = Chromosome(new_x_1, new_y_1, 0, 0, 0)
        new_chromosome_a.bird = self.fitness_function(new_chromosome_a)

        # Second crossover rate
        r2 = 1-r1
        
        # New data calculation for second chromosome
        new_x_2 = (r2*chromosome_a.x + (1-r2)*chromosome_b.x)
        new_y_2 = (r2*chromosome_a.y + (1-r2)*chromosome_b.y)
        
        # Second descendant chromosome
        new_chromosome_b = Chromosome(new_x_2, new_y_2, 0, 0, 0)
        new_chromosome_a.bird = self.fitness_function(new_chromosome_b)
        
        # Returns both descendants
        return (new_chromosome_a, new_chromosome_b)

    # Mutates single chromosome according to environmental mutation rate    
    def chromosome_mutation(self, chromosome):
        x = chromosome.x 
        y = chromosome.y    
        x += random.uniform(-3,3)
        y += random.uniform(-3,3)
        mutated_chromosome = Chromosome(x,y,0,0,0)
        mutated_chromosome.bird = self.fitness_function(mutated_chromosome)
        return mutated_chromosome


    