import random
from Chromosome import Chromosome
from Environment import Environment

if __name__ == "__main__":
    # Data variables
    population_size = 100
    crossover_rate = 0.6
    mutation_rate = 0.015
    
    # Environment instantiation
    environment = Environment(population_size, crossover_rate, mutation_rate)
    
    # This array will store the top results
    top_results = []

    # Runs GA 100 times
    for i in range(100):
        # Evaluates 1000 generations
        for i in range(10**3):
            # Tests and reorganization
            environment.fitness_test()
            environment.linear_ranking()

            # Top ranking selection and display
            environment.top_ranking = environment.population[:5]
            #environment.print_top_ranking(i)

            # Roulette selection
            aptitudes = []
            for sample in environment.population:
                aux = sample.aptitude
                aptitudes.append(aux)
            selected_samples = []
            for i in range(population_size):
                aux = random.randint(0, int(environment.population[-1].aptitude))
                for j in range(population_size):
                    if aptitudes[j] >= aux:
                        selected_samples.append(environment.population[i])
                        break
            environment.population = selected_samples

            # Crossover and mutation
            descendants = []
            for i in range(int(len(environment.population)/2)):
                child_1, child_2 = environment.chromosome_crossover(environment.population[i], environment.population[-i])
                descendants.append(child_1)
                descendants.append(child_2)
            environment.population = descendants
            for i in range(int(len(environment.population))):
                if (random.uniform(0,100)/100) >= environment.mutation_rate:
                    environment.population[i] = environment.chromosome_mutation(environment.population[i])

        # Results display
        top_results.append((environment.top_ranking[0].x, environment.top_ranking[0].y, environment.top_ranking[0].bird))

    print("Top 5 results after 100 iterations of the algorithm:")
    top_results = sorted(top_results, key=lambda x: x[2])
    for result in top_results[:5]:
        print(result)


