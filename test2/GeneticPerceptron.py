import numpy as np
import random as rd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

# Genetic Algorithm

population_size = 3
tax_crossover = 0.7
tax_mutation = 0.1
generations = 70
populations = []

# Function to divide data into training and testing sets
def divide_data(in_array, proportion_type=None):
    array = []
    array2 = []

    if proportion_type is None:
        for i in range(round(len(in_array) * 0.7)):
            array.append(in_array[i])

        for j in range(round(len(in_array) * 0.3 - 1), -1, -1):
            array2.append(in_array[j])

    return [array, array2]


def initialize_population(population_size):
    population = []
    # Generate a population with bounds [-1, 1]
    for _ in range(population_size):
        population.append([np.random.uniform(-1, 1) for _ in range(4)])

    return population


def select_parents(population):
    population_for_select = sorted(population, key=fitness, reverse=True)

    # Calculate the relative position in the linear ranking
    ranking = [1 + (len(population_for_select) - 1) * ((len(population_for_select) - i) / (len(population_for_select) - 1))
               for i in range(len(population_for_select))]

    # Calculate the sum of rankings
    soma = sum(ranking)

    # Select the first parent
    parent_A = select_parent(population_for_select, ranking, soma)

    # Select the second parent
    parent_B = select_parent(population_for_select, ranking, soma)

    return parent_A, parent_B


def select_parent(population_for_select, ranking, soma):
    num_random = rd.randint(0, soma)
    accumulated = 0
    selected_parent = None

    for i in range(len(population_for_select)):
        accumulated += ranking[i]
        if accumulated >= num_random:
            selected_parent = population_for_select[i]
            break

    return selected_parent


def crossover(parent_A, parent_B):
    child_A, child_B = [], []
    alfa = np.random.uniform(0, 1)

    if np.random.uniform(0, 1) < tax_mutation:
        child_A = [alfa * parent_A[i] + (1 - alfa) * parent_B[i] for i in range(len(parent_A))]
        child_B = [alfa * parent_B[i] + (1 - alfa) * parent_A[i] for i in range(len(parent_B))]
        return mutation(child_A), mutation(child_B)
    else:
        return parent_A, parent_B


def mutation(individual):
    if np.random.uniform(0, 1) < tax_mutation:
        individual = [gene + np.random.uniform(-0.5, 0.5) for gene in individual]
    return individual


# Artificial Neural Networks - Perceptron

def sigmoidal(u):
    y = [0, 0, 0]
    max_index = np.argmax(u)
    y[max_index] = 1
    return y


def are_close(individual1, individual2, threshold=1e-6):
    return np.all(np.abs(np.array(individual1) - np.array(individual2)) < threshold)


class Perceptron:
    def __init__(self, activation_function):
        self.weights = np.full((3, 4), 0.1)
        self.bias = np.full((1, 3), 0.1)
        self.activation_function = activation_function

    def train(self, max_it, a, X, D):
        t = 0
        E = 1
        Ep = []

        while t < max_it and E > 0:
            E = 0
            y = []
            e = []

            for i in range(len(X)):
                x = np.array([X[i]])
                u = np.dot(self.weights, x.T) + self.bias.T
                y.append(self.activation_function(u))
                e.append((np.array(D[i]) - np.array(y[i])).tolist())
                self.weights = self.weights + (a * np.dot(np.array([e[i]]).T, x))
                self.bias = self.bias + (a * np.array([e[i]]))
                ee = np.dot(np.array([e[i]]), np.array([e[i]]).T)
                E += ee

            Ep.append(E[0][0])  # Save the error value
            t += 1

        return [self.weights, self.bias, Ep]

    def test(self, X, D):
        hit = 0

        for i in range(len(X)):
            x = np.array([X[i]])
            u = np.dot(self.weights, x.T) + self.bias.T
            y = self.activation_function(u)
            e = (np.array(D[i]) - np.array(y)).tolist()

            if e == [0, 0, 0]:
                hit += 1

        hit_rate = (hit / len(X)) * 100
        return hit_rate

    def predict(self, X):
        return [self.activation_function(np.dot(self.weights, np.array([x]).T) + self.bias.T) for x in X]


DTrad = {'Iris-setosa': [0, 0, 1], 'Iris-versicolor': [0, 1, 0], 'Iris-virginica': [1, 0, 0]}
XTrain, DTrain, XTest, DTest = [], [], [], []
linhas = []

# Read the data file
iris = pd.read_csv('Iris_Data.csv')
iris_data = iris.iloc[:, :].values
rd.shuffle(iris_data)
linhas = iris_data

# Split the data
for linha in divide_data(linhas)[0]:
    XTrain.append([float(value) for value in linha[:4]])
    DTrain.append(DTrad[linha[4]])

for linha in divide_data(linhas)[1]:
    XTest.append([float(value) for value in linha[:4]])
    DTest.append(DTrad[linha[4]])

# Fitness function
perceptron = Perceptron(sigmoidal)


def fitness(weights):
    perceptron.weights = weights
    y_pred = perceptron.predict(XTest)
    return -accuracy_score(DTest, y_pred)


# Execute the genetic algorithm
population = initialize_population(population_size)
populations.append(population)

best_individual = max(population, key=fitness)
best_trio = []

for _ in range(generations):
    new_population = []

    for _ in range(population_size // 2):
        parent_A, parent_B = select_parents(population)
        child_A, child_B = crossover(parent_A, parent_B)

        new_population.extend([mutation(child_A), mutation(child_B)])

    population = new_population

    best_individual_in_generation = max(population, key=fitness)
    if len(best_trio) < 3:
        best_trio.append(best_individual_in_generation)
    else:
        worst_individual_in_trio = max(best_trio, key=fitness)

        for ind in best_trio:
            if fitness(ind) < fitness(worst_individual_in_trio):
                worst_individual_in_trio = ind

        worst_index = next((i for i, ind in enumerate(best_trio) if are_close(ind, worst_individual_in_trio)), None)

        if worst_index is not None:
            best_trio[worst_index] = best_individual_in_generation

    populations.append(population)

best_weights = best_trio

# Set the best weights in the Perceptron
perceptron.weights = best_weights

# Train the perceptron
weights_bias_error = perceptron.train(50, 0.9, XTrain, DTrain)
error_values = weights_bias_error[2]

# Evaluate the perceptron
accuracy = perceptron.test(XTest, DTest)

# Print the matrix of weights with proper formatting
best_weights = np.array(best_weights)
print("Best Weights:")
print(np.array2string(best_weights, precision=4, suppress_small=True))

# Set seaborn style and context for a different plot style
sns.set(style="whitegrid", context="notebook", palette="deep")

# Create a plot with seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(x=range(1, len(error_values) + 1), y=error_values, marker='o', color='coral', markersize=8)
plt.title('Erro vs. Época', fontsize=18)
plt.xlabel('Época', fontsize=14)
plt.ylabel('Erro', fontsize=14)

# Exibir o gráfico
plt.show()

# Print the accuracy in Portuguese
accuracy_percent = "{:.2%}".format(accuracy / 100)
print(f"\nAcurácia da melhor rede neural ajustada: {accuracy_percent}")



