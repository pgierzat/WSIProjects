import numpy as np
import random
import matplotlib.pyplot as plt


w1 = 10.0  # param. w1
w2 = 1.0  # param. w2
R = 0.25  # zasięg anteny
n = 10  # max liczba anten
M = 100  # rozmiar siatki
population_size = 50
generations = 100
mutation_rate = 0.01
sigma = 0.05
mutation_toggle_rate = 0.01


def grid(M):
    x = np.linspace(0, 1, M)
    y = np.linspace(0, 1, M)

    X, Y = np.meshgrid(x, y)
    return list(zip(X.ravel(), Y.ravel()))


def coverage(antennas, R):
    covered_points = 0
    grid_points = grid(M)
    for (x, y) in grid_points:
        for antenna in antennas:
            if antenna is not None:
                if (x - antenna[0])**2 + (y - antenna[1])**2 <= R**2:
                    covered_points += 1
                    break
    return covered_points / (M * M)


def grade(antennas, R):
    active_antennas = [a for a in antennas if a is not None]
    return w1 * coverage(active_antennas, R) - w2 * len(active_antennas)


def initialize_population():
    return [[(random.uniform(0, 1), random.uniform(0, 1)) if random.random() < 0.5 else None for _ in range(n)]]


def roulette_selection(population, grades):
    min_grade = min(grades)
    if min_grade < 0:
        grades = [g - min_grade for g in grades]

    total_grade = sum(grades)
    if total_grade == 0:
        selection_probs = [1 / len(grades)] * len(grades)
    else:
        selection_probs = [g / total_grade for g in grades]
    selected_index = np.random.choice(range(len(population)), p=selection_probs)
    return population[selected_index]


def crossover(parent1, parent2):
    point = random.randint(1, n - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(individual):
    for i in range(n):
        if individual[i] is not None:
            if random.random() < mutation_rate:
                individual[i] = (
                    min(1, max(0, individual[i][0] + np.random.normal(0, sigma))),
                    min(1, max(0, individual[i][1] + np.random.normal(0, sigma)))
                )
            elif random.random() < mutation_toggle_rate:
                individual[i] = None
        else:
            if random.random() < mutation_toggle_rate:
                individual[i] = (random.uniform(0, 1), random.uniform(0, 1))
    return individual


def evolutionary_algorithm(R):
    population = initialize_population()
    best_solution = None
    best_grade = float('-inf')

    for gen in range(generations):
        grades = [grade(individual, R) for individual in population]

        max_index = np.argmax(grades)
        if grades[max_index] > best_grade:
            best_solution = population[max_index]
            best_grade = grades[max_index]

        new_population = []
        while len(new_population) < population_size:
            parent1 = roulette_selection(population, grades)
            parent2 = roulette_selection(population, grades)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])

        population = new_population[:population_size]
    return best_solution, best_grade




def plot_antennas(antennas):
    fig, ax = plt.subplots(figsize=(8, 8))

    grid_points = grid(M)
    x_grid, y_grid = zip(*grid_points)
    ax.scatter(x_grid, y_grid, color='lightgrey', s=1, label='Punkty siatki')

    for antenna in antennas:
        if antenna is not None:
            x, y = antenna
            ax.scatter(x, y, color='red', s=50, marker='o', label='Antena')

            circle = plt.Circle((x, y), R, color='blue', alpha=0.2, edgecolor='blue', linestyle='--')
            ax.add_patch(circle)

    ax.set_aspect('equal', adjustable='box')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.xticks([0, 0.5, 1])
    plt.yticks([0, 0.5, 1])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Rozmieszczenie anten i ich zasięg na siatce")
    plt.legend(["Punkty siatki", "Antena", "Zasięg anteny"], loc="upper right")
    plt.grid(True)
    plt.savefig("Wykres.png")


# R dependancy simulation

R_values = [0.1, 0.15, 0.2, 0.25, 0.3]
generations = 10
num_runs_per_R = 5


def analyze_average_coverage_vs_radius():
    average_best_grades = []
    for R in R_values:
        best_grades_for_R = []
        for _ in range(num_runs_per_R):
            best_solution, best_grade = evolutionary_algorithm(R)
            best_grades_for_R.append(best_grade)
        average_best_grade = sum(best_grades_for_R) / num_runs_per_R
        average_best_grades.append(average_best_grade)

    plt.figure(figsize=(8, 5))
    plt.plot(R_values, average_best_grades, marker='o')
    plt.xlabel('Zasięg anteny (R)')
    plt.ylabel('Średni wynik')
    plt.title(f'Średnie wynik zależności od R\n(Dla {generations} generacji, średnia z {num_runs_per_R} uruchomień)')
    plt.grid(True)
    plt.savefig('rdependancy10.png')


analyze_average_coverage_vs_radius()
