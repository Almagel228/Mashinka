import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

np.random.seed(42)

N_ITEMS = 15
MAX_WEIGHT = 50

weights = np.random.randint(5, 26, N_ITEMS)
values = np.random.randint(10, 101, N_ITEMS)

print("Предметы:")
print(f"{'№':<5} {'Вес':<8} {'Ценность':<10} {'Ценность/Вес':<15}")
for i in range(N_ITEMS):
    print(f"{i + 1:<5} {weights[i]:<8} {values[i]:<10} {values[i] / weights[i]:<15.2f}")

print(f"\nВместимость рюкзака: {MAX_WEIGHT} кг")

POP_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 3


def create_individual():
    return [random.randint(0, 1) for _ in range(N_ITEMS)]


def fitness(individual):
    total_weight = sum(individual[i] * weights[i] for i in range(N_ITEMS))
    total_value = sum(individual[i] * values[i] for i in range(N_ITEMS))
    if total_weight > MAX_WEIGHT:
        return 0
    return total_value


def tournament_selection(population, fitnesses):
    tournament = random.sample(range(len(population)), TOURNAMENT_SIZE)
    best = tournament[0]
    for idx in tournament:
        if fitnesses[idx] > fitnesses[best]:
            best = idx
    return population[best]


def crossover(parent1, parent2):
    point = random.randint(1, N_ITEMS - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(individual):
    for i in range(N_ITEMS):
        if random.random() < MUTATION_RATE:
            individual[i] = 1 - individual[i]
    return individual


population = [create_individual() for _ in range(POP_SIZE)]

best_fitness_history = []
avg_fitness_history = []
best_individual = None
best_fitness = 0

print(f"\nЗапуск генетического алгоритма...")
print(f"Популяция: {POP_SIZE}, Поколений: {GENERATIONS}, Мутация: {MUTATION_RATE}")

for generation in range(GENERATIONS):
    fitnesses = [fitness(ind) for ind in population]

    gen_best = max(fitnesses)
    gen_avg = sum(fitnesses) / len(fitnesses)
    best_fitness_history.append(gen_best)
    avg_fitness_history.append(gen_avg)

    if gen_best > best_fitness:
        best_fitness = gen_best
        best_individual = population[fitnesses.index(gen_best)].copy()

    new_population = []

    best_idx = fitnesses.index(max(fitnesses))
    new_population.append(population[best_idx])

    while len(new_population) < POP_SIZE:
        parent1 = tournament_selection(population, fitnesses)
        parent2 = tournament_selection(population, fitnesses)

        child1, child2 = crossover(parent1, parent2)

        child1 = mutate(child1)
        child2 = mutate(child2)

        new_population.append(child1)
        if len(new_population) < POP_SIZE:
            new_population.append(child2)

    population = new_population

    if generation % 10 == 0:
        print(f"  Поколение {generation}: лучшая ценность = {gen_best}, средняя = {gen_avg:.0f}")

total_weight = sum(best_individual[i] * weights[i] for i in range(N_ITEMS))
total_value = sum(best_individual[i] * values[i] for i in range(N_ITEMS))

print(f"\nРезультат:")
print(f"Лучшая ценность: {total_value}")
print(f"Общий вес: {total_weight} / {MAX_WEIGHT} кг")
print(f"\nВыбранные предметы:")
for i in range(N_ITEMS):
    if best_individual[i] == 1:
        print(f"  Предмет {i + 1}: вес={weights[i]} кг, ценность={values[i]}")

plt.figure(figsize=(10, 6))
plt.plot(best_fitness_history, 'b-', linewidth=2, label='Лучшая ценность')
plt.plot(avg_fitness_history, 'r--', linewidth=1, alpha=0.7, label='Средняя ценность')
plt.xlabel('Поколение')
plt.ylabel('Ценность')
plt.title('Сходимость генетического алгоритма (задача о рюкзаке)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('genetic_algorithm_convergence.png', dpi=100, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
colors = ['green' if best_individual[i] == 1 else 'gray' for i in range(N_ITEMS)]
bars = plt.bar(range(1, N_ITEMS + 1), weights, color=colors, alpha=0.7)
plt.axhline(y=MAX_WEIGHT, color='red', linestyle='--', linewidth=2, label=f'Лимит ({MAX_WEIGHT} кг)')
plt.xlabel('Номер предмета')
plt.ylabel('Вес (кг)')
plt.title('Выбранные предметы (зеленые)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.bar(range(1, N_ITEMS + 1), values, color=colors, alpha=0.7)
plt.xlabel('Номер предмета')
plt.ylabel('Ценность')
plt.title('Ценность выбранных предметов')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('knapsack_solution.png', dpi=100, bbox_inches='tight')
plt.close()

print(f"\nГрафики: genetic_algorithm_convergence.png, knapsack_solution.png")