import numpy as np
import matplotlib.pyplot as plt

# Algorithm parameters
#-- Add your desired values here
num_bats = 100
num_iterations = 150
A = 0.9 #Scan Rate
r = 0.8 #Frequency update rate
alpha = 0.95
gamma = 0.5
Qmin = 0
Qmax = 2

num_devices = 5
max_power = np.array([2, 3, 1, 2, 2])
cost_coefficient = np.array([0.2, 0.3, 0.1, 0.2, 0.2])
demand_profile = np.array([4, 5, 2, 3, 4])


solutions = np.random.rand(num_bats, num_devices)

frequencies = np.zeros(num_bats)
velocities = np.zeros((num_bats, num_devices))

def plot_bats_trend(bats, best_solution, iteration, title="Tendencia Final de los Murciélagos"):
    plt.figure(figsize=(8, 6))

    for i in range(len(bats)):
        plt.scatter(bats[i][0], bats[i][1], c='blue', s=10)

    plt.scatter(best_solution[0], best_solution[1], c='red', marker='*', s=200)
    plt.title(title + f' - Mejor Iteración ({iteration})')
    plt.xlabel('Dimensión 1')
    plt.ylabel('Dimensión 2')
    plt.grid(True)
    plt.show()

def objective_function(solution):
    return np.sum(solution * cost_coefficient)

def rosenbrock(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

cost_history = []

for iteration in range(num_iterations):
    for i in range(num_bats):
        frequencies[i] = Qmin + (Qmax - Qmin) * np.random.rand()
        velocities[i] = velocities[i] + (solutions[i] - solutions[np.random.randint(num_bats)]) * frequencies[i]

        solutions[i] = np.clip(solutions[i] + velocities[i], 0, max_power)

        if np.random.rand() > r:
            if objective_function(solutions[i]) < objective_function(solutions[np.argmax(frequencies)]):
                solutions[i] = solutions[np.argmax(frequencies)] + alpha * np.random.normal(0, 1, num_devices)

    best_solution = solutions[np.argmin([objective_function(sol) for sol in solutions])]

    best_cost = objective_function(best_solution)
    cost_history.append(best_cost)

    validation_result = rosenbrock(best_solution)

    efficiency_percentage = 100.0 / (1.0 + validation_result)

    if (iteration + 1) % 20 == 0:
        print(f"Iteración {iteration + 1}")
        print(f"Mejor Solución - {best_solution}")
        print(f"Costo: {best_cost}")
        print(f"Validación (Rosenbrock): {validation_result}")
        print(f"Eficacia: {efficiency_percentage:.2f}%" + "\n")
    
    r = gamma * r

plot_bats_trend(solutions, best_solution, num_iterations)

plt.plot(range(1, num_iterations + 1), cost_history, marker='o', linestyle='-', color='b')
plt.title('Evolución del Costo de Energía con Bat Algorithm')
plt.xlabel('Iteración')
plt.ylabel('Costo de Energía')
plt.grid(True)
plt.show()

print("\n ===> Resultado Final: ")
print("Mejor solución encontrada:", best_solution)
print("Costo de energía asociado:", objective_function(best_solution))
print("Validación final (Rosenbrock):", rosenbrock(best_solution))
print("Eficacia final en %:", efficiency_percentage)