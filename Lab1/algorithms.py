import numpy as np
import matplotlib.pyplot as plt
import time


def func(x, y):
    return (2 * x ** 2 + y - 7) ** 2 + (x + 2 * y ** 2 - 5) ** 2


def gradient(x, y):
    dx = 16 * x ** 3 + 8 * x * y - 54 * x + 4 * y ** 2 - 10
    dy = 4 * x ** 2 + 8 * x * y + 16 * y ** 3 - 38 * y - 14
    return np.array([dx, dy])


def hessian(x, y):
    dxx = 48 * x ** 2 + 8 * y - 54
    dxy = 8 * x + 8 * y
    dyx = 8 * x + 8 * y
    dyy = 8 * x ** 2 + 48 * y ** 2 - 38
    return np.array([[dxx, dxy], [dyx, dyy]])


def newton_method(start_point, tol=1e-6, max_iter=1000):
    x = start_point.copy()
    path = [x]
    for i in range(max_iter):
        grad_val = gradient(x[0], x[1])
        hessian_inv = np.linalg.inv(hessian(x[0], x[1]))
        step_dir = -hessian_inv @ grad_val
        x_new = x + step_dir
        path.append(x_new.copy())
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return np.array(path), i + 1


def steepest_descent(start_point, step=0.002, tol=1e-6, max_iter=1000, domain=np.array([[-5, 5], [-5, 5]])):
    x = start_point.copy()
    path = [x]
    for i in range(max_iter):
        grad_val = gradient(x[0], x[1])
        x_new = x - step * grad_val
        x_new = np.clip(x_new, domain[:, 0], domain[:, 1])
        path.append(x_new.copy())
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return np.array(path), i + 1
# i + 1 --> liczba iteracji


# Plotter
def compare_algorithms(start_point):

    path_newton, iterations_newton = newton_method(start_point)

    path_sd, iterations_sd = steepest_descent(start_point)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(iterations_newton), [func(x[0], x[1]) for x in path_newton[:iterations_newton]], label='Metoda Newtona', marker='o')
    ax.plot(range(iterations_sd), [func(x[0], x[1]) for x in path_sd[:iterations_sd]], label='Metoda najszybszego spadku', marker='x')

    ax.set_xlabel('Liczba iteracji')
    ax.set_ylabel('Wartość funkcji celu')
    ax.set_title('Porównanie: Metoda Newtona vs. Metoda Najszybszego Spadku')
    ax.legend()
    plt.savefig('comparison_newton_vs_steepest_descent3.png')


def plot_search_space(start_point):

    x = np.linspace(-5, 5, 400)
    y = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    path_newton = newton_method(start_point)[0]
    path_sd = steepest_descent(start_point)[0]

    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=50, cmap='viridis')

    plt.plot(path_newton[:, 0], path_newton[:, 1], label='Metoda Newtona', color='red', marker='o')

    plt.plot(path_sd[:, 0], path_sd[:, 1], label='Metoda Najszybszego Spadku', color='blue', marker='x')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Porównanie: Przeszukiwanie przestrzeni przez metodę Newtona i najszybszego spadku')
    plt.legend()

    plt.savefig('search_space_comparison.png')


def compare_hyperparameters(start_points, steps, tolerances):
    newton_results = []
    steepest_results = []

    for start_point in start_points:
        newton_times = []
        steepest_times = []

        for step in steps:
            newton_iters = []
            steepest_iters = []

            for tol in tolerances:
                newton_iter = newton_method(start_point, tol=tol)[1]
                newton_iters.append(newton_iter)

                steepest_iter = steepest_descent(start_point, step=step, tol=tol)[1]
                steepest_iters.append(steepest_iter)

            newton_times.append(newton_iters)
            steepest_times.append(steepest_iters)

        newton_results.append(newton_times)
        steepest_results.append(steepest_times)

    fig, axs = plt.subplots(len(start_points), len(steps), figsize=(15, 10), sharex=True, sharey=True)
    fig.suptitle("Porównanie algorytmów w zależności od hiperparametrów")

    for i, start_point in enumerate(start_points):
        for j, step in enumerate(steps):

            newton_data = newton_results[i][j]
            steepest_data = steepest_results[i][j]

            axs[i, j].plot(tolerances, newton_data, label="Newton", marker='o', color='b')
            axs[i, j].plot(tolerances, steepest_data, label="Steepest Descent", marker='x', color='r')
            axs[i, j].set_title(f"Start={start_point}, Step={step}")
            axs[i, j].set_xlabel("Kryterium stopu (tolerancja)")
            axs[i, j].set_ylabel("Liczba iteracji")
            axs[i, j].legend()
            axs[i, j].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("Compare hiperparam")


def compare_algorithms_time(start_point, step=0.0005, tol=1e-6, max_iter=1000, runs=100):

    newton_times = []
    steepest_times = []

    for _ in range(runs):
        start_time = time.time()
        newton_method(start_point, tol=tol, max_iter=max_iter)
        newton_times.append(time.time() - start_time)

    for _ in range(runs):
        start_time = time.time()
        steepest_descent(start_point, step=step, tol=tol, max_iter=max_iter)
        steepest_times.append(time.time() - start_time)

    avg_newton = np.mean(newton_times)
    avg_steepest = np.mean(steepest_times)

    print("Porównanie szybkości działania algorytmów:")
    print(f"Metoda Newtona - Średni czas: {avg_newton:.6f}s")
    print(f"Metoda Najszybszego Spadku - Średni czas: {avg_steepest:.6f}s")

    plt.figure(figsize=(10, 6))
    plt.plot(newton_times, label="Metoda Newtona", color='b')
    plt.plot(steepest_times, label="Metoda Najszybszego Spadku", color='r')
    plt.xlabel("Numer uruchomienia")
    plt.ylabel("Czas działania (s)")
    plt.title("Porównanie czasu działania algorytmów dla każdego uruchomienia i stałego p. startu")
    plt.legend()
    plt.grid()
    plt.savefig("compare_time.png")


start_points = [np.array([1.0, 1.0]), np.array([3.0, 3.0]), np.array([-2.0, -2.0])]
steps = [0.0001, 0.001, 0.01]
tolerances = [1e-3, 1e-4, 1e-5, 1e-6]
start_point = np.array([3.0, -4.0])

plot_search_space(start_point)
compare_algorithms_time(start_point)
compare_algorithms(start_point)
compare_hyperparameters(start_points, steps, tolerances)
