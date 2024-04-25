import math
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

def func(x1, x2):
    return 3 * x1 ** 2 + x2 ** 2 - x1 * x2 + x1

def gradient(x1, x2):
    return 6 * x1 - x2 + 1, 2 * x2 - x1

def norm(vector):
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2)

def img_2d(x_values):
    plt.figure()
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = func(X, Y)

    plt.contour(X, Y, Z, levels=20)
    x_val = np.array(x_values)
    plt.plot(x_val[:, 0], x_val[:, 1], '-o', color='blue')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Gradient Descent Method')
    plt.grid(True)
    plt.show()

def img_3d(x_values):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = func(X, Y)

    ax.plot_surface(X, Y, Z, cmap='cool', alpha=0.5)
    x_val = np.array(x_values)
    ax.scatter(x_val[:, 0], x_val[:, 1], func(x_val[:, 0], x_val[:, 1]), c='blue', s=50)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    ax.set_title('Gradient Descent Method')
    ax.grid(True)
    plt.show()

def gradient_descent_method(start_point, alpha=0.1, ep=1e-5, M=100):
    x_values = [start_point]
    k = 0
    while True:
        grad = gradient(*x_values[k])
        grad_norm = norm(grad)
        if grad_norm < ep:
            break
        if k >= M:
            break
        d = [-alpha * grad[0], -alpha * grad[1]]
        new_point = [x_values[k][0] + d[0], x_values[k][1] + d[1]]
        k += 1
        x_values.append(new_point)
    print('x* =', x_values[-1])
    print('f(x*) =', func(*x_values[-1]))
    print('k =', len(x_values) - 1)
    print('Последние координаты точки x на каждой итерации:')
    for i, x_val in enumerate(x_values):
        print(f'Итерация {i + 1}: x1 = {x_val[0]}, x2 = {x_val[1]}, f(x) = {func(*x_val)}')
    return x_values

start_point = (1.5, 1.5)
result = gradient_descent_method(start_point)
img_2d(result)
img_3d(result)