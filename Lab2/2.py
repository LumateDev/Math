import math
import numpy as np
import matplotlib.pyplot as plt

def func(x, y):
    return 3 * x ** 2 + y ** 2 - x * y + x

def grad_f(x, y):
    return [6 * x - y + 1, 2 * y - x]

def hessian(x, y):
    return [[6, -1], [-1, 2]]

def golden_section_search(func, a, b, tol=1e-5):
    # Золотое сечение для поиска минимума функции
    gr = (math.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(c - d) > tol:
        if func(c) < func(d):
            b = d
        else:
            a = c
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    return (b + a) / 2

def newton_method(x, y, eps1=0.1, eps2=0.15, M=10):
    x_values = [(x, y)]
    k = 0

    while True:
        grad = grad_f(x, y)
        norm_grad = math.sqrt(grad[0] ** 2 + grad[1] ** 2)

        # Проверка на условие сходимости по градиенту
        if norm_grad < eps1:
            break

        # Проверка на максимальное количество итераций
        if k >= M:
            break

        hess_inv = np.linalg.inv(hessian(x, y))
        direction = np.dot(hess_inv, grad)

        # Применение метода золотого сечения для поиска оптимального шага
        t = golden_section_search(lambda t: func(x - t * direction[0], y - t * direction[1]), 0, 1)

        x_new = x - t * direction[0]
        y_new = y - t * direction[1]

        f_x = func(x, y)
        f_x_new = func(x_new, y_new)

        # Проверка на условие сходимости по изменению значения функции и изменению координат
        if abs(f_x_new - f_x) < eps2 and math.sqrt((x_new - x) ** 2 + (y_new - y) ** 2) < eps2:
            break

        x, y = x_new, y_new
        x_values.append((x, y))
        k += 1

    return (x, y), func(x, y), k, x_values

x_start, y_start = 1.5, 1.5
result, f_result, iterations, x_values = newton_method(x_start, y_start)
print('Исходная функция: f(x) = 3x^2 + y^2 - xy + x ')
print('x* =', result)
print('f(x*) =', f_result)

print('Количество итераций до сходимости:', iterations + 1)
print('Последние координаты x на каждой итерации:')
for i, x_val in enumerate(x_values):
    print(f'Итерация {i + 1}: x1 = {x_val[0]}, x2 = {x_val[1]}, f(x) = {func(*x_val)}')

# Визуализация пути оптимизации
def img_2d(x_vals, func):
    plt.figure()
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = func(X, Y)

    plt.contour(X, Y, Z, levels=20)
    x_val = np.array(x_vals)
    plt.plot(x_val[:, 0], x_val[:, 1], '-o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Newton Method')
    plt.grid(True)
    plt.show()

def img_3d(x_vals, func):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = func(X, Y)

    ax.plot_surface(X, Y, Z, cmap='jet', alpha=0.5)

    x_val = np.array(x_vals)
    ax.scatter(x_val[:, 0], x_val[:, 1], func(x_val[:, 0], x_val[:, 1]), c='r', s=50)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title('Newton Method')
    ax.grid(True)
    plt.show()

img_2d(x_values, func)
img_3d(x_values, func)
