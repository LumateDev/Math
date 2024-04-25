import math
import matplotlib.pyplot as plt
import numpy as np

def func(x1, x2):
    return 3 * x1 ** 2 + x2 ** 2 - x1 * x2 + x1


def gradient(x1, x2):
    return 6 * x1 - x2 + 1, 2 * x2 - x1

def norm(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def new_x(x, grad_x, t):
    new_x1 = x[0] - t * grad_x[0]
    new_x2 = x[1] - t * grad_x[1]
    return new_x1, new_x2

def img_2d(x_vals, func):
    plt.figure()
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = func(X, Y)

    plt.contour(X, Y, Z, levels=20)
    x_val = np.array(x_vals)
    plt.plot(x_val[:, 0], x_val[:, 1], '-o')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Gradient Descent')
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
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    ax.set_title('Gradient Descent')
    ax.grid(True)
    plt.show()

def gradient_descent(x, eps1=0.1, eps2=0.15, M=100):
    x_values = [x]
    k = 0

    while True:
        grad = gradient(x[0], x[1])
        norm_grad = norm(grad, (0, 0))

        if norm_grad < eps1:
            break

        if k >= M:
            break

        t = 0.1  # Шаг метода наискорейшего градиентного спуска
        x_new = new_x(x, grad, t)

        f_x = func(*x)
        f_x_new = func(*x_new)

        if abs(f_x_new - f_x) < eps2 and norm(x_new, x) < eps2:
            break

        x = x_new
        x_values.append(x)
        k += 1  # Перемещаем инкремент в эту строку

    return x, func(*x), k, x_values

x_start = (1.5, 1.5)
result, f_result, iterations, x_values = gradient_descent(x_start)
print('Иходная функция: f(x) = 3x1^2 + x2^2 - x1x2 + x1 ')
print('x* =', result)
print('f(x*) =', f_result)

print('Итераций до сходимости:', iterations + 1)
print('Последние координаты точки x на каждой итерации:')
for i, x_val in enumerate(x_values):
    print(f'Итерация {i + 1}: x1 = {x_val[0]}, x2 = {x_val[1]}, f(x) = {func(*x_val)}')

img_2d(x_values, func)
img_3d(x_values, func)