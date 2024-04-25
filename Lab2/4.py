import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


def func(x1, x2):
    return 3 * x1 ** 2 + x2 ** 2 - x1 * x2 + x1


def gradient(x1, x2):
    return np.array([6 * x1 - x2 + 1, 2 * x2 - x1])


def norm(vector):
    return np.sqrt(vector[0] ** 2 + vector[1] ** 2)


def fletcher_reeves_method(start_point, eps1=0.1, eps2=0.15, M=10):
    x_values = [np.array(start_point)]
    k = 0
    flag = False
    prev_grad = gradient(*start_point)
    d = -prev_grad

    while True:
        grad = gradient(*x_values[k])
        grad_norm = norm(grad)

        if grad_norm < eps1:
            break
        if k >= M:
            break

        a = optimize.minimize_scalar(lambda alpha: func(*(x_values[k] + alpha * d))).x
        new_point = x_values[k] + a * d

        if grad_norm ** 2 < eps2 or k == 0:
            beta = 0
        else:
            beta = np.dot(grad, grad) / np.dot(prev_grad, prev_grad)

        d = -grad + beta * d

        k += 1
        x_values.append(new_point)
        prev_grad = grad

    print('x* =', x_values[-1])
    print('f(x*) =', func(*x_values[-1]))
    print('k =', len(x_values) - 1)
    print('Последние координаты точки x на каждой итерации:')
    for i, x_val in enumerate(x_values):
        print(f'Итерация {i + 1}: x1 = {x_val[0]}, x2 = {x_val[1]}, f(x) = {func(*x_val)}')

    img_2d(x_values)
    img_3d(x_values)

    return x_values


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
    plt.title('Fletcher-Reeves Method')
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
    ax.plot(x_val[:, 0], x_val[:, 1], func(x_val[:, 0], x_val[:, 1]), '-o', color='blue')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    ax.set_title('Fletcher-Reeves Method')
    ax.grid(True)
    plt.show()


start_point = (1.5, 1.5)
result = fletcher_reeves_method(start_point)