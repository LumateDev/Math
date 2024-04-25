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

#Разница векторов
def norm_diff(vector1, vector2):
    return math.sqrt((vector1[0] - vector2[0]) ** 2 + (vector1[1] - vector2[1]) ** 2)


def hessian_matrix():
    x1, x2 = sp.symbols('x1 x2')
    f = 3 * x1 ** 2 + x2 ** 2 - x1 * x2 + x1

    hes = [[sp.diff(sp.diff(f, var1), var2) for var1 in [x1, x2]] for var2 in [x1, x2]]

    return [[elem.evalf() for elem in row] for row in hes]


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
    plt.title('Newton\'s Method')
    plt.grid(True)
    plt.show()


def img_3d(x_values):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = func(X, Y)

    ax.plot_surface(X, Y, Z, cmap='cool', alpha=0.5)  # Изменен цвет на cool

    x_val = np.array(x_values)
    ax.scatter(x_val[:, 0], x_val[:, 1], func(x_val[:, 0], x_val[:, 1]), c='blue', s=50)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    ax.set_title('Newton\'s Method')
    ax.grid(True)
    plt.show()


def newton_method(start_point, ep1=0.1, ep2=0.15, M=10):
    x_values = [start_point]
    k = 0
    flag = False
    while True:
        grad = gradient(*x_values[k])
        grad_norm = norm(grad)
        if grad_norm < ep1:
            break
        if k >= M:
            break

        hes = hessian_matrix()
        inv_hes = inverse(hes)
        d = None
        if inv_hes[0][0] > 0 and hes[0][0] * hes[1][1] - hes[0][1] * hes[1][0] > 0:
            d = [- (inv_hes[0][0] * grad[0] + inv_hes[0][1] * grad[1]),
                 - (inv_hes[1][0] * grad[0] + inv_hes[1][1] * grad[1])]
        else:
            d = [-grad[0], -grad[1]]
        new_point = [x_values[k][0] + d[0], x_values[k][1] + d[1]]
        if norm_diff(new_point, x_values[k]) > ep2 or abs(func(*new_point) - func(*x_values[k])) > ep2:
            k += 1
            x_values.append(new_point)
        else:
            if flag:
                break
            else:
                flag = True
                k += 1
                x_values.append(new_point)
    print('x* =', x_values[-1])
    print('f(x*) =', func(*x_values[-1]))
    print('k =', len(x_values) - 1)
    print('Последние координаты точки x на каждой итерации:')
    for i, x_val in enumerate(x_values):
        print(f'Итерация {i + 1}: x1 = {x_val[0]}, x2 = {x_val[1]}, f(x) = {func(*x_val)}')
    return x_values


def inverse(matrix):
    det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    inv_det = 1 / det
    inv_matrix = [[matrix[1][1] * inv_det, -matrix[0][1] * inv_det], [-matrix[1][0] * inv_det, matrix[0][0] * inv_det]]
    return inv_matrix


start_point = (1.5, 1.5)
result = newton_method(start_point)
img_2d(result)
img_3d(result)