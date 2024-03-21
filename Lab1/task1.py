import numpy as np
import matplotlib.pyplot as plt

def f(arg):
    # Определение функции
    return arg**2 + 4 * arg + 5

def r(arg):
    # Определение функции для погрешности
    return 1 / 2**(arg / 2)

def plot_graph():
    x_1 = np.linspace(-10, 6, 1000)

    # Построение графика
    plt.title("x^2 + 4x + 5")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x_1, f(x_1))
    plt.scatter(x, f(x))
    plt.show()

ep = 0.2
l = 0.5
k = 0
a = -4
b = 6

while abs(b - a) > l:
    y = (a + b - ep) / 2
    z = (a + b + ep) / 2
    if f(y) < f(z):
        b = z
    else:
        a = y
    k += 2

x = (a + b) / 2
print("Точка минимума: ", x)
print("Значение функции: ", f(x))
print("Интервал: ", a, b)
print("Индекс интервала: ", k)
print("Сходимость: ", r(k))

plot_graph()