import math

import numpy as np
import matplotlib.pyplot as plt


def f(arg):
    # Определение функции
    return arg**2 + 4 * arg + 5

def plot_graph():
    x_1 = np.linspace(-10, 6, 1000)

    # Построение графика
    plt.title("x^2 + 4x + 5")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x_1, f(x_1))
    plt.scatter(x, f(x))
    plt.show()

# Метод золотого сечения
a = -4
b = 6
tau = (1 + math.sqrt(5)) / 2 # 0.618
epsilon = 0.5

while abs(b - a) > epsilon:
    x1 = b - ((b - a) / tau)
    x2 = a + ((b - a) / tau)

    if f(x1) >= f(x2):
        a = x1
    else:
        b = x2

x = (a + b) / 2
print("Точка минимума: ", x)
print("Значение функции: ", f(x))
print("Интервал: ", a, b)


plot_graph()
