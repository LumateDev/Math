import numpy as np
import matplotlib.pyplot as plt


def graph(arg: float) -> None:
    x_1 = np.linspace(-6, 4, 1000)
    plt.title("x^2 + 4x + 5")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x_1, f(x_1))
    plt.scatter(arg, f(arg))
    plt.show()


def fib(n):
    if n <= 1:
        return n
    else:
        return fib(n - 1) + fib(n - 2)


def r(N):
    return 1 / fib(N)


def f(arg):
    # Определение функции
    return arg ** 2 + 4 * arg + 5


a = -4
b = 6
e = 0.2
l = 0.5
fu = (b - a) / l
N = 1

while fu > fib(N):
    N += 1

k = 0
y = a + (fib(N - 2) / fib(N)) * (b - a)
z = a + (fib(N - 1) / fib(N)) * (b - a)

# Метод фибоначи
while abs(b - a) > l:
    if f(y) < f(z):
        b = z
        z = y
        y = a + (fib(N - k - 1) / fib(N - k + 1)) * (b - a)
    else:
        a = y
        y = z
        z = a + (fib(N - k) / fib(N - k + 1)) * (b - a)
    k += 1

y = z
z = y + e

if f(y) <= f(z):
    b = z
else:
    a = y

x = (a + b) / 2

print("Ряд Фибоначчи:")
for i in range(1, N + 1):
    print(fib(i))

print("Количество Итераций:", k)
print("Точка минимума:", x)
print("Значение функции:", f(x))
print("Интервал:", a, b)
print("Сходимость:", r(N + 1))
graph(x)
