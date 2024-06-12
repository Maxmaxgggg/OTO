import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List

ITER_MAX_SD = 10000
ITER_MAX_GR = 1000
TOLERANCE = np.float64(1e-10)


# Определяем квадратичную функцию Химмельблау для построения графиков
def him_pl(x: float, y: float) -> float:
    return (x**2+y-11)**2+(x+y**2-7)**2


# Определяем квадратичную функцию Химмельблау для оптимизации
def himmelblau(x: npt.NDArray[np.float64]) -> np.float64:
    if len(x) != 2:
        raise ValueError("Массив должен состоять из 2-х элементов")
    return np.float64((x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2)


# Вручную задаем градиент целевой функции
def gradient(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.array([4*x[0]*(x[0]**2+x[1]-11)+2*x[0]+2*x[1]**2-14,
                     2*x[0]**2+4*x[1]*(x[0]+x[1]**2-7)+2*x[1]-22], dtype=np.float64)


# Функция поиска минимума с использованием метода золотого сечения
def golden_ratio_search(f: Callable[[np.float64], np.float64],  # Функция, для которой ищем минимум
                        a: np.float64, b: np.float64,  # Левый и правый концы интервала
                        itr=ITER_MAX_GR, tol=np.float64(1e-10)  # Максимальное число итераций и точность
                        ) -> np.float64:
    if tol > TOLERANCE:
        tol = TOLERANCE
    iterations = 0
    while iterations < itr:
        iterations += 1
        phi = np.float64((1+np.sqrt(5))/2)
        x1 = b - (b - a) / phi
        x2 = a + (b - a) / phi
        y1 = f(x1)
        y2 = f(x2)
        if y1 >= y2:
            a = x1
        else:
            b = x2
        if np.abs(b - a) < tol:
            break
    return (a + b) / 2


def steepest_descent(fun: Callable[[npt.NDArray[np.float64]], np.float64],  # Целевая функция для оптимизации
                     grad: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],  # Ее градиент
                     x0: npt.NDArray[np.float64], itr: int = ITER_MAX_GR,  # Нач точка и число итераций
                     tol: np.float64 = TOLERANCE  # Погрешность метода
                     ) -> Tuple[npt.NDArray[np.float64], list]:
    x_it = x0
    conv = [x0.copy()]
    r = -1 * grad(x_it)
    r_n = np.sqrt(np.dot(r.T, r))
    it = 0

    # Одномерный алгоритм оптимизации использующий метод золотого сечения
    def goldenratio_fsd(x_f, d_f) -> np.float64:
        def func_x(x: np.float64):
            return fun(x_f + x * d_f)

        # 1 шаг - устанавливаем интервал
        delt = np.float64(0.01)
        a0 = np.float64(0)
        a_it = a0 + delt
        b_it = a0
        k = 1
        while True:
            #  Спускаемся в выбранном направлении
            f_old, f_new = func_x(b_it), func_x(a_it)
            if f_old >= f_new:
                b_it = a_it
                a_it += 2 ** k * delt
                k += 1
                continue
            if f_old <= f_new:
                break
        # 2 шаг - уменьшаем интервал, используя метод золотого сечения
        return golden_ratio_search(func_x, b_it - delt, a_it + delt)
    while r_n > tol and it < itr:
        a = goldenratio_fsd(x0, r)
        x_it += a*r
        conv.append(x_it.copy())
        r = -1 * grad(x_it)
        r_n = np.sqrt(np.dot(r.T, r))
        it += 1
    return x_it, conv


# Решаем задачу оптимизации
(sol, conv) = steepest_descent(himmelblau, gradient, np.array([0, 0], dtype=np.float64))
fval = himmelblau(sol)

#                  Построение графиков.
# Создаем сетку значений для x и y
x_lin = np.linspace(-5, 5, 100)
y_lin = np.linspace(-5, 5, 100)
x_lin, y_lin = np.meshgrid(x_lin, y_lin)

# Вычисляем z на основе функции f
z = him_pl(x_lin, y_lin)

# Создаем фигуру и ось 3D
fig1 = plt.figure(1)
ax = fig1.add_subplot(111, projection='3d')

# Рисуем поверхность
surface = ax.plot_surface(x_lin, y_lin, z, cmap='viridis', alpha=0.4)

# Добавляем на поверхность точки
conv = np.array(conv)
ax.scatter(conv[:, 0], conv[:, 1], him_pl(conv[:, 0], conv[:, 1]),
           color='red', edgecolor='black', label='Точки сходимости')
ax.plot(conv[:, 0], conv[:, 1], him_pl(conv[:, 0], conv[:, 1]),
        linestyle='--', color='red')
for i, point in enumerate(conv):
    ax.text(point[0], point[1], him_pl(point[0], point[1]),
            f'{i}', color='black', fontsize=10)
# Добавляем цветовую шкалу
fig1.colorbar(surface)

# Добавляем метки осей
ax.set_xlabel('Ось X')
ax.set_ylabel('Ось Y')
ax.set_zlabel('Ось Z')

# Добавляем легенду графика
ax.legend()
# Показываем график
plt.show()

# Строим график с изолиниями
fig2 = plt.figure(2)
plt.contour(x_lin, y_lin, z, levels=np.logspace(0, 5, 50), colors='blue')
plt.scatter(conv[:, 0], conv[:, 1], color='red', edgecolor='black', label='Точки сходимости')
plt.plot(conv[:, 0], conv[:, 1], linestyle='--', color='red')
plt.colorbar(label='Уровни')
# Добавляем метки осей и заголовок
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Изолинии функции Химмельблау')
# Показываем график
plt.show()

# Строим график сходимости
fig3 = plt.figure(3)
eps_arr = [np.sqrt(np.dot(gradient(p).T, gradient(p))) for p in conv]
plt.plot(eps_arr)
plt.title('График сходимости алгоритма')
plt.xlabel('Итерация')
plt.ylabel('|∇f(x)| (логарифмический масштаб)')
plt.yscale('log')
plt.show()
