import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from typing import Callable, Tuple

ITER_MAX_CG = 10000
ITER_MAX_GR = 1000
TOLERANCE_CG = np.float64(1e-10)
TOLERANCE_GR = np.float64(1e-10)


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
                        itr=ITER_MAX_GR, tol=TOLERANCE_GR  # Максимальное число итераций и точность
                        ) -> np.float64:
    if tol > TOLERANCE_CG:
        tol = TOLERANCE_CG
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


def cg(fun: Callable[[npt.NDArray[np.float64]], np.float64],  # Целевая функция для оптимизации
       grad: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],  # Ее градиент
       x0: npt.NDArray[np.float64], itr: int = ITER_MAX_GR,  # Нач точка и число итераций
       tol: np.float64 = TOLERANCE_CG  # Погрешность метода
       ) -> Tuple[npt.NDArray[np.float64], list]:
    # Одномерный алгоритм оптимизации использующий метод золотого сечения
    def goldenratio_fcg(x_f, d_f) -> np.float64:
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
        return golden_ratio_search(func_x, b_it-delt, a_it+delt)

    # Находим размерность алгоритма
    dim = x0.size
    x_it = x0
    convergence = [x0]
    # 1 шаг - вычисляем анти градиент функции в начальной точке
    r = -1 * grad(x_it)
    d = r
    iterations = 0
    while iterations < itr and np.sqrt(np.dot(r.T, r)) > tol:
        # 2 шаг - используя одномерную минимизацию, находим коэффициент a
        a = goldenratio_fcg(x_it, d)
        # 3 шаг - переход в точку, в которой функция принимает минимальное значение
        x_it = x_it + a * d
        # 4 шаг - вычисляем анти градиент в этой точке
        r_old = r
        r = -1 * grad(x_it)
        # 5 шаг - вычисляем коэффициенты алгоритма
        if iterations % (dim + 1) != 0 or iterations == 0:
            b = np.dot(r.T, r) / np.dot(r_old.T, r_old)
        else:
            b = 0
        # 6 шаг - вычисляем новое сопряженное направления
        d = r + b * d
        # Добавляем точку в массив, увеличиваем число итераций
        convergence.append(x_it)
        iterations += 1
    if iterations == ITER_MAX_CG:
        print(f'Алгоритм завершил работу, поскольку число итераций превысило {ITER_MAX_CG}')
    if np.sqrt(np.dot(r.T, r)) < tol:
        print(f'Алгоритм завершил работу, поскольку значение градиента '
              f'функции в найденной точке меньше чем {tol}')
    print(f'Число итераций - {iterations}')
    print(f'Найденное решение - {x_it}')
    print(f'Значение функции в точке минимума - {fun(x_it)}')
    return x_it, convergence


# Решаем задачу оптимизации
(sol, conv) = cg(himmelblau, gradient, np.array([2, -2]))
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
plt.xticks(range(len(eps_arr)))
plt.show()
