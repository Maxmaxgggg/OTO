import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Callable, Tuple, List

DIMENSIONALITY = 2  # Размерность задачи
REDUCTION_FACTOR = np.float64(0.5)  # Коэффициент редукции
SIMPLEX_START_LENGTH = 1  # Начальный размер симплекса
# Max число итераций, при которых вершина симплекса не меняется
M = int(1.65 * DIMENSIONALITY + 0.05 * DIMENSIONALITY ** REDUCTION_FACTOR)
ITER_MAX = 10000  # Максимальное число итераций алгоритма
TOLERANCE = np.float64(1e-5)  # Точность алгоритма


# Определяем квадратичную функцию Химмельблау для оптимизации
def himmelblau(x: npt.NDArray[np.float64]) -> np.float64:
    if len(x) != 2:
        raise ValueError("Массив должен состоять из 2-х элементов")
    return np.float64((x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2)


# Определяем квадратичную функцию Химмельблау для построения графиков
def him_pl(x: float, y: float) -> float:
    return (x**2+y-11)**2+(x+y**2-7)**2


# Определяем класс двумерный симплекс
class Simplex:
    # Атрибуты класса - список вершин и список итераций вершин
    vertices:  List[npt.NDArray[np.float64]]
    vert_iter: List[int]

    # Устанавливаем начальное число итераций каждой вершины
    def __init__(self, vertices: list):
        if len(vertices) != DIMENSIONALITY+1:
            raise ValueError('Неверное количество точек симплекса')
        self.vertices = vertices
        self.vert_iter = [0] * (DIMENSIONALITY + 1)

    # Метод установки значения вершины.
    # Устанавливаем значение 'x' у вершины с индексом num,
    # у других вершин увеличиваем число итераций на 1
    def set(self, num: int, x: np.ndarray
            ) -> None:
        self.vertices[num] = x
        self.vert_iter[num] = 0
        # Увеличиваем все индексы, кроме индекса измененного элемента
        for i in range(len(self.vert_iter)):
            if i != num:
                self.vert_iter[i] += 1

    # Метод поиска максимальной вершины симплекса
    def vert_max(self, fun: Callable[[npt.NDArray[np.float64]], np.float64]
                 ) -> list:
        fun_values = [fun(x) for x in self.vertices]
        index = fun_values.index(np.max(fun_values))
        return [index, self.vertices[index]]

    # Метод поиска минимальной вершины симплекса
    def vert_min(self, fun: Callable[[npt.NDArray[np.float64]], np.float64]
                 ) -> list:
        fun_values = [fun(x) for x in self.vertices]
        index = fun_values.index(np.min(fun_values))
        return [index, self.vertices[index]]

    # Метод поиска центра массы двух точек симплекса (Если аргумент равен -1, то метод ищет центр симплекса)
    def center(self, vert_num: int) -> npt.NDArray[np.float64]:
        x_center = self.vertices[0].copy()
        for i in range(1, len(self.vertices)):
            x_center += self.vertices[i]
        if vert_num == -1:
            x_center /= (DIMENSIONALITY+1)
            return x_center
        x_center -= self.vertices[vert_num]
        x_center /= DIMENSIONALITY
        return x_center

    # Метод отражения вершины симплекса
    def reflect_vert(self, vert_num: int) -> npt.NDArray[np.float64]:
        self.set(vert_num, 2 * self.center(vert_num) - self.vertices[vert_num])
        return self.vertices[vert_num]

    # Метод редукции симплекса к наилучшей вершине
    def reduce(self, coef: np.float64, fun) -> None:
        ind_min = self.vert_min(fun)[0]
        v_min = self.vert_min(fun)[1]
        for i in range(len(self.vertices)):
            if i != ind_min:
                self.vertices[i] = v_min + coef*(self.vertices[i]-v_min)
                self.vert_iter[i] = 0


# Минимизация функции с использованием симплекс-метода
def simplex_search(fun: Callable[[npt.NDArray[np.float64]], np.float64],
                   x0: npt.NDArray[np.float64], itr: int = ITER_MAX, tol: np.float64 = TOLERANCE
                   ) -> Tuple[npt.NDArray[np.float64], list, list]:
    # Коэффициенты для алгоритма
    d1 = np.float64((np.sqrt(DIMENSIONALITY+1)+DIMENSIONALITY+1)/(DIMENSIONALITY*np.sqrt(2))*SIMPLEX_START_LENGTH)
    d2 = np.float64((np.sqrt(DIMENSIONALITY+1)-1)/(DIMENSIONALITY*np.sqrt(2))*SIMPLEX_START_LENGTH)

    # Вычисляем нулевой симплекс
    simplex = Simplex([x0, x0 + np.array([d2, d1]), x0 + np.array([d1, d2])])
    it = 0
    # Создаем список, содержащий центры всех симплексов
    center_arr = [simplex.center(-1)]
    # Создаем список, содержащий все симплексы
    convergence = [[np.append(vert, fun(vert)) for vert in simplex.vertices]]
    while it < itr:
        # Переворачиваем симплекс
        [v_ind, xprev] = simplex.vert_max(fun)
        it_prev = simplex.vert_iter[v_ind]
        funprev = fun(xprev)
        fun_cur = fun(simplex.reflect_vert(simplex.vert_max(fun)[0]))

        #           Случай накрытия симплексом точки минимума
        if fun_cur > funprev:
            vertices = 1  # Число проверенных вершин симплекса
            simplex.set(v_ind, xprev)  # Возвращаемся к предыдущему симплексу
            simplex.vert_iter[v_ind] = it_prev
            while True:
                v_ind += 1  # Индекс вершины, которую будем поворачивать
                # Если индекс превышает максимальный, устанавливаем его на первую вершину
                if v_ind == DIMENSIONALITY + 1:
                    v_ind = 0
                # Сохраняем предыдущее состояние вершины
                xprev = simplex.vertices[v_ind]
                it_prev = simplex.vert_iter[v_ind]
                # И значение функции от предыдущего состояния вершины
                funprev = fun(xprev)
                # Вращаем вершину
                fun_cur = fun(simplex.reflect_vert(v_ind))
                # Увеличиваем число проверенных вершин
                vertices += 1
                # Если уменьшили значение целевой функции, то прерываем цикл
                if fun_cur < funprev:
                    break
                # В ином случае возвращаем старое значение вершины
                simplex.set(v_ind, xprev)
                simplex.vert_iter[v_ind] = it_prev
                # Если перебрали все вершины, то уменьшаем симплекс
                if vertices == DIMENSIONALITY + 1:
                    simplex.reduce(REDUCTION_FACTOR, fun)
                    break

        #           Случай циклического движения
        if M in simplex.vert_iter:
            simplex.reduce(REDUCTION_FACTOR, fun)
        it += 1
        convergence.append([np.append(vert, fun(vert)) for vert in simplex.vertices])
        center_arr.append(simplex.center(-1))
        # Если разница значений функции между максимальной и минимальной вершинами меньше tol
        if abs(fun(simplex.vert_min(fun)[1])-fun(simplex.vert_max(fun)[1])) < tol:
            break
    opt_point = simplex.vert_min(fun)[1]
    return opt_point, convergence, center_arr


x0 = np.array([0, 0], dtype=np.float64)
[opt_p, conv, c_arr] = simplex_search(himmelblau, x0, ITER_MAX, np.float64(1e-10))

#           Создаем трехмерный график
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
# Создаем сетку
x_lin = np.linspace(-5, 5, 100)
y_lin = np.linspace(-5, 5, 100)
x_lin, y_lin = np.meshgrid(x_lin, y_lin)

# Устанавливаем лимиты осей
ax.set_xlim([-6, 6])
ax.set_ylim([-6, 6])
ax.set_zlim([0, 1500])
plt.title('Функция Химмельблау и симплексы')
# Устанавливаем метки осей
ax.set_xlabel('Ось X')
ax.set_ylabel('Ось Y')
ax.set_zlabel('Ось Z')
# Добавляем начальную точку
ax.scatter(x0[0], x0[1], himmelblau(x0), c='b', marker='o')
ax.text(x0[0], x0[1], himmelblau(x0), 'Начальная точка', color='black')
# Добавляем найденную точку оптимума
ax.scatter(opt_p[0], opt_p[1], himmelblau(opt_p), c='r', marker='o')
ax.text(opt_p[0], opt_p[1], himmelblau(opt_p), "Точка оптимума", color='black')
# Вычисляем z на основе функции f
z = him_pl(x_lin, y_lin)
# Рисуем поверхность
surface = ax.plot_surface(x_lin, y_lin, z, cmap='viridis', alpha=0.4)
# Добавляем все симплексы на график
for simp in conv:
    faces = [[simp[0], simp[1], simp[2]]]
    poly3d = Poly3DCollection(faces, facecolors='green', linewidths=1, edgecolors='k', alpha=0.5)
    ax.add_collection3d(poly3d)
plt.show()


# Создаем график изолиний функции Химмельблау с отображением симплексов
fig = plt.figure(2)
ax = fig.add_subplot()
x_lin = np.linspace(-5, 5, 400)
y_lin = np.linspace(-5, 5, 400)
x_lin, y_lin = np.meshgrid(x_lin, y_lin)
z = him_pl(x_lin, y_lin)
ax.contour(x_lin, y_lin, z, levels=np.logspace(0, 5, 35))

# Отображаем симплексы и подписываем центры симплексов
for i, simp in enumerate(conv):
    simp_array = np.array([vert[:2] for vert in simp])
    ax.plot(*zip(*simp_array, simp_array[0]), 'g')

# Отображаем траекторию центров симплексов и подписываем их
for i, center in enumerate(c_arr):
    ax.plot(center[0], center[1], 'bo')
    ax.text(center[0], center[1], str(i), color='black')


plt.title('Изолинии функции Химмельблау с симплексами')
plt.xlabel('Ось X')
plt.ylabel('Ось Y')
plt.show()

#           Создаем график сходимости алгоритма
fig = plt.figure(3)
d_arr = [np.linalg.norm(c_arr[i]-c_arr[i+1]) for i in range(len(c_arr)-1)]
plt.plot(d_arr)
plt.title('График сходимости алгоритма')
plt.xlabel('Итерация')
plt.ylabel('| S.center[i| - S.center[i+1] |')
plt.yscale('log')
plt.xticks(range(len(d_arr)))
plt.show()


