close all

%                      Функция fminsearch
%   Находим минимум функции нескольких переменных без ограничений 
%   с использованием симплексного метода

%% Определение целевой функции
% Двумерный кардинальный синус
fun = @(x)-sinc(hypot(x(1),x(2)));

%% Поиск точки оптимума функции двух переменных

% Из-за выбора начальной точки на спаде находим глобальный минимум
x0 = [-0.15, 1];
% Находим точку оптимума
[xmin, fval] = fminsearch(fun, x0);

%% Построение графика
[x, y] = meshgrid(-10:0.1:10); % Создание сетки точек для x и y
z = -sinc(hypot(x,y));   % Вычисление значений функции sinc(x, y)
surf(x, y, z);
hold on
% Построение начальной точки
scatter3(x0(1), x0(2), fun(x0), 'ro', 'filled', 'MarkerEdgeColor', 'k');
% Построение найденной точки минимума
scatter3(xmin(1), xmin(2), fval, 'go', 'filled', 'MarkerEdgeColor', 'k');
xlabel('x');
ylabel('y');
zlabel('sinc(x, y)');
title('График функции sinc(x, y)');
% Добавление легенды с подписями для точек
legend('Функция sinc(x, y)', 'Начальная точка', 'Точка минимума', 'Location', 'northwest');