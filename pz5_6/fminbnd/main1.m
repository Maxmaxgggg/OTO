close all;

%                    Функция fminbnd
%   fminbnd(fun, x1, x2) – функция скалярной нелинейной минимизации с ограничениями (x1 < х < х2). 
%   Алгоритм базируется на методе золотого сечения и квадратичной интерполяции

%% Определение целевой функции
polynom = @(x) (x-1).^2;

%% Поиск точки оптимума
x = fminbnd(polynom, -2, 2);

%% Построение графиков
plot(-2:0.1:2, polynom(-2:0.1:2))
hold on
plot(x,polynom(x), 'ro', 'MarkerSize', 10)
title("График целевой функции и точки оптимума")
legend('Целевая функция', 'Точка оптимума целевой функции')
