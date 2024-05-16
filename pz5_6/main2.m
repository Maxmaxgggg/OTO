%% Нахождение точки оптимума
% Задание системы уравнений 
% -x(1) - 2x(2) - 2x(3) <= 0
% x(1) + 2x(2) + 2x(3) <= 72
A = [-1 -2 -2; 1 2 2];
b = [0; 72];

% Стартовое значение
x0 = [10; 10; 10];
% fmincon - функция поиска минимума скалярной функции многих переменных 
% при наличии ограничений (она решает задачу нелинейного программирования).
% Находим точку оптимума
[x,fval] = fmincon('fun2', x0, A, b);