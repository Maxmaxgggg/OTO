clc

%                       Функция mldivide
%   Функция обратного деления матриц применяется при решении 
%   систем линейных уравнений вида Сх = d при числе уравнений, 
%   равном числу неизвестных

%% Задание системы уравнений
C = [1 1; -1 2; 2 1];
d = [2; 2; 3];

%% Решение системы уравнений
x = C\d;

%% Вывод найденного решения в консоль
disp('Решение системы уравнений:')
disp(['x(1) = ', num2str(x(1))])
disp(['x(2) = ', num2str(x(2))])
