close all
%                        Функция fsolve
%   Функция fsolve возвращает решение системы нелинейных уравнений F(x) = 0


%% Решение задачи оптимизации
% Стартовое значение
x0 = [0.5; 0.5]; 
% Задание вывода информации на каждой итерации
options = optimset('Display','iter');
[x,fval] = fsolve('fun10',x0,options); % Поиск решения


%% Построение графиков
X1 = -4:0.1:30;
Y2 = -4:0.1:30;
Y1 = 2*X1 - exp(-X1);
X2 = 2*Y2 - exp(-Y2);
plot(X1,Y1,X2,Y2)
hold on
scatter(x(1),x(2), 'red','filled')
xlabel('Ось x')
ylabel('Ось y')
legend('y = 2x - exp(-x)', 'x = 2y - exp(-y)', 'Точка оптимума')