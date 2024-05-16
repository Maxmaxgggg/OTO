close all
%% Нахождение точки оптимума
options = optimset('GradObj','on');
% Начальное значение
x0 = [1.6,2.2];
% Нахождение точки оптимума без задания информации о градиенте
[xmin1,fval1] = fminunc('fun5_1',x0);
% Нахождение точки оптимума с заданием информации о градиенте
% При ручном задании градиента получаем более точное значение
[xmin2,fval2] = fminunc('fun5_2',x0,options);

%% Построение графиков
[x, y] = meshgrid(-10:0.1:10);
z = -exp(-x.^2 - y.^2);
surf(x, y, z);
hold on
% Построение начальной точки
scatter3(x0(1), x0(2), fun5_2(x0), 'ro', 'filled', 'MarkerEdgeColor', 'k');

% Построение найденной точки минимума
scatter3(xmin1(1), xmin1(2), fun5_2(xmin1), 'go', 'filled', 'MarkerEdgeColor', 'k');

% Оформление графиков
xlabel('x');
ylabel('y');
zlabel('-exp(-x^2-y^2)');
title('График функции -exp(-x^2-y^2)');
legend('Функция -exp(-x^2-y^2)', 'Начальная точка', 'Точка минимума', 'Location', 'northwest');