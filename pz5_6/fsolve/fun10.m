function [f1,f2] = fun10(x)
% Задаем систему нелинейных функций
f1 = 2*x(1)-x(2)-exp(-x(1));
f2 = -x(1)+2*x(2)-exp(-x(2));
end