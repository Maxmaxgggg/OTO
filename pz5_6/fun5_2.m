function [f,g] = fun5_2(x)
f = -exp(-x(1)^2-x(2)^2);
if nargout > 1  % Проверка количества аргументов функции
g(1) = 2*x(1)*exp(-x(1)^2-x(2)^2);
g(2) = 2*x(2)*exp(-x(1)^2-x(2)^2);
end
