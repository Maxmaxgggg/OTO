clc

%                   Функция lsqlin
%   Функция lsqlin возвращает решение задачи линейного МНК при наличии ограничений

%% Задаем уравнения
C = [
0.9501	0.7620	0.6153	0.4057
0.2311	0.4564	0.7919	0.9354
0.6068	0.0185	0.9218	0.9169
0.4859	0.8214	0.7382	0.4102
0.8912	0.4447	0.1762	0.8936];

d = [
    0.0578
    0.3528
    0.8131
    0.0098
    0.1388];

%% Задаем ограничения
A = [
0.2027	0.2721	0.7467	0.4659
0.1987	0.1988	0.4450	0.4186
0.6037	0.0152	0.9318	0.8462];

b = [
    0.5251
    0.2026
    0.6721];

lb = -0.1*ones(4,1);
ub = 2*ones(4,1);

%% Решаем задачу оптимизации
% В выводе в консоль присутствуют два новых пункта:
% resnorm – сумма квадратов невязок  (Cx-d)^2
% residual – вектор невязок residual(i) = (Cx(i)-d(i))

[x,resnorm,residual,exitflag,output,lambda] = lsqlin (C,d,A,b,[],[],lb,ub)
