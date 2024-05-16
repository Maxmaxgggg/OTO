close all

%% Исходные данные для программы
sigma = 0.1;  % Среднеквардратичное отклонение
nu = 0; % Среднее значение шума
xdata = 0:0.01:10; % Исходный массив иксов
ydata_s = sin_wn(xdata, sigma, nu); % Массив значений зашумленного синуса
ydata = ydata_s; % Сохраняем первый массив для дальнейшего построения графика
sigma_values = 0.1:0.01:1; % Значения sigma для построения графика зависимости ско от дисперсии
sko_values = [];
x0 = [1,1,0]; % Начальная точка для работы алгоритма


%% Фиттинг кривой
fun = @(x,xdata)x(1) * sin(x(2) * xdata) + x(3);
x = lsqcurvefit(fun,x0,xdata,ydata);

%% Рассчет значений вектора СКО
for i = 1:length(sigma_values)

    sigma = sigma_values(i);
    ydata = sin_wn(xdata, sigma, nu);
    
    % Функция для фиттинга
    fun = @(x,xdata) x(1)*sin(x(2)*xdata) + x(3);
    
    % Фиттинг кривой
    x_fit = lsqcurvefit(fun, x0, xdata, ydata);
    
    % Вычисление СКО
    sko = sum((ydata - fun(x_fit, xdata)).^2);
    sko_values(i) = sko;
end


%% Построение графиков
% График исходных значений и полученной кривой
figure;
plot(xdata(1:10:end), ydata_s(1:10:end), 'o', ...
    'MarkerSize', 8, 'MarkerFaceColor', 'none', ...
    'MarkerEdgeColor', 'black');
hold on; 
plot(xdata, fun(x,xdata), 'Color','red');
hold off;
xlabel('x');
ylabel('y');
title('График данных и функции');
legend('Точки исходных данных', 'Полученная синусоида');
grid on;


% Построение графика зависимости СКО от 1/D
figure;
plot(1./sigma_values, sko_values,  'LineWidth', 1.5);
xlabel('Значение 1/D');
ylabel('СКО');
title('Зависимость СКО от 1/D');
grid on;