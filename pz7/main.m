close all

%% �������� ������ ��� ���������
sigma = 0.1;  % ������������������� ����������
nu = 0; % ������� �������� ����
xdata = 0:0.01:10; % �������� ������ �����
ydata_s = sin_wn(xdata, sigma, nu); % ������ �������� ������������ ������
ydata = ydata_s; % ��������� ������ ������ ��� ����������� ���������� �������
sigma_values = 0.1:0.01:1; % �������� sigma ��� ���������� ������� ����������� ��� �� ���������
sko_values = [];
x0 = [1,1,0]; % ��������� ����� ��� ������ ���������


%% ������� ������
fun = @(x,xdata)x(1) * sin(x(2) * xdata) + x(3);
x = lsqcurvefit(fun,x0,xdata,ydata);

%% ������� �������� ������� ���
for i = 1:length(sigma_values)

    sigma = sigma_values(i);
    ydata = sin_wn(xdata, sigma, nu);
    
    % ������� ��� ��������
    fun = @(x,xdata) x(1)*sin(x(2)*xdata) + x(3);
    
    % ������� ������
    x_fit = lsqcurvefit(fun, x0, xdata, ydata);
    
    % ���������� ���
    sko = sum((ydata - fun(x_fit, xdata)).^2);
    sko_values(i) = sko;
end


%% ���������� ��������
% ������ �������� �������� � ���������� ������
figure;
plot(xdata(1:10:end), ydata_s(1:10:end), 'o', ...
    'MarkerSize', 8, 'MarkerFaceColor', 'none', ...
    'MarkerEdgeColor', 'black');
hold on; 
plot(xdata, fun(x,xdata), 'Color','red');
hold off;
xlabel('x');
ylabel('y');
title('������ ������ � �������');
legend('����� �������� ������', '���������� ���������');
grid on;


% ���������� ������� ����������� ��� �� 1/D
figure;
plot(1./sigma_values, sko_values,  'LineWidth', 1.5);
xlabel('�������� 1/D');
ylabel('���');
title('����������� ��� �� 1/D');
grid on;