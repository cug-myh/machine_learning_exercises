%% 1.һ���򵥵�Octave/Matlab����
% Ҫ�󣺲���warmUpExercise()����������һ��5*5�ĵ�λ����

% ��Ӧ�ļ���warmUpExercise.m
% ��Ӵ������£�
A = eye(5);

% ִ�д������£�
warmUpExercise()

%% 2.�����������Իع�  
%% 2.1 ���ݵĿ��ӻ�
% Ҫ�󣺲���plotData()�����������ݽ��н��п��ӻ�

% ���Ƚ����ݵ��룬ִ�д������£�
data = load('ex1data1.txt');  
X = data(:, 1); y = data(:, 2);  % ��һ��Ϊ����ֵ���ڶ���Ϊ��ǩֵ���ֱ���б���
m = length(y);

% ��������֮�󣬽���������ݽ��п��ӻ�
% ��Ӧ�ļ���plotData.m
% ��Ӵ�������:
plot(x, y, 'rx', 'MarkerSize', 10)
ylabel('Profit in $10,000s'); % ����X��
xlabel('Population of City in 10,000s'); % ����Y��

% �����ϴ���������ļ��в�����󣬶�plotData()�������е��ã�
% ִ�д������£�
plotData(X, y);


%% 2.2 �ݶ��½�
% Ҫ��ʵ����ʧ�����������ݶ��½��㷨�ҳ�����ʵĲ���theta

% ���ȶԳ�ʼ���ݽ������ã�ִ�д������£�
x = [ones(m, 1), data(:, 1)];    % ��X���һ��ά�ȣ�����������˷�����
theta = zeros(2, 1);     % ��ʼ����������Ϊ0
iterations = 1500;   % ��������
alpha = 0.01;   % ѧϰ��

% ��ʧ����ʵ��
% ��Ӧ�ļ���computeCost.m
% ��Ӵ������£�
J = sum((X * theta - y).^2) / (2 * m); 

% ����computeCost()������֤��ȷ�ԣ�ִ�д������£�
computeCost(x, y, theta)
% ����ɵõ�theta0��theta1��Ϊ0ʱ����ʧֵΪ32.0727

% ���������ݶ��½�����
% ��Ӧ�ļ���gradientDescent.m
% ��Ӵ������£�
theta = theta - (alpha / m * sum((X * theta - y) .* X))';

% ���ò���õ��ݶ��½��㷨ȷ��������ֵ��ִ�д������£�
% ����ֵΪ����thetaֵ��ÿһ�ε��������ʧֵ
[theta, J_history] = gradientDescent(x, y, theta, alpha, iterations);

% ���õõ���thetaֵ����Ԥ�⣬ִ�д������£�
predict1 = [1, 3.5] * theta;
predict2 = [1, 7] * theta;

%% 2.3 ���Բ����ӻ�
% Ҫ�󣺽�Ԥ���ֱ�߻�����

% �Ƚ�ԭʼ���ݼ�����������������ϻ������Իع��ֱ�ߣ�ִ�д������£�
plot(X, y, 'rx', 'MarkerSize', 10)
ylabel('Profit in $10,000s'); % ����X��
xlabel('Population of City in 10,000s'); % ����Y��
hold on;
plot(X, theta(1) + theta(2)*X, 'b-');   % �����Իع�ֱ��
legend({'ѵ����','���Իع�'},'Location','southeast')   % ���ͼ��

%% 2.4 ���ӻ���ʧ����
% Ҫ�󣺽���ʧ�������п��ӻ���������2ά�ģ�theta0��theta1�������յ��ݶ��½�ƽ��Ӧ����һ����ά��

% ����һϵ��ϵ��,ִ�д������£�
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% ����һ����ά���飬���ڴ洢������ʧֵ���˴�Ϊ100*100����ִ�д������£�
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% ����ÿ�����ʧֵ�����J_vals��ִ�д������£�
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
        t = [theta0_vals(i); theta1_vals(j)];
        J_vals(i, j) = computeCost(x, y, t);
    end
end

% ����ʧ�������ת�ã�surf�����ԣ�����ִ�д������£�
J_vals = J_vals';

% ����ά��ʧ������ִ�к������£�
figure;
surf(theta0_vals, theta1_vals, J_vals)   % ����surf����
xlabel('\theta_0'); ylabel('\theta_1');

% ���ȸ��ߣ�ִ�к������£�
figure;
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold off;

%% 3.��������Իع�
%% 3.1 ������һ��
% Ҫ�����ڷ��ӵĴ�С�����ҵ���������һ������������Ҫ���й�һ��
% �����ܹ������������ݶ��½��㷨����õ�����ֵ

% ���ȵ�������ex1data2.txt��ִ�д������£�
clc; clear;  % ��������к����ϵ�������
data2 = load('ex1data2.txt');
X = data2(:, 1:2);
y = data2(:, 3);
m = length(y);

% ��ӡǰ10�����ݵ㣬ִ�д������£�
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

% ���Կ�����������������������ܴ󣬴�ʱ��Ҫ����������һ��
% ��Ӧ�ļ���featureNormalize.m
% ��Ӵ������£�
mu = mean(X);
sigma = std(X);
X_norm = (X_norm - mu) ./ sigma;

% ����featureNormalize()�������������й�һ����ִ�д������£�
[X_norm, mu, sigma] = featureNormalize(X);

%% 3.2 ������ݶ��½�

% ���ȶԳ�ʼ���ݽ��и�ֵ��ִ�д������£�
x = [ones(m, 1), X_norm];
theta = zeros(3, 1);
alpha = 0.03;
iterations = 500;

% ���Ƚ��ж������ʧ����������
% ��Ӧ�ļ���computeCostMulti.m
% ���㷽���͵�����һ����ֻ��������һ�־���˷�������Ч������Ӵ������£�
J = (X * theta - y)' * (X * theta - y)/(2 * m);

% Ȼ����ж�����ݶ��½�
% ��Ӧ�ļ���gradientDescentMulti.m
% ��Ӵ������£�
theta = theta - (alpha / m) * (X' * (X * theta - y));

% ���ò���õ��ݶ��½��㷨ȷ��������ֵ��ִ�д������£�
% ����ֵΪ����thetaֵ��ÿһ�ε��������ʧֵ,�����п��ӻ�
% ͨ�����ϵ���ѧϰ�ʺ͵���������ֵ���ҵ����ʺϵ�ֵ
theta = zeros(3, 1);
alpha = 0.01;
iterations = 1500;
[theta, J_history] = gradientDescentMulti(x, y, theta, alpha, iterations);
% ������ʧֵ
plot(1:1500, J_history(1:1500), 'b')

% ʹ�õõ����ݽ���Ԥ�ⷿ��
norm_predict = ([1650, 3] - mu) ./ sigma;    % �Ƚ��й�һ��
predict = [1, norm_predict] * theta

%% 3.3 ���淽��

% �������淽��ֱ�Ӽ���ó�����ֵ
% ��Ӧ�ļ���normalEqn.m
% ��Ӵ������£�
theta = pinv(X' * X) * X' * y;

% �������淽�̼������theta��ֵ
theta = normalEqn(x, y);
% ����Ԥ��
predict = [1, norm_predict] * theta

