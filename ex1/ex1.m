%% 1.一个简单的Octave/Matlab函数
% 要求：补充warmUpExercise()函数，生成一个5*5的单位矩阵

% 对应文件：warmUpExercise.m
% 添加代码如下：
A = eye(5);

% 执行代码如下：
warmUpExercise()

%% 2.单变量的线性回归  
%% 2.1 数据的可视化
% 要求：补充plotData()函数，将数据进行进行可视化

% 首先将数据导入，执行代码如下：
data = load('ex1data1.txt');  
X = data(:, 1); y = data(:, 2);  % 第一列为特征值，第二列为标签值，分别进行保存
m = length(y);

% 导入数据之后，将导入的数据进行可视化
% 对应文件：plotData.m
% 添加代码如下:
plot(x, y, 'rx', 'MarkerSize', 10)
ylabel('Profit in $10,000s'); % 设置X轴
xlabel('Population of City in 10,000s'); % 设置Y轴

% 将以上代码添加至文件中并保存后，对plotData()函数进行调用；
% 执行代码如下：
plotData(X, y);


%% 2.2 梯度下降
% 要求：实现损失函数、利用梯度下降算法找出最合适的参数theta

% 首先对初始数据进行设置，执行代码如下：
x = [ones(m, 1), data(:, 1)];    % 给X添加一个维度，方便后面矩阵乘法计算
theta = zeros(2, 1);     % 初始化两个参数为0
iterations = 1500;   % 迭代次数
alpha = 0.01;   % 学习率

% 损失函数实现
% 对应文件：computeCost.m
% 添加代码如下：
J = sum((X * theta - y).^2) / (2 * m); 

% 调用computeCost()函数验证正确性，执行代码如下：
computeCost(x, y, theta)
% 计算可得当theta0和theta1都为0时，损失值为32.0727

% 接着完善梯度下降函数
% 对应文件：gradientDescent.m
% 添加代码如下：
theta = theta - (alpha / m * sum((X * theta - y) .* X))';

% 利用补充好的梯度下降算法确定参数的值，执行代码如下：
% 返回值为最终theta值、每一次迭代后的损失值
[theta, J_history] = gradientDescent(x, y, theta, alpha, iterations);

% 利用得到的theta值进行预测，执行代码如下：
predict1 = [1, 3.5] * theta;
predict2 = [1, 7] * theta;

%% 2.3 调试并可视化
% 要求：将预测的直线画出来

% 先将原始数据集画出来，在其基础上画上线性回归的直线，执行代码如下：
plot(X, y, 'rx', 'MarkerSize', 10)
ylabel('Profit in $10,000s'); % 设置X轴
xlabel('Population of City in 10,000s'); % 设置Y轴
hold on;
plot(X, theta(1) + theta(2)*X, 'b-');   % 画线性回归直线
legend({'训练集','线性回归'},'Location','southeast')   % 添加图例

%% 2.4 可视化损失函数
% 要求：将损失函数进行可视化，变量是2维的（theta0和theta1），最终的梯度下降平面应该是一个三维面

% 生成一系列系数,执行代码如下：
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% 定义一个二维数组，用于存储所有损失值（此处为100*100），执行代码如下：
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% 计算每组的损失值并填充J_vals，执行代码如下：
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
        t = [theta0_vals(i); theta1_vals(j)];
        J_vals(i, j) = computeCost(x, y, t);
    end
end

% 对损失矩阵进行转置（surf的特性？），执行代码如下：
J_vals = J_vals';

% 画三维损失函数，执行函数如下：
figure;
surf(theta0_vals, theta1_vals, J_vals)   % 调用surf函数
xlabel('\theta_0'); ylabel('\theta_1');

% 画等高线，执行函数如下：
figure;
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold off;

%% 3.多变量线性回归
%% 3.1 特征归一化
% 要求：由于房子的大小和卧室的数量不在一个数量级，需要进行归一化
% 这样能够更加有利于梯度下降算法更快得到最优值

% 首先导入数据ex1data2.txt，执行代码如下：
clc; clear;  % 清除命令行和以上导入数据
data2 = load('ex1data2.txt');
X = data2(:, 1:2);
y = data2(:, 3);
m = length(y);

% 打印前10个数据点，执行代码如下：
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

% 可以看到房子面积和卧室数量相差很大，此时需要进行特征归一化
% 对应文件：featureNormalize.m
% 添加代码如下：
mu = mean(X);
sigma = std(X);
X_norm = (X_norm - mu) ./ sigma;

% 调用featureNormalize()函数对特征进行归一化，执行代码如下：
[X_norm, mu, sigma] = featureNormalize(X);

%% 3.2 多变量梯度下降

% 首先对初始数据进行赋值，执行代码如下：
x = [ones(m, 1), X_norm];
theta = zeros(3, 1);
alpha = 0.03;
iterations = 500;

% 首先进行多变量损失函数的完善
% 对应文件：computeCostMulti.m
% 计算方法和单变量一样，只不过换了一种矩阵乘法（更高效！）添加代码如下：
J = (X * theta - y)' * (X * theta - y)/(2 * m);

% 然后进行多变量梯度下降
% 对应文件：gradientDescentMulti.m
% 添加代码如下：
theta = theta - (alpha / m) * (X' * (X * theta - y));

% 利用补充好的梯度下降算法确定参数的值，执行代码如下：
% 返回值为最终theta值、每一次迭代后的损失值,并进行可视化
% 通过不断调整学习率和迭代次数的值，找到最适合的值
theta = zeros(3, 1);
alpha = 0.01;
iterations = 1500;
[theta, J_history] = gradientDescentMulti(x, y, theta, alpha, iterations);
% 画出损失值
plot(1:1500, J_history(1:1500), 'b')

% 使用得到数据进行预测房价
norm_predict = ([1650, 3] - mu) ./ sigma;    % 先进行归一化
predict = [1, norm_predict] * theta

%% 3.3 正规方程

% 利用正规方程直接计算得出参数值
% 对应文件：normalEqn.m
% 添加代码如下：
theta = pinv(X' * X) * X' * y;

% 调用正规方程计算参数theta的值
theta = normalEqn(x, y);
% 进行预测
predict = [1, norm_predict] * theta

