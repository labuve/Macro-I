%% Macro I, Problem Set 3
% Fatima & Johanna 

clear all
clc

%% Exercise 1

% Parameters
n = 10;             % number of draws
t = 51;             % time
x_0_hat = 100;      % initial guess of theta: mean
sigma_0 = 10;       % initial guess of theta: variance
R = 100;            % variance of error term in signal eq

% Generating random draws
theta = x_0_hat + sqrt(sigma_0).*randn(n, 1);
v = sqrt(R).*randn(n,t);

% Initialize path matrices

x_hat = zeros(n,t);     % initialize matrix for IQ_t
x_hat(:, 1) = x_0_hat;  % start at IQ_0
sigma = zeros(n,t);     % initialize matrix Sigma_t
sigma(:,1) = sigma_0;   % start at Sigma_0

y = theta* ones(1,t) + v;   % Signal variable: test scores

for i = 1:n     % for each random draw
    for j = 2:t % for each time
        x_hat(i,j) = (sigma(i,j-1)/(R + sigma(i,j-1))) * y(i,j-1) + ...
            (R/(R + sigma(i,j-1))) * x_hat(i,j-1);
        sigma(i,j) = (R*sigma(i,j-1)/(R + sigma(i,j-1)));
    end 
end

figure (1)
plot(1:t, x_hat(1,:), 'b-', 1:t, x_hat(2,:), 'r-', ...
    1:t, theta(1)*ones(1,t), 'b--', 1:t, theta(2)*ones(1, t),'r--')
xlabel('Time periods')
ylabel('Intelligence')
title('Simulation of Kalman filter')
legend('IQ(1)', 'IQ(2)', '\theta(1)', '\theta(2)', 'Location', 'Best')
xlim([1 t])

%% Exercise 2
clear all

excelDates = xlsread('A191RL1A225NBEA.xls', 'A2:B69');
matlabDates = datenum('30-Dec-1899') + excelDates(:,1);
mydata = matlabDates + zeros(length(excelDates),2);
mydata(:, 2) = excelDates(:,2);

% Part 1
% Plotting the actual GDP growth rate
figure(2)
plot(mydata(:,1), mydata(:,2))
dateFormat=10;
datetick('x', dateFormat)
xlabel('Time')
title('Observed GDP growth rate')
ylabel('Annual percentage change')
xlim([711128, 735600])

% Part 2

% Parameters
t = length(mydata);                 % time
x_hat_zero = 2;                     % initial guess: mean
sigma_0 = 2;                        % initial guess: variance
sigma_nu = [0.01, 0.01, 0.0001];    % variance of error term in state eq
sigma_eps = [0.01, 0.0001, 0.01];   % variance of error term in signal eq

% Initialize path matrices
x_hat = zeros(t, length(sigma_nu));
x_hat(1,:) = x_hat_zero;
sigma = zeros(t, length(sigma_nu));
sigma(1,:) = sigma_0;

for i = 1:length(sigma_nu)
    for j = 2:t
        x_hat(j,i)=(sigma(j-1,i)/(sigma_eps(i)+sigma(j-1,i)))*mydata(j-1,2)...
            + (sigma_eps(i)/(sigma_eps(i) + sigma(j-1,i))) * x_hat(j-1,i);
        sigma(j,i) = sigma_nu(i) + (sigma_eps(i) * sigma(j-1, i)/...
            (sigma_eps(i) + sigma(j-1,i)));
    end
end

figure(3)
plot(mydata(:,1), mydata(:,2), mydata(:,1), x_hat(:,1), ...
    mydata(:,1), x_hat(:,2), mydata(:,1), x_hat(:,3))
dateFormat=10;
datetick('x', dateFormat)
xlabel('Time')
title('GDP growth rate')
ylabel('Annual percentage change')
legend('Observed', '\sigma_\epsilon^2 = \sigma_\nu^2 = 10^{-2}',...
    '\sigma_\epsilon^2 = 10^{-4}, \sigma_\nu^2 = 10^{-2}', ...
    '\sigma_\epsilon^2 = 10^{-2}, \sigma_\nu^2 = 10^{-4}', 'Location', 'Best')
xlim([711128, 735600])
