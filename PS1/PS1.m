%% Problem Set I

% Nurfatima Jandarova

clear all
clc

%% Question 2.1.2

% Assumptions
beta = 0.99; % patience
delta = 0.02; % depreciation rate
T = 20; % number of time periods
x0 = 1; % initial state
N = 100; % size of state grid

xmin = 0; % lower bound for possible values of x
xmax = 1; % upper bound for possible values of x

x_opt = zeros(1, T+2); % vector for optimal path
c_opt = zeros(1, T+2);
error = 100; % iteration error
tol = 10^-5; % tolerance level

a = (1-delta)*(1+beta); % coefficients in EE
b = beta*(1-delta)^2; % coefficients in EE

while abs(error) > tol
    x_opt(T+1)=0.5*xmin+0.5*xmax; % initial guess
    for t = 1:T
        x_opt(T+1-t)=(a/b)*x_opt(T+2-t)-(1/b)*x_opt(T+3-t); % EE
    end
    error = x_opt(1)-x0;
    if error > 0
        xmax = x_opt(T+1);
    else
        xmin = x_opt(T+1);
    end
    disp(['Current evaluation: ', num2str(error)]);
end

time = linspace(0, T+1, T+2); 

figure(1)
plot(time, x_opt)
xlabel('Time')
ylabel('Optimal path for x_t')

%% Question 2.2.2

xs_opt = ones(1,T+2); % vector for optimal x

for t=1:T+1
    v = beta*ones(1,T+1-t); 
    vsum = 1+sum(cumprod(v));
    xs_opt(t+1)=(1-delta)*xs_opt(t)*(vsum-1)/vsum;
end

figure(2)
plot(time, x_opt, 'r-', time, xs_opt, 'b*')
xlabel('Time')
ylabel('Optimal path for x_t')
legend('From Euler equation', 'From value function iteration')

% It is easy to see that two methods provide exactly the same optimal path
% for x_t.

%% Question 3.3

clear all

% Assumptions
beta = 0.95; % patience
delta = 0.025; % depreciation rate
alpha = 0.36; % capital share
ro = 2; % degree of relative risk aversion
k0 = 2; % starting level of assets
T = 1000; % number of time periods
N = 100; % number of capital grid-points
tolv = 1e-5; % tolerance for value function iteration

% capital grid
k_grid = linspace(0, 6*k0, N);

% flow utility matrix
u = zeros(N,N);
c = zeros(N,N);

for i = 1:N % current capital
  for j = 1:N % future capital
    c(i,j) = k_grid(i)^alpha + (1-delta)*k_grid(i) - k_grid(j);
    if c(i,j) > 0
      % if consumption is positive (feasible)
      u(i,j) = (c(i,j)^(1-ro))/(1-ro);
    else % consumption is not feasible
      u(i,j) = NaN; % will never be chosen
    end
  end
end

% Initialize the value function
V_0 = zeros(T+1,N); % +1 to ensure that value function after the world ends is zero
V_1 = zeros(T+1,N);
% Initialize the optimal capital choice
k_opt = zeros(T,N);

% Value Function iteration
err = 1;
while err > tolv
  for t = 1:T % time periods
    for i = 1:N % current capital
      % calculate V(t,k) for each possible k'
      V_aux = zeros(1,N);
      for j = 1:N % future capital
        V_aux(j) = u(i,j) + beta*V_0(t+1,j);
      end
      % choose k' that maximizes V(t,k)
      [V_1(t,i),k_opt(t,i)] = max(V_aux);
    end
  end
  % calculate the error term
  err = abs(max(max(V_0 - V_1)));
  disp(['Current error: ', num2str(err)]);
  % update value function
  V_0 = V_1;
end

[k_aux ind0] = min(abs(k_grid-k0));
clear k_aux
ind1 = 0;
k_path = zeros(1, T+1);
k_path(1)=k_grid(ind0);
for i = 2:T+1
    ind1 = k_opt(i-1, ind0);
    k_path(i)=k_grid(ind1);
    ind0=ind1;
end

time = linspace(0, T, T+1);

figure(3)
plot(time, k_path)
xlabel('Time')
ylabel('Optimal path for capital')

figure(4)
plot(1:N, V_1(T-2,:), 'r-', 1:N, V_1(T-1,:), 'b*')
title('Value functions at T-1 and T-2')
legend('Value function at T-2', 'Value function at T-1', 'Location', 'best')

%% Question 4.1

% Simulating Markov chain process with 3 nodes for 200 periods
clear all

% Parameters
rho = 0.8; % AR(1) coefficient
sigma = 0.06; % standard deviation of white noise
z0 = 1; % initial value
mu = 0; % unconditional mean
m = 3; % max +- std. devs
N = 3; % number of grid points

% Parameters
T = 200; % time periods

% Get the discretized state vector and transition matrix
[theta Pi] = tauchen(N,mu,rho,sigma,m);

% number of states
num = size(theta);
num = num(1);

% initialize path
s_path = zeros(1,T);
s_path(1) = ceil(N/2); % initialize starting state
% Assuming that $z_0$ is our assumption for $a_0$, then $\ln(a_0)=0$=0. 
% Recall as well that Tauchen algorithm will return the vector of states 
% centered around zero. Therefore, with odd number of grid points it is 
% sufficient to find the center element in the vector of the states.

% make random draws
draw_path = rand(1,T);

% simulate the path (in indices)
for t = 2:T
  % select random draw
  draw = draw_path(t);
  % transform transition probabilities into cumulative sums
  c_sum = cumsum(Pi(s_path(t-1),:));

  % select the next state (smallest element of c_sum that is strictly
  % greater than the random draw)
  % short version
  s_path(t) = min(find(c_sum > draw));
end

% transform index path to path in states
s_path = theta(s_path);

% Sample mean and variance 
s_path_mean = mean(s_path);
disp(['Sample mean = ', num2str(s_path_mean)]);

s_path_var = var(s_path);
disp(['Sample variance = ', num2str(s_path_var)]);

% Plot the resulting path
figure(5)
plot(1:T,s_path)
xlabel('Periods')
ylabel('log(a_{t+1})')


%% Question 4.2

% Simulating Markov chain process with 3 nodes for 200 periods
clear all

% Parameters
rho = 0.8; % AR(1) coefficient
sigma = 0.06; % standard deviation of white noise
z0 = 1; % initial value
mu = 0; % unconditional mean
m = 3; % max +- std. devs
N = 11; % number of grid points

% Parameters
T = 200; % time periods

% Get the discretized state vector and transition matrix
[theta Pi] = tauchen(N,mu,rho,sigma,m);

% number of states
num = size(theta);
num = num(1);

% initialize path
s_path = zeros(1,T);
s_path(1) = ceil(N/2); % initialize starting state

% make random draws
draw_path = rand(1,T);

% simulate the path (in indices)
for t = 2:T
  % select random draw
  draw = draw_path(t);
  % transform transition probabilities into cumulative sums
  c_sum = cumsum(Pi(s_path(t-1),:));

  % select the next state (smallest element of c_sum that is strictly
  % greater than the random draw)
  % short version
  s_path(t) = min(find(c_sum > draw));
end

% transform index path to path in states
s_path = theta(s_path);

% Sample mean and variance 
s_path_mean = mean(s_path);
disp(['Sample mean = ', num2str(s_path_mean)]);

s_path_var = var(s_path);
disp(['Sample variance = ', num2str(s_path_var)]);

% Plot the resulting path
figure(6)
plot(1:T,s_path)
xlabel('Periods')
ylabel('log(a_{t+1})')

%% Discussion of results in question 4

% First of all, notice that here we have a zero mean process. Since
% $|\rho|<1$, we know the process is stationary. Hence, unconditional
% mean is zero and variance is $\frac{\sigma^2}{1-\rho^2} = 0.01$. So, in
% both cases it is easy to see that the sample variances are relatively
% close to the unconditional variance of a continuous process. And the
% sample means are within one standard deviation from the population mean.
