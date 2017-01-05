%% Problem Set 5

% Nurfatima Jandarova

clear all
clc

%% Exercise 4

% Parameters
alpha = 2/3;        % labour share of output
beta = 0.95;        % patience parameter
delta = 0.025;      % depreciation rate
T = 60;             % number of time periods
k0 = 2;             % starting level of capital
N = 100;            % number of capital grid-points
tolv = 1e-5;        % tolerance for value function iteration

% capital grid
k_grid = linspace(0, 10*k0, N);

% flow utility matrix
c = alpha/(1 - alpha)*(ones(N,1)*k_grid - (1-delta)*k_grid'*ones(1, N));
c(c<0) = NaN;   % lower bound on consumption
n = ((ones(N,1)*k_grid - (1-delta)*k_grid'*ones(1, N))./...
    ((1 - alpha)*(k_grid'*ones(1, N)).^(1 - alpha))).^(1/alpha);
n(n<0) = NaN;   % lower bound on labour
n(n>1) = NaN;   % upper bound on labour

u = log(c)-log(n);

% Initialize the value function
V_0 = zeros(T+1,N); 
V_1 = zeros(T+1,N);
% Initialize the optimal capital choice
k_opt = zeros(T,N);

% Value Function iteration
err = 1;
while err > tolv
  for t = 1:T % time periods
      V_aux = u + beta*ones(N,1)*V_0(t+1,:);
      % choose k' that maximizes V(t,k)
      [V_1(t,:),k_opt(t,:)] = max(V_aux', [], 1);
  end
  % calculate the error term
  err = abs(max(max(V_1 - V_0)));
  disp(['Current error: ', num2str(err)]);
  % update value function
  V_0 = V_1;
end

% Optimal paths of capital, labour and consumption starting at k0=2
[~,idx] = min(abs(k_grid-k0));
% initialize variable for the capital path (should hit 0 in period T+1)
k_path = zeros(1,T+1);
k_path(1) = idx;
k_path(T+1) = 1;

% initialize variables for labour and consumption paths
n_path = zeros(1, T);
c_path = zeros(1, T);

for i = 2:T
  k_path(i) = k_opt(i,k_path(i-1));
end
% transform into capital levels
k_path = k_grid(k_path);

for i = 1:T-1
    n_path(i) = ((k_path(i+1)-(1-delta)*k_path(i))/...
        ((1-alpha)*k_path(i)^(1-alpha)))^(1/alpha);
    c_path(i) = alpha/(1-alpha)*(k_path(i+1)-(1-delta)*k_path(i));
end

c_path(T) = (1 - delta)*k_path(T);

%%% Plotting the results
% optimal capital path
figure(1)
plot(1:T+1,k_path)
xlabel('Time')
ylabel('Current Capital')
xlim([1 T+1])
saveas(gcf,'kpath','epsc')

% optimal path for labour
figure(2)
plot(1:T, n_path)
xlabel('Time')
ylabel('Labour')
saveas(gcf,'npath','epsc')

% optimal path for consumption
figure(3)
plot(1:T, c_path)
xlabel('Time')
ylabel('Consumption')
saveas(gcf,'cpath','epsc')