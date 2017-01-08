%% Problem Set 5

% Nurfatima Jandarova

clear all
clc

%% Exercise 2

% Parameters
rho = 0.8;      % persistence of productivity
sigma = 0.06;   % standard deviation of the TFP shock
alpha = 1/3;    % capital share of output
beta = 0.99;    % patience
delta = 0.025;  % depreciation rate
S = 5;          % number of points in the markov chain
T = 2000;       % number of periods (for the simulation)
m = 2;          % width of Markov Process (Tauchen)
mu = 0;         % mean of Markov Process (centered on zero)
a0 = 1;         % starting value of productivity
N = 500;        % number of grid points for capital
tolv = 1e-7;    % tolerance for value function iteration

% Discretize Markov chain
[Z, Zprob] = tauchen(S, mu, rho, sigma, m);

% Steady-state level of capital
k_ss = ((1/beta - 1 + delta)/alpha)^(1/(alpha-1));

% Capital grid
k_grid = linspace(0, 2*k_ss, N);

% Initialize the value function
V_0 = zeros(S,N);
V_1 = zeros(S,N);

% Value Function iteration
err = 1;
c = (kron((k_grid.^alpha)', exp(Z)) + kron((1 - delta)*k_grid', ones(S,1)))*ones(1,N) - ...
    ones(S*N,1)*k_grid; 
U = log(max(min(c, kron((k_grid.^alpha)',exp(Z))*ones(1,N)), 0));

while err > tolv
    W = U + beta*kron(ones(N,1), Zprob*V_0);
    [V_aux, k_aux] = max(W, [], 2);
    V_1 = reshape(V_aux, S, N);
    k_idx = reshape(k_aux, S, N);
    err = abs(max(max(V_1-V_0)));
    V_0 = V_1;
    disp(['Current error value: ', num2str(err)])
end

figure(1)
plot(k_grid, k_grid(k_idx(1,:)), 'b-', k_grid, k_grid(k_idx(5,:)), 'r-', ...
    k_grid, k_grid, 'k--')
xlabel('Capital at the beginning of current period')
ylabel('Capital at the beginning of next period')
title('Decision rule for capital next period')
legend('ln(a) = -0.2', 'ln(a) = 0.2', '45 degree line', 'Location', 'Best')
saveas(gcf,'ex2kpolicy','epsc')

figure(2)
plot(1:N, V_1(1,:), 1:N, V_1(2,:), 1:N, V_1(3,:), 1:N, V_1(4,:), 1:N, V_1(5,:))
xlabel('Current Capital')
ylabel('Lifetime Value')
title('Lifetime Value at Different Capital Levels')
legend('ln(a) = -0.2', 'ln(a) = -0.1', 'ln(a) = 0.0', ...
    'ln(a) = 0.1', 'ln(a) = 0.2', 'Location', 'Best')
saveas(gcf,'ex2value','epsc')



%%% Simulate a Markov Chain (starting at z0=0)
% we start out with indices and then transform them into levels
mc = zeros(1,T);
a_id = zeros(1,T);
[~,a0_id] = min(abs(Z - log(a0)));
a_id(1)  = a0_id; % fix starting point (corresponds to z0=1)
y = rand(1,T);

for i = 2:T
  x = y(i); % uniform random number on (0,1)
  % we move to the LOWEST state q out of (1,...,N) such that the
  % CUMULATIVE probabilities in the relevant row of the transition matrix
  % are STRICTLY above the drawn random number
  cumu_p = cumsum(Zprob(a_id(i-1),:)); % cumulative probabilities
  % pick lowest index such that the cumulative sum up to this index is
  % strictly greater than x
  diff = 0;
  j = 0 ; % start with the first index
  while diff <= 0
    j = j + 1;
    trial = cumu_p(j); % exit loop as soon as the cumulative sum up to the
    % current index exceeds x
    diff = trial - x;
  end
  % once the above loop exits, we have found our next step in the markov chain
  a_id(i) = j;
end
% transform the index chain into the actual markov chain (in levels)
% s is the vector of states
mc = Z(a_id);

%%% Plot the markov chain
figure(3)
plot(1:T, mc)
xlabel('Time')
ylabel('Current State')
title('Simulated Markov Chain with 5 States')
saveas(gcf,'ex2mc','epsc')

%%% Optimal path for capital starting at k0 = k_ss, a0 = 1;
% find capital grid point closest to k0
[~,k0_id] = min(abs(k_grid-k_ss));

% initialize variable for the capital path (should hit 0 in period T+1)
k_path = zeros(1,T);
c_path = zeros(1,T);
i_path = zeros(1,T);
k_id = zeros(1,T);
% "walk" the path using optimal index
% we start by filling the path with indices and then transforming
% those to levels that can be plotted
k_id(1) = k0_id;
k_path(1) = k_grid(k0_id); % start of at k0
for i = 2:T
    k_id(i) = k_idx(a_id(i-1), k_id(i-1));
    k_path(i) = k_grid(k_id(i)); % use optimal capital choice from VFI
    c_path(i-1) = exp(mc(i-1))*k_path(i-1)^alpha + ...
        (1-delta)*k_path(i-1) - k_path(i); % optimal consumption path
    i_path(i-1) = k_path(i) - (1-delta)*k_path(i-1);
end
c_path(T) = NaN;
i_path(T) = NaN;
    
%%% Plotting optimal capital path
figure(4)
plot(1:T,k_path, 1:T, ones(1,T)*k_ss, '--')
xlabel('Time')
ylabel('Current Capital')
title('Optimal capital path starting at steady state and a_0 = 1')
legend('Capital path', 'Steady-state capital','Location','Best')
saveas(gcf,'ex2kpath','epsc')

%%% Plotting optimal consumption path
figure(5)
plot(1:T,c_path)
xlabel('Time')
ylabel('Current Consumption')
title('Optimal consumption path starting at steady-state and a_0 = 1')
saveas(gcf,'ex2capth','epsc')

%%% Plotting optimal consumption path
figure(6)
plot(1:T,i_path)
xlabel('Time')
ylabel('Current Investment')
title('Optimal investment path starting at steady-state and a_0 = 1')
saveas(gcf,'ex2ipath','epsc')

%% Exercise 4 (wrong)

% Parameters
alpha = 2/3;        % labour share of output
beta = 0.95;        % patience parameter
delta = 0.025;      % depreciation rate
T = 100;            % number of time periods
k0 = 2;             % starting level of capital
N = 200;            % number of capital grid-points
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
figure(7)
plot(1:T+1,k_path)
xlabel('Time')
ylabel('Current Capital')
xlim([1 T+1])
% saveas(gcf,'kpath','epsc')

% optimal path for labour
figure(8)
plot(1:T, n_path)
xlabel('Time')
ylabel('Labour')
% saveas(gcf,'npath','epsc')

% optimal path for consumption
figure(9)
plot(1:T, c_path)
xlabel('Time')
ylabel('Consumption')
% saveas(gcf,'cpath','epsc')