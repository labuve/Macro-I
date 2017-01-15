%% Problem Set 6
% Nurfatima Jandarova
% January 15, 2017

%% Exercise 2
clear all
clc

% Parameters
alpha = 0.36;       % capital share of output
beta = 0.99;        % time preference parameter
delta = 0.025;      % depreciation rate
gamma = 1;          % relative risk aversion
rho = 0.95;         % productivity persistence parameter
sigma = 0.06;       % st.dev of productivity shock
a0 = 1;             % starting level of productivity
S = 5;              % number of states of productivity levels
m = 2;              % width of Markov Process (Tauchen)
mu = 0;             % mean of Markov Process (centered on zero)
T = 5000;           % length of simulation period
N = 500;            % number of capital grid points
k_ss = (1/alpha*(1/beta - 1 + delta))^(1/(alpha - 1)); % steady state capital
k_0 = 0.1*k_ss;     % starting level of capital
tolv = 1e-5;        % tolerance for value function iteration

% Capital grid
k_grid = linspace(0, 3*k_ss, N);

% Discretize Markov chain
[Z, Zprob] = tauchen(S, mu, rho, sigma, m);

% Initialize the value function
V_0 = zeros(S,N);
V_1 = zeros(S,N);

% Value Function iteration
err = 1;
c = (kron((k_grid.^alpha)', exp(Z)) + kron((1 - delta)*k_grid', ones(S,1)))*ones(1,N) - ...
    ones(S*N,1)*k_grid;
if gamma == 1
    U = log(max(c, 0));
else
    U = (max(c, 0).^(1 - gamma)-1)./(1-gamma);
end

while err > tolv
    W = U + beta*kron(ones(N,1), Zprob*V_0);
    [V_aux, k_aux] = max(W, [], 2);
    V_1 = reshape(V_aux, S, N);
    k_idx = reshape(k_aux, S, N);
    err = abs(max(max(V_1-V_0)));
    V_0 = V_1;
    disp(['Current error value: ', num2str(err)])
end

clear V_aux k_aux
% Plot policy function for capital
figure(1)
plot(k_grid, k_grid(k_idx(1,:)), 'b-', k_grid, k_grid(k_idx(S,:)), 'r-', ...
    k_grid, k_grid, 'k--')
xlabel('Capital at the beginning of current period')
ylabel('Capital at the beginning of next period')
title('Decision rule for capital next period')
h = legend(strcat('ln(a) = ',num2str(Z(1), '%.2f')), ...
    strcat('ln(a) = ', num2str(Z(S), '%.2f')), '$45^\circ$ line', ...
    'Location', 'Best');
set(h, 'Interpreter', 'latex')


%%% Simulate a Markov Chain (starting at z0=0)
% we start out with indices and then transform them into levels
[mc, a_id] = markovsim(T, Z, log(a0), Zprob);

%%% Plot the markov chain
figure(2)
plot(1:T, mc)
xlabel('Time')
ylabel('Current State')
title(strcat('Simulated Markov Chain with ', num2str(S), ' States'))

%%% Optimal path for capital starting at k0, a0 = 1;
% find capital grid point closest to k0
[~,k0_id] = min(abs(k_grid-k_0));

% initialize variable for the capital path (should hit 0 in period T+1)
k_path = zeros(1,T);
k_id = zeros(1,T);
% "walk" the path using optimal index
% we start by filling the path with indices and then transforming
% those to levels that can be plotted
k_id(1) = k0_id;
k_path(1) = k_grid(k0_id); % start of at k0
for i = 2:T
    k_id(i) = k_idx(a_id(i-1), k_id(i-1));
    k_path(i) = k_grid(k_id(i)); % use optimal capital choice from VFI
end
    
%%% Plotting optimal capital path
figure(3)
plot(1:T, k_path)
    hold on
    plot(1001:T, k_path(1001:T))
        hold on
        plot(1:T, ones(1,T)*k_ss, 'k--')
hold off
xlabel('Time')
ylabel('Current Capital')
title('Optimal capital path starting at steady state and a_0 = 1')
legend('Capital path', 'Capital path without first 1000 periods', ...
    'Steady-state capital', 'Location','Best')