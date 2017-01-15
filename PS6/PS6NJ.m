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


%% Simulate a Markov Chain (starting at z0=0)
% we start out with indices and then transform them into levels
[mc, a_id] = markovsim(T, Z, log(a0), Zprob);

%%% Plot the markov chain
figure(2)
plot(1:T, mc)
xlabel('Time')
ylabel('Current State')
title(strcat('Simulated Markov Chain with ', num2str(S), ' States'))

%%% Optimal path for capital starting at k0, a0 = 1;
[~,k0_id] = min(abs(k_grid-k_0));

% initialize variable for the capital path (should hit 0 in period T+1)
k_path = zeros(1,T);
c_path = zeros(1,T);
i_path = zeros(1,T);
y_path = zeros(1,T);
k_id = zeros(1,T);
k_id(1) = k0_id;
k_path(1) = k_grid(k0_id);
y_path(1) = exp(mc(1))*k_path(1)^alpha;
for i = 2:T
    k_id(i) = k_idx(a_id(i-1), k_id(i-1)); % use optimal capital from VFI
    k_path(i) = k_grid(k_id(i));
    y_path(i) = exp(mc(i))*k_path(i)^alpha;
    i_path(i-1) = k_path(i) - (1-delta)*k_path(i-1);
    c_path(i-1) = y_path(i-1) - i_path(i-1); % optimal consumption path
end
c_path(T) = NaN;
i_path(T) = NaN;

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

% Unconditional moments

growth_y = zeros(1, T);
growth_c = zeros(1, T);
growth_i = zeros(1, T);
growth_a = zeros(1, T);

for i = 2:T
    growth_y(i) = y_path(i)/y_path(i-1)-1;
    growth_c(i) = c_path(i)/c_path(i-1)-1;
    growth_i(i) = i_path(i)/i_path(i-1)-1;
    growth_a(i) = exp(mc(i) - mc(i-1))-1;
end

mat_std = zeros(1,3);
mat_std(1) = nanstd(growth_y(1001:T));
mat_std(2) = nanstd(growth_c(1001:T));
mat_std(3) = nanstd(growth_i(1001:T));
mat_std

corrcoef(growth_c(1001:T-1), growth_a(1001:T-1))
corrcoef(growth_i(1001:T-1), growth_a(1001:T-1))
corrcoef(growth_c(1001:T-1), growth_y(1001:T-1))
corrcoef(growth_i(1001:T-1), growth_y(1001:T-1))

k_mean = mean(k_path(1001:T));