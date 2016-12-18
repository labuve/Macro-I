%% Problem Set 2

% Nurfatima Jandarova

clear all
clc

%% Exercise 1.3

% Parameters
betta = 0.95;       % time preference parameter
deltta = 0.025;     % depreciation rate
alphha = 0.36;      % capital share
rhho = 2;           % degree of relative risk aversion
k0 = 2;             % starting level of capital
N = 400;            % number of capital grid-points
tolv = 1e-6;        % tolerance for value function iteration
T = 1000;           % number of time periods to plot capital path

% capital grid
k_grid = linspace(0, 6*k0, N);

% flow utility matrix
c = (k_grid.^alphha + (1-deltta)*k_grid)'*ones(1, N) - ones(N, 1)*k_grid;
c(c<=0) = NaN;
u = (c.^(1-rhho))./(1-rhho);

% Initialize the value function
V_0 = zeros(1,N);
V_1 = zeros(1,N);

% Value Function iteration
err = 1;

while err > tolv
    % Computes all possible value functions for each value of initial capital
    W = u + betta * kron(ones(N, 1), V_0);
    % Finds the maximal value function for each values of initial capital
    [V_1, k_idx] = max(W, [], 2);
    V_1 = V_1';
    err = max(abs(V_0-V_1));
    % Update the guess
    V_0 = V_1;
end

% Plotting the results
% Value function
figure(1)
plot(k_grid, V_1)
xlabel('Current Capital')
ylabel('Lifetime Value')
title('Lifetime Value at Different Capital Levels')

%Decision rule
figure(2)
plot(k_grid, k_grid(k_idx))
xlabel('Capital at the beginning of current period')
ylabel('Capital at the beginning of next period')
title('Decision rule for capital next period')

%%% Optimal path for capital starting at k0 = 2
% find capital grid point closest to k0
[~,idx] = min(abs(k_grid-k0));
% initialize variable for the capital path (should hit 0 in period T+1)
k_path = zeros(1,T);
% "walk" the path using optimal index
% we start by filling the path with indices and then transforming
% those to levels that can be plotted
k_path(1) = idx; % start of at k0
for i = 2:T
  k_path(i) = k_idx(k_path(i-1)); % use optimal capital choice from VFI
  % k_path(i-1) denotes the current position on the capital grid
end
% transform into capital levels
k_path = k_grid(k_path);

%%% Plotting optimal capital path
figure(3)
plot(1:T,k_path)
xlabel('Time')
ylabel('Current Capital')
title('Optimal capital path starting at k_0 \approx 2')

%% Exercise 2.2

clear all

% Assumptions
rhho = 0.8;         % persistence of productivity shock
sigmma = 0.06;      % se of the error term in AR(1)
S = 5;              % number of points in the Markov chain
m = 2;              % width of Markov process
mmu = 0;            % mean of Markov process
alphha = 0.36;      % capital share
betta = 0.99;       % time preference parameter
deltta = 0.025;     % capital depreciation rate
tolv = 1e-7;        % tolerance for value function iteration
N = 500;            % number of capital grid-points
k0 = 2;             % starting level of capital
T = 200;            % time period for simulation
a0 = 1;             % starting level of productivity
kmax = 70;          % upper bound for capital grid

%%% Generate the Markov Transition Matrix
[Z,Zprob] = tauchen(S,mmu,rhho,sigmma,m);

% Capital grid
k_grid = linspace(0, kmax, N);

% Initialize the value function
V_0 = zeros(S,N);
V_1 = zeros(S,N);

% Value Function iteration
err = 1;
U = log(max((kron((k_grid.^alphha)',exp(Z)) + ...
    kron((1 - deltta)*k_grid', ones(S,1)))*ones(1,N) -...
    ones(S*N,1)*k_grid,0));

while err > tolv
    W = U + betta*kron(ones(N,1),Zprob*V_0);
    [V_aux, k_aux] = max(W, [], 2);
    V_1 = reshape(V_aux, S, N);
    k_idx = reshape(k_aux, S, N);
    err = max(max(abs(V_0-V_1)));
    V_0 = V_1;
    disp(['Current error value: ', num2str(err)])
end

figure(4)
plot(k_grid, k_grid(k_idx(1,:)), 'b-', k_grid, k_grid(k_idx(5,:)), 'r-', ...
    k_grid, k_grid, 'k--')
xlabel('Capital at the beginning of current period')
ylabel('Capital at the beginning of next period')
title('Decision rule for capital next period')
legend('ln(a) = -0.2', 'ln(a) = 0.2', '45 degree line', 'Location', 'Best')

figure(5)
plot(1:N, V_1(1,:), 1:N, V_1(2,:), 1:N, V_1(3,:), 1:N, V_1(4,:), 1:N, V_1(5,:))
xlabel('Current Capital')
ylabel('Lifetime Value')
title('Lifetime Value at Different Capital Levels')
legend('ln(a) = -0.2', 'ln(a) = -0.1', 'ln(a) = 0.0', ...
    'ln(a) = 0.1', 'ln(a) = 0.2', 'Location', 'Best')

%% Exercise 2.2.3
%%% Simulate a Markov Chain (starting at z0=0)
% we start out with indices and then transform them into levels
mc     = zeros(1,T);
a_id = zeros(1,T);
[~,a0_id] = min(abs(Z - log(a0)));
a_id(1)  = a0_id; % fix starting point (corresponds to z0=1 - need to change
% depending on the number of states!)
% generate T random numbers on (0,1)
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
figure(6)
plot(1:T,mc)
xlabel('Time')
ylabel('Current State')
title('Simulated Markov Chain with 3 States')

%%% Optimal path for capital starting at k0 = 2, a0 = 2;
% find capital grid point closest to k0
[~,k0_id] = min(abs(k_grid-k0));

% initialize variable for the capital path (should hit 0 in period T+1)
k_path = zeros(1,T);
c_path = zeros(1,T);
k_id = zeros(1,T);
% "walk" the path using optimal index
% we start by filling the path with indices and then transforming
% those to levels that can be plotted
k_id(1) = k0_id;
k_path(1) = k_grid(k0_id); % start of at k0
for i = 2:T
    k_id(i) = k_idx(a_id(i-1), k_id(i-1));
    k_path(i) = k_grid(k_id(i)); % use optimal capital choice from VFI
    c_path(i-1) = exp(mc(i-1))*k_path(i-1)^alphha + ...
        (1-deltta)*k_path(i-1) - k_path(i); % optimal consumption path
    % k_path(i-1) denotes the current position on the capital grid
end
c_path(T) = NaN;
    
%%% Plotting optimal capital path
figure(7)
plot(1:T,k_path)
xlabel('Time')
ylabel('Current Capital')
title('Optimal capital path starting at k_0 \approx 2 and a_0 = 1')

%%% Plotting optimal consumption path
figure(8)
plot(1:T,c_path)
xlabel('Time')
ylabel('Current Consumption')
title('Optimal consumption path starting at k_0 \approx 2 and a_0 = 1')