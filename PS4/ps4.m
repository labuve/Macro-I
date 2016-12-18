%% Problem Set 4

% Nurfatima Jandarova

clear all
clc

%% Exercise 5.1 (LS)

% n = size(R, 1); % Size of matrix R
% P0 = zeros(n, n); % Initial guess for P
% error = 1;  % Intialize the error
% tolv = 10^-5; % tolerance value
% 
% while error > tolv
%     P1 = R + beta*A'*P*A - (H' + beta*A'*P*B)*(Q + beta*B'*P*B)^{-1}*(H + beta*B'*P*A);
%     error = abs(max(max(P1 - P0)));
%     P0 = P1;
% end

%% Exercise 5.3 (LS)

% Parameters
beta = 0.95;        % time patience parameter
r = (0.95)^(-1)-1;  % interest rate
b = 30;
gamma = 1;
rho1 = 1.2;         % persistence of one-period shock
rho2 = -0.3;        % persistene of two-period shock

% For some reason, the iteration does not work when I have a constant in
% u_t
% R = [r^2, r, 0; r, 1, 0; 0, 0, 0];
% Q = [1 + gamma, b; b, b^2];
% H = [-r, -1, 0; -b*r, -b, 0];
% A = [1, 0, 0; 0, rho1, rho2; 0, 1, 0];
% B = [1, 0; 0, 0; 0, 0];
% 
% n = size(R, 1); % Size of matrix R
% P0 = zeros(n, n); % Initial guess for P
% P1 = zeros(n, n);
% error = 1;  % Intialize the error
% tolv = 10^-5; % tolerance value
% 
% while error > tolv
%     P1 = R + beta*A'*P0*A - (beta*A'*P0*B + H')*(Q + beta*B'*P0*B)^(-1)*(beta*B'*P0*A + H);
%     error = abs(max(max(P1 - P0)));
%     P0 = P1;
% end
% 
% F = -(Q+beta*(B')*P0*B)^(-1)*(beta*(B')*P0*A+H);

%% Second attempt

R = [gamma*r^2, gamma*r, 0, 0; gamma*r, gamma, 0, 0; 0, 0, 0, 0; 0, 0, 0, b^2];
Q = 1+gamma;
H = [-gamma*r, -gamma, 0, b];
A = [1+r, 1, 0, 0; 0, rho1, rho2, 0; 0, 1, 0, 0; 0, 0, 0, 1];
B = [-1, 0, 0, 0]';

n = size(R, 1); % Size of matrix R
P0 = zeros(n, n); % Initial guess for P
P1 = zeros(n, n);
error = 1;  % Intialize the error
tolv = 10^-5; % tolerance value

while error > tolv
    P1 = R + beta*A'*P0*A - (beta*A'*P0*B + H')*(Q + beta*B'*P0*B)^(-1)*(beta*B'*P0*A + H);
    error = abs(max(max(P1 - P0)));
    P0 = P1;
end

F1=-(Q+beta*(B')*P0*B)^(-1)*(beta*(B')*P0*A+H);