function [Z,Zprob] = tauchen(N, k, mu, rho, sigma, m)
%Function TAUCHEN
%
%Purpose:    Finds a Markov chain whose sample paths
%            approximate those of the AR(1) process
%                z(t+1) = (1-rho)*mu + rho * z(t) + eps(t+1)
%            where eps are normal with stddev sigma
%
%Format:     {Z, Zprob} = Tauchen(N,mu,rho,sigma,m)
%
%Input:      N       scalar, number of nodes for Z
%            k       scalar, number of state variables
%            mu      1xk vector of unconditional mean of each variable
%                    or scalar, if they have same mean
%            rho     1xk vector of persistence parameters
%                    or scalar, if they have same persistence
%            sigma   kxk varcovar matrix of epsilons or
%                    1xk vector of variances if they are independent
%            m       max +- std. devs.
%
%Output:     Z       N*k vector, nodes for Z
%            Zprob   Nk*Nk matrix, transition probabilities
%
%    Martin Floden
%    Fall 1996
%
%    This procedure is an implementation of George Tauchen's algorithm
%    described in Ec. Letters 20 (1986) 177-181.
%

% complete dimensions of some objects
if isscalar(rho)
    rho = diag(rho*eye(k))';
end
if isscalar(mu)
    mu = diag(mu*eye(k))';
end
if isvector(sigma)
    sigma = diag(sigma);
else
    sigma = sigma*eye(k);
end

Z     = zeros(N,k);
Zprob = zeros(N^k,N^k);
zstep = zeros(1, k);
a     = (ones(1,k)-rho).*mu;

% How to take into account possible cross correlation?
for j = 1: k
    Z(N, j)  = m * sqrt(sigma(j, j)/(1 - rho(j)^2));
    Z(1, j)  = -Z(N, j);
    zstep(j) = (Z(N,j) - Z(1,j))/(N - 1) %distance between points in the grid for z
end

for i=2:(N-1)
    Z(i, :) = Z(1, :) + zstep * (i - 1); %we create a grid
end 

Z = Z + ones(N,1)*(a./(ones(1,k)-rho));

% Generating every possible combination of the k states
Z_aux = zeros(N^k,k);

for j = 1:k-1
    Z_aux(:,j) = kron(ones(N^(j-1),1),kron(Z(:,j), ones(N^(k-j),1)));
end

Z_aux(:,k) = kron(ones(N^(k-1),1),Z(:,k));

%how to work this out?
p = mvncdf(Z_aux,mu,sigma);

for j = 1:N^k
    for s = 1:N^k
        
%         if k == 1
%             Zprob(j,k) = cdf_normal((Z(1) - a - rho * Z(j) + zstep / 2) / sigma);
%         elseif k == N
%             Zprob(j,k) = 1 - cdf_normal((Z(N) - a - rho * Z(j) - zstep / 2) / sigma);
%         else
%             Zprob(j,k) = cdf_normal((Z(k) - a - rho * Z(j) + zstep / 2) / sigma) - ...
%                          cdf_normal((Z(k) - a - rho * Z(j) - zstep / 2) / sigma);
%         end

        if s == 1
            Zprob(j,s) = normcdf((Z(1) - a - rho * Z(j) + zstep / 2) / sigma,0,1);
        elseif s == N^k
            Zprob(j,s) = 1 - normcdf((Z(N) - a - rho * Z(j) - zstep / 2) / sigma,0,1);
        else
            Zprob(j,s) = normcdf((Z(s) - a - rho * Z(j) + zstep / 2) / sigma,0,1) - ...
                         normcdf((Z(s) - a - rho * Z(j) - zstep / 2) / sigma,0,1);
        end



        
    end
end


% function c = cdf_normal(x)
%     c = 0.5 * erfc(-x/sqrt(2));

