function [Zsim, z_id] = markovsim(T, Z, z0, Zprob)
Zsim = zeros(1,T);
z_id = zeros(1,T);
[~,z0_id] = min(abs(Z - z0));
z_id(1)  = z0_id; % fix starting point (corresponds to z0=1)
y = rand(1,T);

for i = 2:T
  x = y(i); % uniform random number on (0,1)
  % we move to the LOWEST state q out of (1,...,N) such that the
  % CUMULATIVE probabilities in the relevant row of the transition matrix
  % are STRICTLY above the drawn random number
  cumu_p = cumsum(Zprob(z_id(i-1),:)); % cumulative probabilities
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
  z_id(i) = j;
end
% transform the index chain into the actual markov chain (in levels)
% s is the vector of states
Zsim = Z(z_id);