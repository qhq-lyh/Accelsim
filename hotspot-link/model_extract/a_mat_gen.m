function [A, A_bar] = a_mat_gen(G, C, B, method)
  %% Generate A and A_bar for power budgeting usage
  %% method should be 'steady' or 'transient'
  
  %% set the power budgeting time step, works for transient GDP (A_bar)
  t_budget = 0.001; 

  %% load('G.mat');
  %% load('C.mat');
  %% load('B.mat');

  %% compute A matrix for steady state GDP
  A = full(B'*(G\B));
  save('A.mat', 'A');

  if strcmp(method,'transient')
    %% compute A_bar matrix for transient GDP with power budget step t_budget
    Ac = full(- (C \ G));
    Bc = full(C \ B) ;
    Cc = full(B');
    Dc = zeros(size(A,1), size(A,1));
    [M,N,L,~] = c2dm(Ac,Bc,Cc,Dc,t_budget); % M and N matrices in GDP paper
    A_bar = B'*N;
    save(strcat('A_',string(t_budget*1000),'ms.mat'), 'A_bar');
  end
end
