function model_extract(method)
  %% This function generates thermal system matrices G, C, B, L
  %% It also computes A, A_bar for usage of GDP, etc.
  %% method should be 'steady' or 'transient'

  if strcmp(method,'transient')
    [G,C,B,L] = gcbl_gen(method);
    [A, A_bar] = a_mat_gen(G, C, B, method);
  elseif strcmp(method,'steady')
    [G,C,B,L] = gcbl_gen(method);
  else
    error("method for model_extract(method) should be 'transient' or 'steady'");
  end


