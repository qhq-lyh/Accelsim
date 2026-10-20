function [G,C,B,L] = gcbl_gen(method)
  %% Generate and save G, C, B, L matrices from the HotSopt dumped matrices
  %% method should be 'steady' or 'transient'

  %% Amatrix in HotSpot is G matrix
  Gmatrixcolptr = importdata('Amatrixcolptr');
  Gmatrixnzval = importdata('Amatrixnzval');
  Gmatrixrowind = importdata('Amatrixrowind');
  Gmatrixcolind=[];
  All_grids = length(Gmatrixcolptr)-1;
  for i=1:All_grids
    for j=1:Gmatrixcolptr(i+1)-Gmatrixcolptr(i)
      Gmatrixcolind = [Gmatrixcolind, i];
    end
  end
  Gmatrixrowind = (Gmatrixrowind + 1)';
  G = sparse(Gmatrixrowind,Gmatrixcolind,Gmatrixnzval);

  if strcmp(method,'transient')
    Cmatrix = importdata('Cmatrix');
    C = spdiags(Cmatrix, 0, length(Cmatrix), length(Cmatrix));
  elseif strcmp(method,'steady')
    C = sparse(0,0); % create a dummy matrix for return
  end

  Bmatrix = importdata('Bmatrix');
  B_row = size(Bmatrix, 1);

  Lmatrix = importdata('Lmatrix');
  L_row = size(Lmatrix, 1);

  %% Note that row/col indices wrote in Bmatrix/Lmatrix files starts
  %% from 0, should +1 here in matlab!
  B = sparse(Bmatrix(1:B_row-1,1)+1, Bmatrix(1:B_row-1,2)+1, Bmatrix(1:B_row-1,3), Bmatrix(B_row,1), Bmatrix(B_row,2));
  L = sparse(Lmatrix(1:L_row-1,1)+1, Lmatrix(1:L_row-1,2)+1, Lmatrix(1:L_row-1,3), Lmatrix(L_row,1), Lmatrix(L_row,2));
  L = L';

  save('G.mat', 'G');
  save('B.mat', 'B');
  save('L.mat', 'L');
  if strcmp(method,'transient')
    save('C.mat', 'C');
  end
end
