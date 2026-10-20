import sys
import numpy as np
from scipy.sparse import csc_matrix, spdiags
from scipy.sparse.linalg import spsolve
from scipy.signal import cont2discrete
from scipy.io import loadmat, savemat

def gcbl_gen(method):
    """
    Generate and save G, C, B, L matrices from the HotSopt dumped matrices
    method should be 'steady' or 'transient'
    """
    
    # Amatrix in HotSpot is G matrix
    Gmatrixcolptr = np.loadtxt('Amatrixcolptr', dtype=int)
    Gmatrixnzval = np.loadtxt('Amatrixnzval')
    Gmatrixrowind = np.loadtxt('Amatrixrowind', dtype=int)
    G = csc_matrix((Gmatrixnzval, Gmatrixrowind, Gmatrixcolptr))

    if method == 'transient':
        Cmatrix = np.loadtxt('Cmatrix')
        C = spdiags(Cmatrix, 0, Cmatrix.size, Cmatrix.size, format='csc')
    elif method == 'steady':
        C = csc_matrix((0, 0)) # create a dummy matrix for return

    Bmatrix = np.loadtxt('Bmatrix')
    B_row = Bmatrix.shape[0]

    Lmatrix = np.loadtxt('Lmatrix')
    L_row = Lmatrix.shape[0]

    row_ind_B = Bmatrix[:B_row-1, 0].astype(int)
    col_ind_B = Bmatrix[:B_row-1, 1].astype(int)
    data_B = Bmatrix[:B_row-1, 2]
    shape_B = (int(Bmatrix[B_row-1, 0]), int(Bmatrix[B_row-1, 1]))
    B = csc_matrix((data_B, (row_ind_B, col_ind_B)), shape=shape_B)

    row_ind_L = Lmatrix[:L_row-1, 0].astype(int)
    col_ind_L = Lmatrix[:L_row-1, 1].astype(int)
    data_L = Lmatrix[:L_row-1, 2]
    shape_L = (int(Lmatrix[L_row-1, 0]), int(Lmatrix[L_row-1, 1]))
    L = csc_matrix((data_L, (row_ind_L, col_ind_L)), shape=shape_L)
    L = L.T

    # Save in Matlab compatible format.
    # To load the saved matrices:
    # In Matlab, just use load('G.mat')
    # In python, just use G = loadmat('G.mat')['G']
    savemat("G.mat", {'G':G})
    savemat("B.mat", {'B':B})
    savemat("L.mat", {'L':L})
    if method == 'transient':
        savemat("C.mat", {'C':C})

    return G, C, B, L

def a_mat_gen(G, C, B, method):
    """
    Generate A and A_bar for power budgeting usage
    method should be 'steady' or 'transient'
    """

    # set the power budgeting time step, works for transient GDP (A_bar)
    t_budget = 0.001

    # G = loadmat('G.mat')['G']
    # C = loadmat('C.mat')['C']
    # B = loadmat('B.mat')['B']

    # compute A matrix for steady state GDP
    G_inv_B = spsolve(G, B.toarray()) # return G_inv_B is dense
    # G_inv_B = spsolve(G, B.toarray(), permc_spec='MMD_AT_PLUS_A') # 'MMD_AT_PLUS_A' is better than default 'COLAMD' tested in my large size case
    # import pypardiso;  G_inv_B = pypardiso.spsolve(G, B.toarray()) # this is much faster using intel MKL
    A = B.T @ G_inv_B # A is already dense
    # np.save('A.npy', A)
    savemat("A.mat", {'A':A})

    if method == 'transient':
        # compute A_bar matrix for transient GDP with power budget step t_budget
        Ac = -spsolve(C, G.toarray())
        Bc = spsolve(C, B.toarray())
        Cc = B.T.toarray()
        Dc = np.zeros((A.shape[0], A.shape[0]))
        M, N, L, _, _ = cont2discrete((Ac, Bc, Cc, Dc), t_budget)
        A_bar = B.T @ N
        #np.save(f'A_{int(t_budget * 1000)}ms.npy', A_bar)
        savemat(f'A_{int(t_budget * 1000)}ms.mat', {'A_bar':A_bar})
    elif method == 'steady':
        A_bar = np.zeros((0, 0)) # create a dummy matrix for return
    else:
        raise Exception("method must be 'steady' or 'transient'")

    return A, A_bar


def model_extract(method):
    """
    This function generates thermal system matrices G, C, B, L
    It also computes A, A_bar for usage of GDP, etc.
    method should be 'steady' or 'transient'
    """
    if method in ['transient', 'steady']:
        G, C, B, L = gcbl_gen(method) # steady case C is dummy matrix
        A, A_bar = a_mat_gen(G, C, B, method) # steady case A_bar is dummy matrix
        return G, C, B, L, A, A_bar
    else:
        raise Exception("method for model_extract(method) should be 'transient' or 'steady'")

    

if __name__ == "__main__":
    if (len(sys.argv)==1 or (sys.argv[1] != 'transient' and sys.argv[1] != 'steady')):
        raise Exception("Please specify the simulation type: 'transient' or 'steady'")
    elif len(sys.argv)>2:
        raise Exception("Invalid number of inputs.")
    method = sys.argv[1]
    model_extract(method)
