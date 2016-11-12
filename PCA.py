import numpy as np
from sklearn.decomposition import PCA

from fitlc_params import ALBDFILE

#---------------------------------------------------
def do_PCA ( Dij, E_cutoff=5e-2, output=True, run_dir='' ):

    # number of principle components
    decomposer = PCA(n_components=len(Dij.T))

    # execute
    # YFYFYF    decomposer.fit(data.T)
    U = decomposer.fit_transform( Dij )

    M = decomposer.mean_

    # PCs
    V = decomposer.components_

    # eigen values
    E = decomposer.explained_variance_ratio_

    N_PCs = np.sum( E > E_cutoff )

    print "   cut-off below", E_cutoff, "..."
    print "   # of PCs: ", N_PCs

    if output :
        np.savetxt( run_dir+'PCA_V_jn', V.T, header='Principal Components ( dimension: band x index )' ) 
        np.savetxt( run_dir+'PCA_E_n', E, header='Contribution factor ( dimension: index )' ) 
        np.savetxt( run_dir+'PCA_U_in', U, header='Conefficient ( dimension: time x index )' ) 
        np.savetxt( run_dir+'AVE_j', M, header='Time Average ( dimension: band )' ) 
        # projection of 'answer' onto PC plane
        albd_kj  = np.loadtxt( ALBDFILE ).T
        dalbd_kj = albd_kj - M
    #    coeff_kn = np.dot( dalbd_kj, np.linalg.inv( V ) )
        coeff_kn = np.dot( dalbd_kj, V.T )
        np.savetxt( run_dir+'PCA_answer_projected', coeff_kn )

    return N_PCs, V[:N_PCs][:], U[:,:N_PCs], M

