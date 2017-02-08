import numpy as np
import sys
import PCA
import shrinkwrap

INFILE = 'data/raddata_2_norm'

#===================================================
if __name__ == "__main__":

    # input data
    Obs_ij = np.loadtxt(INFILE)
    n_slice = len(Obs_ij)

    # PCA
    N_PCs, V_nj, U_in = PCA.do_PCA( Obs_ij ) 

#    print 'Principal Components (V_lj) : '
#    print V_nj
#    print ''
#    print 'Coefficients (U_il) : '
#    print U_in
#    print ''
    print '# ---------------------------------------------------'
    print '# U_in'
    # print samples
    for ii in xrange( len( U_in ) ):
        for nn in xrange( len( U_in.T ) ):
            print U_in[ii][nn],
        print ''
    print ''

    # shrink wrap
    A_mn = shrinkwrap.do_shrinkwrap ( U_in, N_PCs )

    
    # print vertices
    print ''
    print '# ---------------------------------------------------'
    print '# A_mn'
    for mm in xrange( len( A_mn ) ):
        for nn in xrange( len( A_mn.T ) ):
            print A_mn[mm][nn],
        print ''

