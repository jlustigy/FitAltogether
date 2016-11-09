import numpy as np
import sys
import PCA
import shrinkwrap

INFILE = 'data/raddata_2_norm'

#===================================================
def do ( Obs_ij, run_dir ) :

    # PCA
    N_PCs, V_nj, U_in = PCA.do_PCA( Obs_ij, run_dir ) 

    print '# ---------------------------------------------------'
    print '# U_in'
    # print samples
    for ii in xrange( len( U_in ) ):
        for nn in xrange( len( U_in.T ) ):
            print U_in[ii][nn],
        print ''
    print ''

    # shrink wrap
    A_mn = shrinkwrap.do_shrinkwrap ( U_in, N_PCs, run_dir )
    
