import numpy as np

#---------------------------------------------------
def get_ln_prior_albd( y_albd_kj ):

    prior_kj = np.exp( y_albd_kj ) / ( 1 + np.exp( y_albd_kj ) )**2
    ln_prior = np.sum( np.log( prior_kj ) )
    return ln_prior



#---------------------------------------------------
def get_ln_prior_area( y_area_lk, x_area_lk ):

    l_dim = len( y_area_lk )
    k_dim = len( y_area_lk.T )
    kk_dim = len( y_area_lk.T )

    sumF = np.cumsum( x_area_lk, axis=1 )

    # when kk < k
    l_indx, k_indx, kk_indx = np.meshgrid( np.arange( l_dim ), np.arange( k_dim ), np.arange( kk_dim ), indexing='ij' )
    dgdF = np.zeros( [ l_dim, k_dim, kk_dim  ] )
    dgdF[ l_indx, k_indx, kk_indx ] = x_area_lk[ l_indx, kk_indx ] / x_area_lk[ l_indx, k_indx ] / ( 1 - sumF[ l_indx, k_indx ] )

    # when kk > k
    k_tmp, kk_tmp   = np.triu_indices(k_dim)
    l_indx, k_indx  = np.meshgrid( np.arange( l_dim ), k_tmp,  indexing='ij' )
    l_indx, kk_indx = np.meshgrid( np.arange( l_dim ), kk_tmp, indexing='ij' )
    dgdF[ l_indx, k_indx, kk_indx ] = 0.

    # when kk = k
    k_tmp, kk_tmp = np.diag_indices(k_dim)
    l_indx, k_indx  = np.meshgrid( np.arange( l_dim ), k_tmp,  indexing='ij' )
    l_indx, kk_indx = np.meshgrid( np.arange( l_dim ), kk_tmp, indexing='ij' )
    dgdF[ l_indx, k_indx, kk_indx ] = 1./x_area_lk[l_indx,k_indx]*(1. - sumF[l_indx, k_indx-1]) / ( 1 - sumF[ l_indx, k_indx ] )

    dgdF_factor = np.linalg.det( dgdF )
    ln_prior = np.sum( np.log( dgdF_factor ) )

    return ln_prior


#---------------------------------------------------
def regularize_area( x_area_lk, wn_rel_amp, lambda_angular ):

    l_dim = len( x_area_lk )
    cov = get_cov( wn_rel_amp, lamnda_angular, l_dim )
    inv_cov = np.linalg.inv( cov )
    det_cov = np.linalg.det( cov )

    x_area_ave = 1./3
    dx_area_lk = x_area_lk - x_area_ave
    term1 = -0.5 * np.dot( dx_area_lk, np.dot( inv_cov, dx_area_lk ) )
    term2 = -0.5 * np.log( det_cov )
    return term1 + term2

#---------------------------------------------------
def get_cov( wn_rel_amp, lambda_angular, l_dim, periodic=True):

#    kappa0 = np.log(output["x"][-1]) - np.log(360.0 - output["x"][-1])
    Sigma_ll = np.zeros([l_dim, l_dim])
    lon_l = 2.0 * np.pi * np.arange( l_dim ) / ( l_dim * 1. )
    dif_lon_ll = lon_l[:,np.newaxis] - lon_l[np.newaxis,:]
    if periodic :
        dif_lon_ll = np.minimum( abs(lon_ll), abs( 2. * np.pi - lon_ll ) )
    else :
        dif_lon_ll = abs( lon_ll )

    Sigma_ll = np.exp( - 0.5 * dif_lon_ll**2 / ( lambda_angular**2 ) )

    cov_ll = Sigma_ll * ( 1 - wn_rel_amp )
    cov[np.diag_indices(l_dim)] += wn_rel_amp
    cov /= (1.0 +  wn_rel_amp)

    return cov



##---------------------------------------------------
#def get_ln_prior_albd_old( y_albd_kj ):
#
#    ln_prior = 0.
#    for k in xrange( len(y_albd_kj) ):
#        for j in xrange( len(y_albd_kj.T) ):
#            yy = y_albd_kj[k,j]
#            prior = np.exp( yy ) / ( 1 + np.exp( yy ) )**2
#            if ( prior > 0. ):
#                ln_prior = ln_prior + np.log( prior )
#            else:
 #                print "ERROR! ln_prior_albd is NaN"
#                print "  y, prior   ", yy, prior
#                ln_prior = ln_prior + 0.0
#
#    return ln_prior



#---------------------------------------------------
#def get_ln_prior_area_old( y_area_lj, x_area_lj ):
#
#    dydx_det = 1.
#    for ll in xrange( len( y_area_lj ) ):
#        dydx = np.zeros( [ len( y_area_lj.T ), len( y_area_lj.T ) ] )
#        for ii in xrange( len( dydx ) ):
#            jj = 0
#            # jj < ii
#            while ( jj < ii ):
#                g_i = y_area_lj[ll,ii]
#                f_i = x_area_lj[ll,ii]
#                f_j = x_area_lj[ll,jj]
#                sum_fi = np.sum( x_area_lj[ll,:ii+1] )
#                dydx[ii][jj] = 1. / ( 1. - sum_fi )
#                jj = jj + 1
#            # jj == ii
#            g_i = y_area_lj[ll,ii]
#            f_i = x_area_lj[ll,ii]
#            f_j = x_area_lj[ll,jj]
#            sum_fi = np.sum( x_area_lj[ll,:ii+1] )
#            dydx[ii][jj] = 1. / f_i * ( 1. - sum_fi + f_i ) / ( 1 - sum_fi )
#
##        print "dydx", dydx
##        print "det", np.linalg.det( dydx )
#        dydx_det = dydx_det * np.linalg.det( dydx )
#    dxdy_det = 1. / dydx_det
#
#    if ( dxdy_det <= 0. ):
#        print "ERROR! ln_prior_area is NaN"
#        print "     ", dxdy_det
#        sys.exit()
#
#    ln_prior = np.log( dxdy_det )
#    return ln_prior
 
