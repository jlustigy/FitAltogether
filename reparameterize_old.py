#---------------------------------------------------
def transform_Y2X_old(Y_array, n_band):

    Y_array = np.maximum(Y_array, -10)
    Y_array = np.minimum(Y_array, 10)

    Y_albd_kj = Y_array[0:N_TYPE*n_band].reshape([N_TYPE, n_band])
    X_albd_kj = np.exp( Y_albd_kj )/( 1 + np.exp( Y_albd_kj ) )
    Y_area_lk = Y_array[N_TYPE*n_band:].reshape([N_SLICE, N_TYPE-1])
    X_area_lk = np.zeros([len(Y_area_lk), len(Y_area_lk[0]) + 1 ])

    YY_area_lk = 1./( 1 + Y_area_lk )
    cumprodY_area_lk = np.cumprod( YY_area_lk, axis=1 )
    X_area_lk = Y_area_lk * cumprodY_area_lk

#    X_area_sum_lk = np.cumsum( X_area_lk, axis=1 ) - X_area_lk
#    X_area_lk = ( 1. - X_area_sum_lk ) * np.exp( Y_area_lk ) / ( 1 + np.exp( Y_area_lk ) )
    X_area_lk[:,-1] = 1. - np.sum( X_area_lk[:,:-1], axis=1 )
    return X_albd_kj, X_area_lk


#---------------------------------------------------
def transform_Y2X_old(Y_array, n_band):

    Y_array = np.maximum(Y_array, -10)
    Y_array = np.minimum(Y_array, 10)

    Y_albd_kj = Y_array[0:N_TYPE*n_band].reshape([N_TYPE, n_band])
    X_albd_kj = np.exp( Y_albd_kj )/( 1 + np.exp( Y_albd_kj ) )
    Y_area_lk = Y_array[N_TYPE*n_band:].reshape([N_SLICE, N_TYPE-1])
    X_area_lk = np.zeros([len(Y_area_lk), len(Y_area_lk[0]) + 1 ])
#    X_area_sum_lk = np.cumsum( x_area_lk, axis=1 )
    for kk in xrange( len(Y_area_lk[0]) ):
        X_area_lk[:,kk] = ( 1. - np.sum( X_area_lk[:,:kk], axis=1 ) ) * np.exp(Y_area_lk[:,kk]) / ( 1 + np.exp(Y_area_lk[:,kk]) )

    X_area_lk[:,-1] = 1. - np.sum( X_area_lk[:,:-1], axis=1 )
    return X_albd_kj, X_area_lk

 
