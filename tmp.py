
#===================================================
if __name__ == "__main__":

    # print start time
    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")

    # Create directory for this run
    startstr = now.strftime("%Y-%m-%d--%H-%M")
    run_dir = "mcmc_output/" + startstr + "/"
    os.mkdir(run_dir)
    print "Created directory:", run_dir

    # input data
    Obs_ij = np.loadtxt(INFILE)
    n_slice = len(Obs_ij)

    n_band = len(Obs_ij[0])
    Time_i = np.arange( n_slice )

    Obsnoise_ij = ( NOISELEVEL * Obs_ij )

    # set kernel
#    Kernel_il = kernel(Time_i, n_slice)
    Kernel_il = np.identity( n_slice )
#    Sigma_ll = np.identity(n_slice)

#    print 1/0
#    set initial condition
#    Y0_array = np.ones(N_TYPE*n_band+n_slice*(N_TYPE-1))
    X0_albd_kj = 0.3+np.zeros([N_TYPE, n_band])
    X0_area_lk = 0.1+np.zeros([n_slice, N_TYPE-1])

## albedo ( band x surface type )
#    X0_albd_kj = np.array( [[1.000000000000000056e-01, 9.000000000000000222e-01],
#                            [2.999999999999999889e-01, 5.000000000000000000e-01],
#                            [3.499999999999999778e-01, 5.999999999999999778e-01]]).T
#
#
## area fraction ( longitude slice x suface type )
#    X0_area_lk = np.array([[2.000000000000000111e-01, 8.000000000000000444e-01],
#                           [5.500000000000000444e-01, 4.499999999999999556e-01],
#                           [5.999999999999999778e-01, 4.000000000000000222e-01],
#                           [9.000000000000000222e-01, 9.999999999999997780e-02]])

#    X0_array = np.r_[ X0_albd_kj.flatten(), X0_area_lk.T[0].flatten() ]

#    X0_albd_kj[0,0:7] = np.array([0.35, 0.28, 0.28, 0.32, 0.40, 0.47, 0.35])
#    X0_albd_kj[1,0:7] = np.array([0.37, 0.28, 0.28, 0.30, 0.40, 0.48, 0.37])
#    X0_albd_kj[2,0:7] = np.array([0.67, 0.52, 0.40, 0.35, 0.33, 0.33, 0.32])
#    X0_albd_kj[0,0:7] = np.array([0.35, 0.28, 0.28, 0.32, 0.40, 0.47])
#    X0_albd_kj[1,0:7] = np.array([0.37, 0.28, 0.28, 0.30, 0.40, 0.48])
#    X0_albd_kj[2,0:7] = np.array([0.67, 0.52, 0.40, 0.35, 0.33, 0.33])
#    X0_area_lk[0,0] = np.array([0.3])
    Y0_array = transform_X2Y(X0_albd_kj, X0_area_lk, n_slice)
    n_dim = len(Y0_array)
    print '# of parameters', n_dim

    n_param = 0
    if FLAG_REG_AREA:
        Y0_array = np.append(Y0_array, [1.0, 1.0])
        n_param += 2
    if FLAG_REG_ALBD:
        Y0_array = np.append(Y0_array, [1.0, 1.0])
        n_param += 2

#    Y0_albd_kj = np.zeros([N_TYPE,  len(Obs_ij[0])])
#    Y0_area_lk = np.zeros([n_slice, N_TYPE-1])
#    Y0_area_lk[:,0] = 1.
#    Y0_list = [Y0_albd_kj, Y0_area_lk]
#    print "Y0_array", Y0_array

    if (n_param > 0):
        X_albd_kj, X_area_lk =  transform_Y2X(Y0_array[:-1*n_param], n_band)
    else:
        X_albd_kj, X_area_lk =  transform_Y2X(Y0_array, n_band)

#    print "X_area_lk", X_area_lk
#    print "X_albd_kj", X_albd_kj

    ########## use optimization for mcmc initial guesses ##########

    # minimize
    print "finding best-fit values..."
    data = (Obs_ij, Obsnoise_ij, Kernel_il, n_param, True, False)
    output = minimize(lnprob, Y0_array, args=data, method="Nelder-Mead")

    best_fit = output["x"]
    print "best-fit", best_fit

    data = (Obs_ij, Obsnoise_ij, Kernel_il, n_param, True, False)
    lnprob_bestfit = lnprob( output['x'], *data )
    BIC = 2.0 * lnprob_bestfit + len( output['x'] ) * np.log( len(Obs_ij.flatten()) )
    print 'BIC: ', BIC

    # Transform back to physical params
    X_albd_kj, X_area_lk =  transform_Y2X(output["x"], n_band)
    X_albd_kj_T = X_albd_kj.T
    #np.savetxt("X_area_lk", X_area_lk)
    #np.savetxt("X_albd_kj_T", X_albd_kj.T)
    bestfit = np.r_[ X_albd_kj.flatten(), X_area_lk.T.flatten() ]

    # Calculate residuals
    residuals = Obs_ij - np.dot( X_area_lk, X_albd_kj )
    print "residuals", residuals

    # Save initialization run as npz
    print "Saving:", run_dir+"initial_minimize.npz"
    np.savez(run_dir+"initial_minimize.npz", data=data, best_fity=best_fit, \
        lnprob_bestfit=lnprob_bestfit, BIC=BIC, X_area_lk=X_area_lk, \
        X_albd_kj_T=X_albd_kj_T, residuals=residuals, best_fitx = bestfit)
