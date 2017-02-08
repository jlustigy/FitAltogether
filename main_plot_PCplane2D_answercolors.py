import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec
import sys, getopt

# Specify directory of run to analyze
MCMC_DIR = "mcmc_output/2016-07-13--11-59/"
DIR = "mcmc_output/"

# RESOLUTION=1000
#RESOLUTION=500
RESOLUTION=100

#--------------------------------------------------
def allowed_region( mcmc_dir ):

    # read PCs
    V_nj = np.loadtxt( mcmc_dir+'PCA_V_jn' ).T
    PC1 = V_nj[0]
    PC2 = V_nj[1]
    n_band = len( PC1 )
    band_ticks = np.arange( n_band )

    ave_j = np.loadtxt( mcmc_dir+'AVE_j' )

    print '# ave_j ', ave_j

    x_ticks = np.linspace(-0.4,0.2,RESOLUTION)
    y_ticks = np.linspace(-0.2,0.4,RESOLUTION)
    x_mesh, y_mesh, band_mesh = np.meshgrid( x_ticks, y_ticks, band_ticks, indexing='ij' )

    vec_mesh = x_mesh * PC1[ band_mesh ] + y_mesh * PC2[ band_mesh ] + ave_j[ band_mesh ]

    x_grid, y_grid = np.meshgrid( x_ticks, y_ticks, indexing='ij' )
    prohibited_grid = np.zeros_like( x_grid )

    for ii in xrange( len( x_ticks ) ) :
        for jj in xrange( len( y_ticks ) ) :

            print ii, jj, vec_mesh[ii][jj]

            if np.any( vec_mesh[ii][jj] < 0. ) :
                prohibited_grid[ii][jj] = 1
                if np.any( vec_mesh[ii][jj] > 1. ) :
                    prohibited_grid[ii][jj] = 3
            elif np.any( vec_mesh[ii][jj] > 1. ) :
                prohibited_grid[ii][jj] = 2
            else :
                prohibited_grid[ii][jj] = 0
    tmp=np.vstack([ x_grid.flatten(), y_grid.flatten(), prohibited_grid.flatten() ]).T
    np.savetxt( 'allowed_region', tmp )

    return x_grid, y_grid, prohibited_grid



def generate_cmap(colors):
    """
    copied from) http://qiita.com/kenmatsu4/items/fe8a2f1c34c8d5676df8
    """
    values = range(len(colors))
    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append( ( v/ vmax, c) )
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)


#===================================================
if __name__ == "__main__":


    # Read command line args
    myopts, args = getopt.getopt(sys.argv[1:],"d:b:")
    run = ""
    for o, a in myopts:
        # o == option
        # a == argument passed to the o
        # Get MCMC directory timestamp name
        if o == '-d':
            run=a
        else:
            print("Please specify run directory using -d: \n e.g. >python mcmc_physical.py -d 2016-07-13--11-59")
            sys.exit()

    MCMC_DIR = DIR + run + "/"

    # allowed region
    x_grid, y_grid, prohibited_grid = allowed_region( MCMC_DIR )
    print 'prohibited_grid.shape', prohibited_grid.shape

    # Load data
    try:
        tmp = np.loadtxt(MCMC_DIR+"PCA_U_in", unpack=True)
        data_x, data_y = tmp[0:2,:]
    except IOError:
        print "Run directory does not exist! Check -d argument."
        sys.exit()

    # Load shrinkwrap data
    try:
        vertices_x, vertices_y = np.loadtxt(MCMC_DIR+"shrinkwrap_final", unpack=True)
    except IOError:
        print "shrinkwrap_final does not exist!"
        sys.exit()

    # Load answers
    try:
        tmp = np.loadtxt(MCMC_DIR+"PCA_answer_projected" , unpack=True)
        answer_x, answer_y = tmp[0:2,:]
    except IOError:
        print "PCA_answer_projected does not exist!"
        sys.exit()

    # Load estimated
    try:
        tmp = np.loadtxt(MCMC_DIR+"albedo_median.txt")
        estimated_kj = tmp[:,0:3].T # number of surface types x band
        print 'estimated_kj.shape', estimated_kj.shape
    except IOError:
        print "albedo_median.txt does not exist!"
        sys.exit()
    # Load ave
    try:
        AVE_j = np.loadtxt(MCMC_DIR+"AVE_j" )
    except IOError:
        print "AVE_j does not exist! Check -d argument."
        sys.exit()
    ### 
    estimated_var_kj = estimated_kj - AVE_j
    print "estimated_var_jk"
    print estimated_var_kj.T

    # Load PCA
    try:
        V_jn = np.loadtxt(MCMC_DIR+"PCA_V_jn" )
    except IOError:
        print "Run directory does not exist! Check -d argument."
        sys.exit()

    # projection of estimated albedo
    # estimated_kj = C_kn x PCA_nj
    # C_kn = estimated_kj*(PCA_nj)^T
    projected_estimated_kn = np.dot( estimated_var_kj, V_jn )
    print "projected_estimated_kn"
    print projected_estimated_kn

    #-------------------------------------------------

    fig = plt.figure(figsize=(8, 6)) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2]) 
    ax1 = plt.subplot(gs[0], adjustable='box', aspect=1.0)


    cm = generate_cmap(['white', 'gray'])
    plt.pcolor( x_grid, y_grid, prohibited_grid, cmap=cm )
#    plt.imshow( prohibited_grid, interpolation='bilinear')
#    plt.contourf( x_grid, y_grid, prohibited_grid, 100, cmap=cm )

    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    ax1.set_xlim([-0.4,0.2])
    ax1.set_ylim([-0.2,0.4])
    ax1.set_xticks([ -0.4, -0.2, 0.0, 0.2 ])
    ax1.set_yticks([ -0.2, 0.0, 0.2, 0.4 ])

    plt.plot( vertices_x, vertices_y, 'orange' )
    plt.plot( data_x, data_y, 'k' )

    answer_x_loop = np.r_[ answer_x , answer_x[0] ]
    answer_y_loop = np.r_[ answer_y , answer_y[0] ]
    plt.plot( answer_x_loop, answer_y_loop, 'k--' )


    plt.scatter( projected_estimated_kn.T[0], projected_estimated_kn.T[1], marker='+' )

    plt.scatter( answer_x[0], answer_y[0], marker='o', c='blue' )
    plt.scatter( answer_x[1], answer_y[1], marker='s', c='red' )
    plt.scatter( answer_x[2], answer_y[2], marker='^', c='green' )

    plt.text( answer_x[0]-0.04, answer_y[0]-0.05, 'ocean', color='blue' ) # ocean
    plt.text( answer_x[1]-0.02, answer_y[1]+0.02, 'soil',  color='red' ) # soil
    plt.text( answer_x[2]-0.02, answer_y[2]+0.02, 'vege',  color='green' ) # vegetation


    ax2 = fig.add_subplot(gs[1], adjustable='box', aspect=1.0)

    ax2.set_xlim([-0.1,0.1])
    ax2.set_ylim([-0.05,0.05])
    ax2.set_xticks([-0.1,-0.05,0.0,0.05,0.1])
    ax2.set_yticks([-0.05,0.0,0.05])

#    cm = generate_cmap(['white', 'gray'])
#    plt.pcolor( x_grid, y_grid, prohibited_grid, cmap=cm )

    plt.xlabel('PC 1')
    plt.plot( data_x, data_y, 'k' )
    plt.plot( data_x, data_y, marker='.', c="black", label='data' )

    plt.plot( vertices_x, vertices_y, 'orange', label='shrink-wrapping' )

    answer_x_loop = np.r_[ answer_x , answer_x[0] ]
    answer_y_loop = np.r_[ answer_y , answer_y[0] ]
    plt.plot( answer_x_loop, answer_y_loop, 'k--' )

    plt.scatter( answer_x[0], answer_y[0], marker='o', c='blue', s=50 )
#    plt.scatter( answer_x[1], answer_y[1], marker='s', c='red' )
#    plt.scatter( answer_x[2], answer_y[2], marker='^', c='green' )

    plt.text( answer_x[0]-0.02, answer_y[0]-0.01, 'ocean', color='blue' ) # ocean
    plt.text( -0.09, 0.04, 'zoom in', color='black' ) # ocean

    handles, labels = ax2.get_legend_handles_labels()

    # reverse the order
    ax2.legend(handles, labels, loc='lower right', numpoints=1, fontsize=10 )

#    plt.savefig( MCMC_DIR+'mockdata_quadrature_PCplane.pdf' )
#    plt.savefig( MCMC_DIR+'mockdata_quadrature_PCplane.png', bbox_inches='tight' )
    plt.savefig( MCMC_DIR+'test.png', bbox_inches='tight' )

