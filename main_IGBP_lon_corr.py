import numpy as np

FILENAME = 'IGBP_lon.txt'

lon_l   = np.loadtxt( FILENAME, usecols=(0,) )
area_lk = np.loadtxt( FILENAME, usecols=(1,2,3) )

corrcoeff_lk = np.zeros( [ len( lon_l ), len( area_lk.T ) ] )
for kk in xrange( len( area_lk.T ) ):

#    area_l = area_lk.T[kk] - np.average( area_lk.T[kk] )
    area_l = area_lk.T[kk]

    # result = np.correlate( area_lk.T[kk], area_lk.T[kk], mode='full' )
    # auto-corraltion function
    for dl in xrange( len( lon_l ) ):

        corrcoeff_lk[dl,kk] = np.dot( area_l, np.roll( area_l, dl ) ) / np.dot( area_l, area_l )
#        corrcoeff_lk[dl,kk] = np.dot( area_l, np.roll( area_l, dl ) )


for ll in xrange( len( lon_l ) ):
    print 1.* ll / len( lon_l ) * 360. ,

    for kk in xrange( len( area_lk.T ) ):
        print corrcoeff_lk[ll,kk],

    print ''
