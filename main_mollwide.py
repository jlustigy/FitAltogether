import numpy as np
import matplotlib.pyplot as plt

import sys


deg2rad = np.pi/180.

filename = sys.argv[1]

XX, YY, ZZ = np.loadtxt( filename, unpack=True )
ax = plt.subplot(111, projection = 'mollweide')
# ax.grid(True)
ax.contourf( XX*deg2rad, YY*deg2rad, ZZ*deg2rad, 1 )


plt.savefig( filename+'.png' )
