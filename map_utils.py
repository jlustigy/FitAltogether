import numpy as np

#---------------------------------------------------
def generate_tex_names(n_type, n_band, n_slice):
    """
    Generate an array of Latex strings for each parameter in the
    X and Y vectors.

    Returns
    -------
    Y_names : array
        Non-physical fitting parameters
    X_names : array
        Physical parameters for Albedo and Surface Area Fractions
    """
    # Create list of strings for Y parameter names
    btmp = []
    gtmp = []
    for i in range(n_type):
        for j in range(n_band):
            btmp.append(r"b$_{"+str(i+1)+","+str(j+1)+"}$")
    for j in range(n_type - 1):
        for i in range(n_slice):
            gtmp.append(r"g$_{"+str(i+1)+","+str(j+1)+"}$")
    Y_names = np.concatenate([np.array(btmp), np.array(gtmp)])

    # Create list of strings for X parameter names
    Atmp = []
    Ftmp = []
    for i in range(n_type):
        for j in range(n_band):
            Atmp.append(r"A$_{"+str(i+1)+","+str(j+1)+"}$")
    for j in range(n_type):
        for i in range(n_slice):
            Ftmp.append(r"F$_{"+str(i+1)+","+str(j+1)+"}$")
    X_names = np.concatenate([np.array(Atmp), np.array(Ftmp)])

    return Y_names, X_names
