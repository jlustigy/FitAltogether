import numpy as np

def add_gaussnoise( data, noiselevel ):

    noise = data * noiselevel * np.random.normal( 0, 1, [ len(data), len(data[0]) ] )
    data_with_noise = data + noise
    return data_with_noise

