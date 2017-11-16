import numpy as np
from scipy.spatial.distance import pdist, squareform

def get_log_grid(a, b, size):
    step_log = np.log(b / a) / (float)(size)
    x_prev = a * np.logspace(0, size, num=size, base=np.exp(step_log), endpoint=False)
    centers = x_prev * (np.exp(step_log) + 1.0) / 2.0
    widths = x_prev * (np.exp(step_log) - 1.0)
    return centers, widths

def pairwise( data ):
    '''
    Input:  (data) NumPy array where the first two columns
                   are the spatial coordinates, x and y
    '''
    # determine the size of the data
    npoints, cols = data.shape
    # give a warning for large data sets
    if npoints > 10000:
        print("You have more than 10,000 data points, this might take a minute.")
    # return the square distance matrix
    return squareform( pdist( data[:,:2] ) )

def lagindices( pwdist, lag, tol ):
    '''
    Input:  (pwdist) square NumPy array of pairwise distances
            (lag)    the distance, h, between points
            (tol)    the tolerance we are comfortable with around (lag)
    Output: (ind)    list of tuples; the first element is the row of
                     (data) for one point, the second element is the row
                     of a point (lag)+/-(tol) away from the first point,
                     e.g., (3,5) corresponds fo data[3,:], and data[5,:]
    '''
    # grab the coordinates in a given range: lag +/- tolerance
    i, j = np.where( ( pwdist >= lag - tol )&( pwdist < lag + tol ) )
    # zip the coordinates into a list
    indices = zip( i, j )
    # take out the repeated elements,
    # since p is a *symmetric* distance matrix
    indices = np.array([ i for i in indices if i[1] > i[0] ])
    return indices

def semivariance( data, indices ):
    '''
    Input:  (data)    NumPy array where the fris t two columns
                      are the spatial coordinates, x and y, and
                      the third column is the variable of interest
            (indices) indices of paired data points in (data)
    Output:  (z)      semivariance value at lag (h) +/- (tol)
    '''
    # take the squared difference between
    # the values of the variable of interest
    z = [ ( data[i,2] - data[j,2] )**2.0 for i,j in indices ]
    # the semivariance is half the mean squared difference
    return np.mean( z ) / 2.0
