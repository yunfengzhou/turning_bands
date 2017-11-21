import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

from utilities import pairwise, lagindices, semivariance
from cov_models import *

N = 100
M = 100
max_x = 1000.0
max_y = 1000.0
dx = max_x / N 
dy = max_y / M

x,y = np.mgrid[0:max_x:dx, 0:max_y:dy]
x += dx / 2
y += dy / 2

lines_num = 16
dphi = 2.0 * np.pi / lines_num
lines_phi = np.arange(0, 2.0 * np.pi, dphi)
proj = (np.outer(np.cos(lines_phi),x) + np.outer(np.sin(lines_phi), y))#.reshape((lines_num, N, M))

K = 4 * (np.max([N, M]) + 1)
max_eta = np.sqrt((max_x + dx) ** 2 + (max_y + dy) ** 2)
h_eta = (float)(2.0 * max_eta / K) 
eta = np.mgrid[-max_eta:max_eta:h_eta] + h_eta / 2.0

cov_model = CovModel()
z_s = np.zeros( (lines_num, K) )
z = np.zeros( (N, M) )
for i in range(0, lines_num):
    z_s[i] = one_dim_dist(cov_model, eta, i)
    print("sigma = ", np.sqrt(np.var(z_s[i])))
    f = interpolate.interp1d(eta, z_s[i], kind = 'nearest', assume_sorted = True)
    z += (f(proj[i])).reshape(N, M)

z /= np.sqrt((float)(lines_num))

# point variance
print("\nsigma = ", np.sqrt(np.var(z.ravel())))

# resulted variogram
data = np.array([x[:(int)(N/2), :(int)(M/2)].ravel(), y[:(int)(N/2), :(int)(M/2)].ravel(), z[:(int)(N/2), :(int)(M/2)].ravel()]).transpose()
#data = np.array([x.ravel(), y.ravel(), z.ravel()]).transpose()
pwdist = pairwise( data )
tol = dx / 2.0
lags = np.arange(0.7 * dx, cov_model.length * 6.0, 2.0 * tol)
y_model = cov_model.variogram(lags)
index = [ lagindices( pwdist, lag, tol ) for lag in lags ]
v = [ semivariance( data, indices ) for indices in index ]
pts_lag_size = [len(indices) for indices in index]

fig = plt.figure()                                            
ax = fig.add_subplot(2, 1, 1)
ax.set_xticks(lags + tol, minor=True)
ax.xaxis.grid(True, which='minor')
ax.plot(lags, v, '-o', markersize=5)
ax.plot(lags, y_model, '-o', markersize=5)

ax = fig.add_subplot(2, 1, 2)
ax.set_xticks(lags + tol, minor=True)
ax.xaxis.grid(True, which='minor')
ax.plot(lags, pts_lag_size)
plt.show()

fig1 = plt.figure()    
plt.imshow(z)
plt.show()
 

