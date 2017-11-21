import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rand
import numpy.linalg as lin

from utilities import get_log_grid

class CovModel:
    sigma = 10.0
    avg = 0.0
    length = 100.0

    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    def setSigma(self, sigma):
        self.sigma = sigma
    def setAvg(self, avg):
        self.avg = avg
    def setCorrLength(self, length):
        self.length = length
    def cov(self, x, y):
        return (self.sigma ** 2) * np.exp(-lin.norm(x-y) / self.length)
    def variogram(self, h):
        return (self.sigma ** 2) * (1-np.exp(-h / self.length))
    def rad_spec_dens(self, omega):
        return np.abs(omega * self.length * self.length / ((1 + (omega * self.length) ** 2) ** 1.5))
    def spec_dens(self, omega):
        return (self.sigma ** 2) / 2 * self.rad_spec_dens(omega)

def one_dim_dist(cov_model, eta, i):
    N = eta.size
    domega = 0.01 / np.sqrt(cov_model.length)
    omega = np.arange(0, 100 / cov_model.length, domega) + domega / 2
    M = omega.size
    #M = 30
    #omega, domega = get_log_grid(0.01 / cov_model.length, 1000 / cov_model.length, M)
    #plt.plot(omega, cov_model.spec_dens(omega), 'go', linewidth = 1)
    #plt.plot(centers, cov_model.spec_dens(centers), 'bo-')
    #plt.grid(True)
    #plt.show()

    rand.seed((i + 1) * 100)
    phi = 2 * np.pi * rand.sample(M)
    ddw = domega / 100
    dw =  ddw * rand.sample(M) - ddw / 2

    b = np.cos(np.outer(eta, omega + dw) + np.outer(np.ones(N), phi))
    return 2.0 * np.dot(b, np.sqrt(domega * cov_model.spec_dens(omega)))