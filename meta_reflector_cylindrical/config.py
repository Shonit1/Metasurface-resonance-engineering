import numpy as np
import grcwa

a = 0.7206415591170998



L1 = [a,0]
L2 = [0,a]

w = 1.5
f = 1/1.5

ns = 3.45
nsio2 = 1.44
nair = 1

es = ns**2
esio2 = nsio2**2
eair = nair**2

nG = 101

theta = np.pi/180 * 0
phi = 0

Nx = 300
Ny = 300

hs = w/(4*ns)
hsio2 = 0.4613939870153822

h = 0.0745696719591017
r = 0.3035082512261301