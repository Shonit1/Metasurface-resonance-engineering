import numpy as np
import grcwa

a = 0.9



L1 = [a,0]
L2 = [0,a]

w = 1.5
f = 1/1.525

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
hsio2 = 0.3

h = 0.2
r = 0.4

hsio2_dbr = 0.2586