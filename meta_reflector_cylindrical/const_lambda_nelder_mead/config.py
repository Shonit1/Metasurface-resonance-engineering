import numpy as np
import grcwa

a = 0.3885776666186527



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

nG = 31

theta = np.pi/180 * 0
phi = 0

Nx = 120
Ny = 120

hs = w/(4*ns)
hsio2 = 0.2741874778683301

h = 0.057776795253681956
r = 0.1747241335989025