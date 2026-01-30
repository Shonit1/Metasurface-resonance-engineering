import numpy as np
import matplotlib.pyplot as plt
import grcwa
from config import *

# ============================================================
# USER SETTINGS
# ============================================================

wavelengths = np.linspace(1.4, 1.6, 60)   # microns
theta = 0.0
phi = 0.0


# ============================================================
# NUMERICAL SAFETY
# ============================================================

def snap_lattice(L, tol=1e-6):
    return np.round(L / tol) * tol


# ============================================================
# SOLVER
# ============================================================

def solver_system(nG, L1, L2, wavelength, hs, hsio2, p=1, s=0):

    

    freq = 1.0 / wavelength
    obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=0)

    # Incident medium
    obj.Add_LayerUniform(0.1, eair)

    # DBR stack
    for _ in range(5):
        obj.Add_LayerUniform(hs, es1)
        obj.Add_LayerUniform(hsio2, esio2)

    # Exit medium
    obj.Add_LayerUniform(0.1, eair)

    obj.Init_Setup()

    obj.MakeExcitationPlanewave(
        p_amp=p, p_phase=0,
        s_amp=s, s_phase=0,
        order=0
    )

    Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)
    Mi = obj.GetAmplitudes(which_layer=0, z_offset=0.0)

    return Ri, Ti, Mi, obj


# ============================================================
# r00 EXTRACTION (ROBUST)
# ============================================================

def reflection_r00(obj, Mi):
    ai, bi = Mi

    k0 = np.where((obj.G[:, 0] == 0) & (obj.G[:, 1] == 0))[0][0]
    nV = obj.nG

    if abs(ai[k0]) > abs(ai[k0 + nV]):
        return bi[k0] / ai[k0]
    else:
        return bi[k0 + nV] / ai[k0 + nV]


# ============================================================
# SWEEP
# ============================================================

phases = []
Rvals = []
Tvals = []
wls_ok = []

for wl in wavelengths:
    try:
        Ri, Ti, Mi, obj = solver_system(
            nG=nG,
            L1=L1,
            L2=L2,
            wavelength=wl,
            hs=hs,
            hsio2=hsio2
        )

        r00 = reflection_r00(obj, Mi)

        phases.append(np.angle(r00))
        Rvals.append(Ri[0])
        Tvals.append(Ti[0])
        wls_ok.append(wl)

    except Exception:
        print(f"Skipped λ={wl:.4f} μm")


phases = np.unwrap(np.array(phases))
Rvals = np.array(Rvals)
Tvals = np.array(Tvals)
wls_ok = np.array(wls_ok)

# ============================================================
# PLOTS
# ============================================================

plt.figure(figsize=(7,5))
plt.plot(wls_ok, phases, 'o-')
plt.xlabel("Wavelength (μm)")
plt.ylabel("Reflection phase (rad)")
plt.title("DBR Reflection Phase vs Wavelength")
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5))
plt.plot(wls_ok, Rvals, 'o-', label="R")
plt.plot(wls_ok, Tvals, 'o-', label="T")
plt.xlabel("Wavelength (μm)")
plt.ylabel("Power")
plt.title("Reflectance & Transmittance vs Wavelength")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
