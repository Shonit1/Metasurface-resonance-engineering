import numpy as np
import matplotlib.pyplot as plt
import grcwa
import cma
from config import *

# ============================================================
# USER SETTINGS
# ============================================================

wavelengths = np.linspace(1.50, 1.55, 10)   # microns
theta = 0.0
phi = 0.0

SIGMA0 = 1.5
MAXITER = 40

# IMPORTANT: lattice must be > max wavelength
BOUNDS_LOWER = [0.1, 1.2, 1.2]   # [hsio2, L1, L2]
BOUNDS_UPPER = [0.6,  1.5, 1.5]

# ============================================================
# NUMERICAL SAFETY
# ============================================================

def snap_lattice(L, tol=1e-6):
    return np.round(L / tol) * tol


# ============================================================
# SOLVER
# ============================================================

def solver_system(nG, L1, L2, wavelength, hs, hsio2, p=1, s=0):

    L1 = snap_lattice(L1)
    L2 = snap_lattice(L2)

    L10 = [L1, 0.0]
    L20 = [0.0, L2]

    freq = 1.0 / wavelength
    obj = grcwa.obj(nG, L10, L20, freq, theta, phi, verbose=0)

    obj.Add_LayerUniform(0.1, eair)

    for _ in range(5):
        obj.Add_LayerUniform(hs, es1)
        obj.Add_LayerUniform(hsio2, esio2)

    obj.Add_LayerUniform(0.1, eair)

    obj.Init_Setup()

    obj.MakeExcitationPlanewave(
        p_amp=p, p_phase=0,
        s_amp=s, s_phase=0,
        order=0
    )

    Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)
    Mi = obj.GetAmplitudes(which_layer=0, z_offset=0.0)

    return Ri, Mi, obj


# ============================================================
# CORRECT r00 EXTRACTION (YOUR METHOD)
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
# OBJECTIVE FUNCTION
# ============================================================

def loss_function(x):
    hsio2, L1, L2 = x

    phases = []
    Rs = []
    wls_ok = []

    for wl in wavelengths:
        try:
            Ri, Mi, obj = solver_system(
                nG=nG,
                L1=L1,
                L2=L2,
                wavelength=wl,
                hs=hs,
                hsio2=hsio2
            )

            r00 = reflection_r00(obj, Mi)
            phases.append(np.angle(r00))
            Rs.append(Ri[0])
            wls_ok.append(wl)

        except Exception:
            pass

    # 🚨 HARD SAFETY CHECKS (CRITICAL)
    if len(phases) < 5:
        return 1e6

    phases = np.unwrap(np.array(phases))
    Rs = np.array(Rs)
    wls_ok = np.array(wls_ok)

    # Linear fit: φ = A λ + B
    A, B = np.polyfit(wls_ok, phases, 1)

    slope_penalty = np.abs(A)
    reflectance_penalty = np.mean(np.maximum(0.95 - Rs, 0.0))**2

    loss = slope_penalty + 0 * reflectance_penalty

    print(
        f"A={A:+.3e} | "
        f"hsio2={hsio2:.3f}, L1={L1:.3f}, L2={L2:.3f} | "
        f"loss={loss:.3e}"
    )

    return loss


# ============================================================
# CMA-ES OPTIMIZATION
# ============================================================

x0 = [0.3, 1.4, 1.4]

es = cma.CMAEvolutionStrategy(
    x0,
    SIGMA0,
    {
        "bounds": [BOUNDS_LOWER, BOUNDS_UPPER],
        "maxiter": MAXITER,
        "popsize": 10,
        "verb_disp": 1
    }
)

es.optimize(loss_function)

hsio2_opt, L1_opt, L2_opt = es.result.xbest

print("\n===== OPTIMAL PARAMETERS =====")
print(f"hsio2 = {hsio2_opt:.4f}")
print(f"L1    = {L1_opt:.4f}")
print(f"L2    = {L2_opt:.4f}")


# ============================================================
# FINAL VALIDATION
# ============================================================

phases = []
Rs = []
wls_ok = []

for wl in wavelengths:
    try:
        Ri, Mi, obj = solver_system(
            nG=nG,
            L1=L1_opt,
            L2=L2_opt,
            wavelength=wl,
            hs=hs,
            hsio2=hsio2_opt
        )

        r00 = reflection_r00(obj, Mi)
        phases.append(np.angle(r00))
        Rs.append(Ri[0])
        wls_ok.append(wl)

    except Exception:
        print(f"Skipped λ={wl:.4f} μm")

if len(wls_ok) < 5:
    raise RuntimeError("Final design unstable across wavelength range")

phases = np.unwrap(np.array(phases))
Rs = np.array(Rs)
wls_ok = np.array(wls_ok)

A, B = np.polyfit(wls_ok, phases, 1)

# ============================================================
# PLOTS
# ============================================================

plt.figure(figsize=(7,5))
plt.plot(wls_ok, phases, 'o-', label="Phase")
plt.plot(wls_ok, A*wls_ok + B, '--', label=f"Fit A={A:.2e}")
plt.xlabel("Wavelength (μm)")
plt.ylabel("Reflection phase (rad)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5))
plt.plot(wls_ok, Rs, 'o-')
plt.xlabel("Wavelength (μm)")
plt.ylabel("Reflectance")
plt.grid()
plt.tight_layout()
plt.show()
