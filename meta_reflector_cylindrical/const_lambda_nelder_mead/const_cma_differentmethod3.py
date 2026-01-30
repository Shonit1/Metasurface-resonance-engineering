import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import cma
import grcwa
from config import *

# =========================================================
# GLOBAL TARGET
# =========================================================
LAM1 = 1.40
LAM2 = 1.6
N_FINE = 20

# =========================================================
# MATERIAL MODEL
# =========================================================
def epsilon_lambda(wavelength, _cache={}):
    if "interp" not in _cache:
        data = pd.read_csv("C:\\Users\\ASUS\\Downloads\\Li-293K.csv")
        wl = data.iloc[:, 0].values
        n = data.iloc[:, 1].values
        _cache["interp"] = interp1d(
            wl, n, kind="cubic",
            bounds_error=False,
            fill_value="extrapolate"
        )
    return _cache["interp"](wavelength)**2


# =========================================================
# GEOMETRY GRID
# =========================================================
def get_epgrids_cylinder(r, et, a):
    x0 = np.linspace(0, a, Nx, endpoint=False)
    y0 = np.linspace(0, a, Ny, endpoint=False)
    x, y = np.meshgrid(x0, y0, indexing="ij")
    x -= a / 2
    y -= a / 2
    mask = (x**2 + y**2) < r**2
    ep = np.ones((Nx, Ny), dtype=complex) * eair
    ep[mask] = et
    return ep


# =========================================================
# RCWA SOLVER
# =========================================================
def solver_system(f, r, h, hsio2, a):

    L1, L2 = [a, 0], [0, a]
    es = epsilon_lambda(1 / f)

    obj = grcwa.obj(nG, L1, L2, f, theta, phi, verbose=0)

    obj.Add_LayerUniform(0.1, eair)
    obj.Add_LayerGrid(h, Nx, Ny)
    obj.Add_LayerUniform(hsio2, esio2)

    for _ in range(5):
        obj.Add_LayerUniform(hs, es)
        obj.Add_LayerUniform(hsio2, esio2)

    obj.Add_LayerUniform(0.1, eair)
    obj.Init_Setup()

    obj.GridLayer_geteps(get_epgrids_cylinder(r, es, a).flatten())

    obj.MakeExcitationPlanewave(
        p_amp=1, p_phase=0,
        s_amp=0, s_phase=0,
        order=0
    )

    ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0)
    k0 = np.where((obj.G[:, 0] == 0) & (obj.G[:, 1] == 0))[0][0]
    nV = obj.nG

    return (
        bi[k0] / ai[k0]
        if abs(ai[k0]) > abs(ai[k0 + nV])
        else bi[k0 + nV] / ai[k0 + nV]
    )


# =========================================================
# PHASE AT SINGLE WAVELENGTH (NO UNWRAP)
# =========================================================
def phase_at_lambda(params, lam):
    r, h, hsio2, a = params
    return np.angle(solver_system(1 / lam, r, h, hsio2, a))


# =========================================================
# GEOMETRY PARAMETERIZATION (a >= 2r)
# =========================================================
geom_bounds = np.array([
    [0.05, 0.45],   # r
    [0.05, 0.40],   # h
    [0.05, 0.60],   # hsio2
    [0.00, 0.40],   # delta_a
])

def decode_geom(x):
    x = np.clip(x, 0.0, 1.0)
    raw = geom_bounds[:,0] + x*(geom_bounds[:,1]-geom_bounds[:,0])
    r, h, hsio2, delta_a = raw
    a = 2*r + delta_a
    return np.array([r, h, hsio2, a])


# =========================================================
# ==================== PHASE 1 ============================
# Two-point slope screening
# =========================================================
candidates = []

def loss_phase1(x):
    params = decode_geom(x)

    phi1 = phase_at_lambda(params, LAM1)
    phi2 = phase_at_lambda(params, LAM2)

    dphi = np.angle(np.exp(1j*(phi2 - phi1)))
    slope = dphi / (LAM2 - LAM1)

    candidates.append((slope**2, params))
    print(f"[P1] slope = {slope:.3e}")
    return slope**2


print("\n=== PHASE 1: COARSE SCREENING ===")

es1 = cma.CMAEvolutionStrategy(
    np.random.rand(4), 0.3,
    {"popsize": 12, "maxiter": 35}
)

while not es1.stop():
    xs = es1.ask()
    es1.tell(xs, [loss_phase1(x) for x in xs])

# Pick top 10
candidates = sorted(candidates, key=lambda x: x[0])[:10]
top_geometries = [c[1] for c in candidates]

print("\nTop 10 candidates saved.")

'''
# =========================================================
# ==================== PHASE 2 ============================
# RMS validation inside 50 nm band
# =========================================================
def rms_error(params):
    lambdas = np.linspace(LAM1, LAM2, N_FINE)
    phis = np.array([phase_at_lambda(params, lam) for lam in lambdas])
    phis = np.unwrap(phis)

    A, B = np.polyfit([LAM1, LAM2],
                      [phis[0], phis[-1]], 1)
    phi_line = A*lambdas + B

    return np.sqrt(np.mean((phis - phi_line)**2))


print("\n=== PHASE 2: BAND VALIDATION ===")

rms_list = []
for i, g in enumerate(top_geometries):
    rms = rms_error(g)
    rms_list.append((rms, g))
    print(f"Candidate {i}: RMS = {rms:.3e}")

best_geom = min(rms_list, key=lambda x: x[0])[1]
print("\nBest geometry after Phase 2:", best_geom)

'''


for i, g in enumerate(top_geometries):

    lambdas = np.linspace(LAM1, LAM2, N_FINE)
    phis = np.unwrap([phase_at_lambda(g, lam) for lam in lambdas])
    A, B = np.polyfit(lambdas, phis, 1)

    plt.plot(lambdas, phis, 'o-', label="Phase")
    plt.plot(lambdas, A*lambdas + B, '--', label="Linear fit")
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Phase (rad)")
    plt.title(f"Final slope A = {A:.3e}")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

'''
# =========================================================
# ==================== PHASE 3 ============================
# Local geometry refinement
# =========================================================
def loss_phase3(x):
    params = decode_geom(x)
    lambdas = np.linspace(LAM1, LAM2, N_FINE)
    phis = np.unwrap([phase_at_lambda(params, lam) for lam in lambdas])

    A, B = np.polyfit(lambdas, phis, 1)
    rms = np.std(phis - (A*lambdas + B))

    print(f"[P3] A = {A:.3e}, RMS = {rms:.3e}")
    return 200*A**2 + 2*rms


print("\n=== PHASE 3: LOCAL OPTIMIZATION ===")

x0 = (best_geom - np.array([0,0,0,2*best_geom[0]])) / \
     (geom_bounds[:,1]-geom_bounds[:,0])
x0 = np.clip(x0, 0.2, 0.8)

es3 = cma.CMAEvolutionStrategy(
    x0, 0.15,
    {"popsize": 8, "maxiter": 40}
)

while not es3.stop():
    xs = es3.ask()
    es3.tell(xs, [loss_phase3(x) for x in xs])

final_geom = decode_geom(es3.result.xbest)
print("\nFINAL GEOMETRY:", final_geom)


# =========================================================
# FINAL PLOT
# =========================================================
lambdas = np.linspace(LAM1, LAM2, N_FINE)
phis = np.unwrap([phase_at_lambda(final_geom, lam) for lam in lambdas])
A, B = np.polyfit(lambdas, phis, 1)

plt.plot(lambdas, phis, 'o-', label="Phase")
plt.plot(lambdas, A*lambdas + B, '--', label="Linear fit")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Phase (rad)")
plt.title(f"Final slope A = {A:.3e}")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
'''