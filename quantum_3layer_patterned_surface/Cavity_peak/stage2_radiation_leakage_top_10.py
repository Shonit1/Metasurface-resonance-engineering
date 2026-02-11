import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # MUST be before pyplot
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
import cma
import grcwa
from config import *

'''This locks in on the quasi BIC '''



# =========================================================
# FIXED PATTERNS & CONSTANTS
# =========================================================
P1 = np.array([[0,0,0],[1,1,1],[0,0,0]])
P2 = np.array([[1,1,0],[0,1,1],[0,0,1]])

a_fixed = 1.4641
lambda0 = 1.5

# =========================================================
# MATERIAL MODEL
# =========================================================
def epsilon_lambda(wavelength, _cache={}):
    if "li" not in _cache:
        data = pd.read_csv("C:\\Users\\ASUS\\Downloads\\Li-293K.csv")
        _cache["li"] = interp1d(
            data.iloc[:,0].values,
            data.iloc[:,1].values,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate"
        )
    return _cache["li"](wavelength)**2

# =========================================================
# GRID
# =========================================================
def get_epgrid_3x3(pattern, eps, a):
    ep = np.ones((Nx,Ny), dtype=complex) * eair
    dx = dy = a/3

    x = np.linspace(0,a,Nx,endpoint=False)
    y = np.linspace(0,a,Ny,endpoint=False)
    X,Y = np.meshgrid(x,y,indexing="ij")

    for i in range(3):
        for j in range(3):
            if pattern[i,j]:
                ep[(X>=i*dx)&(X<(i+1)*dx)&
                   (Y>=j*dy)&(Y<(j+1)*dy)] = eps
    return ep

# =========================================================
# RCWA SOLVERS
# =========================================================
def solver_r00(lam, h1, h2, hs, hsio2):
    try:
        eps_si = epsilon_lambda(lam)
        obj = grcwa.obj(nG, [a_fixed,0], [0,a_fixed],
                        1/lam, theta, phi, verbose=0)

        obj.Add_LayerUniform(0.1, eair)
        obj.Add_LayerGrid(h1, Nx, Ny)
        obj.Add_LayerUniform(hsio2, esio2)
        obj.Add_LayerGrid(h2, Nx, Ny)

        for _ in range(5):
            obj.Add_LayerUniform(hs, eps_si)
            obj.Add_LayerUniform(hsio2_dbr, esio2)

        obj.Add_LayerUniform(0.1, eair)
        obj.Init_Setup()

        ep1 = get_epgrid_3x3(P1, eps_si, a_fixed).flatten()
        ep2 = get_epgrid_3x3(P2, eps_si, a_fixed).flatten()
        obj.GridLayer_geteps(np.concatenate([ep1, ep2]))

        obj.MakeExcitationPlanewave(1,0,0,0)

        ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0)
        k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]
        nV = obj.nG

        return bi[k0]/ai[k0] if abs(ai[k0]) > abs(ai[k0+nV]) \
            else bi[k0+nV]/ai[k0+nV]
    except:
        return None

def solver_t00(lam, h1, h2, hs, hsio2):
    try:
        eps_si = epsilon_lambda(lam)
        obj = grcwa.obj(nG, [a_fixed,0], [0,a_fixed],
                        1/lam, theta, phi, verbose=0)

        obj.Add_LayerUniform(0.1, eair)
        obj.Add_LayerGrid(h1, Nx, Ny)
        obj.Add_LayerUniform(hsio2, esio2)
        obj.Add_LayerGrid(h2, Nx, Ny)

        for _ in range(5):
            obj.Add_LayerUniform(hs, eps_si)
            obj.Add_LayerUniform(hsio2_dbr, esio2)

        obj.Add_LayerUniform(0.1, eair)
        obj.Init_Setup()

        ep1 = get_epgrid_3x3(P1, eps_si, a_fixed).flatten()
        ep2 = get_epgrid_3x3(P2, eps_si, a_fixed).flatten()
        obj.GridLayer_geteps(np.concatenate([ep1, ep2]))

        obj.MakeExcitationPlanewave(1,0,0,0)

        ai, bi = obj.GetAmplitudes(which_layer=-1, z_offset=0)
        k0 = np.where((obj.G[:,0]==0)&(obj.G[:,1]==0))[0][0]
        nV = obj.nG

        t = bi[k0]/ai[k0] if abs(ai[k0]) > abs(ai[k0+nV]) \
            else bi[k0+nV]/ai[k0+nV]
        return abs(t)**2
    except:
        return None

# =========================================================
# LOSS FUNCTION
# =========================================================
history = []


# =========================================================
# EVENT THRESHOLDS
# =========================================================
PHASE_TARGET = np.pi
PHASE_TOL = 0.20          # radians
BIC_LEAK_THRESHOLD = 1e-8

# To avoid printing the same thing repeatedly
printed_resonances = set()
printed_bics = set()

def save_geometry(tag, slope, leak, x):
    h1, h2, hs, hsio2 = x

    msg = (
        f"[{tag}] "
        f"slope={slope:.4e}, leak={leak:.4e} | "
        f"h1={h1:.6f}, h2={h2:.6f}, "
        f"hs={hs:.6f}, hsio2={hsio2:.6f}"
    )

    print("\n" + "="*70)
    print(msg)
    print("="*70 + "\n")

    with open("event_geometries.txt", "a") as f:
        f.write(msg + "\n")





def loss_pull_pole(x):
    h1, h2, hs, hsio2 = x
    lambda0 = 1.5

    # wavelength window
    dlam = 2.5e-4
    lambdas = np.linspace(lambda0 - dlam, lambda0 + dlam, 41)

    Rs = []
    phis = []

    for lam in lambdas:
        r = solver_r00(lam, h1, h2, hs, hsio2)
        if r is None:
            return 1e6
        Rs.append(abs(r)**2)
        phis.append(np.angle(r))

    Rs = np.array(Rs)
    phis = np.unwrap(np.array(phis))

    # -----------------------------
    # PHYSICAL OBSERVABLES
    # -----------------------------

    # Phase derivative
    dphi = np.gradient(phis, lambdas)

    # Total phase winding (≈ Δφ)
    slope = np.trapz(np.abs(dphi), lambdas)

    # Radiation leakage (∝ Im pole)
    leakage = np.trapz(1.0 - Rs, lambdas)

    # Numerical stability
    curvature = np.std(np.gradient(dphi, lambdas))

    # -----------------------------
    # VISIBILITY CONSTRAINT
    # -----------------------------
    # Dark modes: slope → 0, leakage → 0
    # We explicitly forbid them
    phi_min = 1.0  # radians (≈ π/3)
    visibility_penalty = max(0.0, phi_min - slope)**2

    # -----------------------------
    # FINAL SCORE (∝ Q)
    # -----------------------------
    score = slope / (leakage + 1e-8)

    phi_target = np.pi
    phase_center_penalty = ((slope - phi_target)/phi_target)**2

    loss = (
        -score
        + 5.0 * visibility_penalty
        + 1.0 * phase_center_penalty
        + 0.1 * curvature
    )

    print(
        f"slope={slope:.2e}, "
        f"leak={leakage:.2e}, "
        f"score={score:.2e}"
    )

        # -----------------------------
    # EVENT TRIGGERS (SAVE GEOMETRY)
    # -----------------------------

    # A) Visible resonance (phase winding ~ π)
    if abs(slope - PHASE_TARGET) < PHASE_TOL:
        key = tuple(np.round(x, 6))
        if key not in printed_resonances:
            printed_resonances.add(key)
            save_geometry("RESONANCE", slope, leakage, x)

    # B) Near-BIC (radiation leakage → 0)
    if leakage < BIC_LEAK_THRESHOLD:
        key = tuple(np.round(x, 6))
        if key not in printed_bics:
            printed_bics.add(key)
            save_geometry("NEAR-BIC", slope, leakage, x)

    

    return loss


# =========================================================
# CMA-ES
# =========================================================
x0 = np.array([0.2368, 0.0800, 0.2399, 0.2152])
bounds = [[0.15,0.05,0.20,0.05],[0.30,0.25,0.30,0.40]]

es = cma.CMAEvolutionStrategy(x0, 0.02,
    {"bounds": bounds, "popsize": 12, "maxiter": 60})

while not es.stop():
    xs = es.ask()
    es.tell(xs, [loss_pull_pole(x) for x in xs])

# =========================================================
# TOP 10 GEOMETRIES
# =========================================================
history.sort(key=lambda x: x[0])
top10 = history[:10]

print("\n===== TOP 10 GEOMETRIES =====")
for i,(loss,x) in enumerate(top10):
    print(f"{i+1}: loss={loss:.3e}, h1,h2,hs,hsio2={x}")

# =========================================================
# PLOTTING
# =========================================================
import matplotlib
matplotlib.use("Agg")  # <-- IMPORTANT (no GUI freeze)

# =========================================================
# TOP 10 GEOMETRIES
# =========================================================
history.sort(key=lambda x: x[0])
top10 = history[:10]

print("\n===== TOP 10 GEOMETRIES =====")
for i, (loss, x) in enumerate(top10):
    print(f"{i+1}: loss={loss:.3e}, h1,h2,hs,hsio2={x}")

# =========================================================
# PLOTTING
# =========================================================
dlam_plot = 3e-4
lams = np.linspace(lambda0 - dlam_plot,
                   lambda0 + dlam_plot, 401)

for idx, (loss, x) in enumerate(top10):
    print(f"\nPlotting geometry {idx+1}/10 ...")

    h1, h2, hs, hsio2 = x

    Rs, Ts, phis, lams_valid = [], [], [], []

    for lam in lams:
        r = solver_r00(lam, h1, h2, hs, hsio2)

        if r is None or not np.isfinite(r):
            continue

        # Transmission is OPTIONAL near BIC
        try:
            t = solver_t00(lam, h1, h2, hs, hsio2)
            Tval = abs(t)**2 if np.isfinite(t) else np.nan
        except:
            Tval = np.nan

        lams_valid.append(lam)
        Rs.append(abs(r)**2)
        Ts.append(Tval)
        phis.append(np.angle(r))

    if len(lams_valid) < 30:
        print("  ⚠ Too few valid points, skipping")
        continue

    lams_valid = np.array(lams_valid)
    Rs = np.array(Rs)
    Ts = np.array(Ts)
    phis = np.unwrap(np.array(phis))

    plt.figure(figsize=(14,4))

    plt.subplot(1,3,1)
    plt.plot(lams_valid, phis, lw=1.5)
    plt.xlabel("λ (µm)")
    plt.ylabel("arg(r)")
    plt.title("Phase vs λ")

    plt.subplot(1,3,2)
    plt.plot(lams_valid, Rs, lw=1.5)
    plt.xlabel("λ (µm)")
    plt.ylabel("R")
    plt.title("Reflectance")

    plt.subplot(1,3,3)
    plt.plot(lams_valid, Ts, lw=1.5)
    plt.xlabel("λ (µm)")
    plt.ylabel("T")
    plt.title("Transmittance")

    plt.suptitle(
        f"Geom #{idx+1} | loss={loss:.2e}\n"
        f"h1={h1:.4f}, h2={h2:.4f}, hs={hs:.4f}, hsio2={hsio2:.4f}"
    )

    plt.tight_layout()
    fname = f"geometry_{idx+1}.png"
    plt.savefig(fname, dpi=200)
    plt.close()

    print(f"  ✓ Saved {fname}")

