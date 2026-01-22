import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import grcwa
from config import *




lambdas = np.linspace(1.2, 2.1, 30)  # µm



lambda_min = 1.4
lambda_max = 1.6
window_mask = (lambdas >= lambda_min) & (lambdas <= lambda_max)



def epsilon_lambda(wavelength, _cache={}):
    if "interp" not in _cache:
        data = pd.read_csv("C:\\Users\\ASUS\\Downloads\\Li-293K.csv")
        wl = data.iloc[:, 0].values
        n = data.iloc[:, 1].values

        _cache["interp"] = interp1d(
            wl, n,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate"
        )

    n_val = _cache["interp"](wavelength)
    return n_val**2




def get_epgrids_cylinder(r, et):
    x0 = np.linspace(0, L1[0], Nx, endpoint=False)
    y0 = np.linspace(0, L2[1], Ny, endpoint=False)
    x, y = np.meshgrid(x0, y0, indexing='ij')

    x_c = x - L1[0] / 2
    y_c = y - L2[1] / 2

    mask = (x_c**2 + y_c**2) < r**2

    epgrid = np.ones((Nx, Ny), dtype=complex) * eair
    epgrid[mask] = et

    return epgrid



def solver_system(f, r, h, hsio2):

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

    epgrid = get_epgrids_cylinder(r, es).flatten()
    obj.GridLayer_geteps(epgrid)

    obj.MakeExcitationPlanewave(
        p_amp=1, p_phase=0,
        s_amp=0, s_phase=0,
        order=0
    )

    ai, bi = obj.GetAmplitudes(which_layer=0, z_offset=0.0)

    k0 = np.where((obj.G[:, 0] == 0) & (obj.G[:, 1] == 0))[0][0]
    nV = obj.nG

    if np.abs(ai[k0]) > np.abs(ai[k0 + nV]):
        a00 = ai[k0]
        b00 = bi[k0]
    else:
        a00 = ai[k0 + nV]
        b00 = bi[k0 + nV]

    return b00 / a00




def compute_phase_spectrum(r, h, hsio2):
    phis = []

    for lam in lambdas:
        f = 1 / lam
        r00 = solver_system(f, r, h, hsio2)
        phis.append(np.angle(r00))

    phis = np.unwrap(np.array(phis))
    phis -= phis[0]

    return phis




def objective(params):
    r, h, hsio2 = params

    # lattice-safe radius
    r_max = 0.5 * min(L1[0], L2[1])

    if not (0.05 <= r <= min(0.5, r_max)):
        return 1e3
    if not (0.001 <= h <= 1.2):
        return 1e3
    if not (0.05 <= hsio2 <= 1.5):
        return 1e3

    phis = compute_phase_spectrum(r, h, hsio2)

    # ONLY FIT INSIDE DESIGN WINDOW
    lambdas_win = lambdas[window_mask]
    phis_win = phis[window_mask]

    x = np.sqrt(lambdas_win)

    A, B = np.polyfit(x, phis_win, 1)
    phis_fit = A * x + B

    return np.sqrt(np.mean((phis_win - phis_fit)**2))





r_vals = np.linspace(0.05, 0.45, 6)
h_vals = np.linspace(0.005, 0.8, 6)
hsio2_fixed = 0.3

results = []

print("Running piece-wise coarse sweep...\n")

for r in r_vals:
    for h in h_vals:
        loss = objective([r, h, hsio2_fixed])
        results.append((loss, r, h))
        print(f"r={r:.3f}, h={h:.3f} → loss={loss:.4f}")

results.sort(key=lambda x: x[0])
best_loss, best_r, best_h = results[0]

print("\nBest coarse geometry:")
print("loss =", best_loss)
print("r =", best_r)
print("h =", best_h)





x0 = [best_r, best_h, hsio2_fixed]

res = minimize(
    objective,
    x0=x0,
    method="Nelder-Mead",
    options={"maxiter": 40, "disp": True}
)

r_opt, h_opt, hsio2_opt = res.x

print("\nOptimized geometry:")
print("r =", r_opt)
print("h =", h_opt)
print("hsio2 =", hsio2_opt)
print("Final RMS loss:", res.fun)




phis = compute_phase_spectrum(r_opt, h_opt, hsio2_opt)

plt.figure(figsize=(6,4))
plt.plot(lambdas, phis, 'o-', label="RCWA phase")

# highlight fitting window
plt.axvspan(lambda_min, lambda_max, color='orange', alpha=0.2,
            label="√λ design window")

plt.xlabel("Wavelength (µm)")
plt.ylabel("Reflection phase (rad)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
