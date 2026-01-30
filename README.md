# Metasurface Resonance Engineering

This repository contains research code developed for the **design, simulation, and optimization of dielectric metasurfaces**, with an emphasis on engineering optical resonances through geometry, material choice, and multilayer patterning.

The work focuses on understanding resonance linewidths, phase dispersion, and modal behavior using rigorous electromagnetic simulations.

---

## 🧠 Project Overview

Metasurfaces are subwavelength structured systems that enable fine control over light–matter interaction.  
This repository documents a collection of numerical studies aimed at:

- Engineering narrow resonances and high-Q behavior
- Studying phase and dispersion near resonant modes
- Exploring multilayer and patterned metasurface geometries
- Linking geometric parameters to modal confinement and loss mechanisms
- Using optimization methods to refine resonant responses

The code reflects **research workflows**, not a packaged software library.

---

## 🧪 Methods and Tools

Electromagnetic simulations are performed using **rigorous coupled-wave analysis (RCWA)**, primarily through the Python package **grcwa**.

Key tools used:
- Python
- grcwa (RCWA simulations)
- NumPy, SciPy, Matplotlib
- CMA-ES for selected optimization tasks

Optimization is guided by physical intuition and spectral analysis rather than blind parameter searches.

---

## 🗂 Repository Structure

Metasurface-resonance-engineering/
├── Simple Bragg Grating/                             # 1D periodic structures and Bragg resonances
├── ellipse/                                          # Elliptical unit-cell metasurface studies
├── meta_reflector_cylindrical/                          # Cylindrical reflector metasurface designs
├── quantum_3layer_patterned_surface/                      # Multilayer patterned metasurfaces and resonant modes
├── outcmaes/                                               # CMA-ES based optimization runs and outputs
├── sqrt_lambda_lmbda0_CMA/                                  # Scaling-based resonance optimization studies
├── figures/                                                 # Representative simulation results
└── README.md
