# 🛰️ Satellite Orbit Simulator

A high-fidelity orbital trajectory propagator written in Python. It integrates the
equations of motion for a satellite around a central body using the two-body model
augmented with **J2 oblateness** and **atmospheric drag** perturbations, then
visualizes the results with publication-quality charts.

> **Difficulty:** Advanced · **Core topics:** Orbital mechanics, numerical methods

---

## ✨ Features

- **Keplerian ↔ Cartesian conversion** — full classical-elements to ECI state
  vector transform (and the inverse, for diagnostics), validated to machine
  precision.
- **High-order numerical propagation** — adaptive `DOP853` Runge–Kutta
  integration via SciPy with tight tolerances (`rtol=atol=1e-9`).
- **Perturbation models**
  - J2 zonal harmonic (nodal regression / apsidal precession).
  - Exponential-atmosphere drag with planetary co-rotation.
- **Re-entry detection** — terminal event stops propagation at the surface.
- **Analysis tools** — orbital period, specific-energy conservation diagnostic,
  and Earth-rotation-aware ground tracks.
- **Visualization suite** — 3D trajectory, sub-satellite ground track, and
  altitude/energy profiles.

## 📊 Example Output

| 3D Trajectory | Ground Track |
|---|---|
| ![3D orbit](fig_orbit3d.png) | ![Ground track](fig_groundtrack.png) |

![Altitude & energy](fig_alt_energy.png)

## 🚀 Quick Start

### Google Colab
Open `satellite_orbit_simulator.ipynb` and run all cells top to bottom.

### Local
```bash
pip install -r requirements.txt
python satellite_orbit_simulator.py
```

## 🧑‍💻 Usage

```python
from satellite_orbit_simulator import (
    OrbitalElements, PropagatorConfig, propagate, orbital_period,
    plot_orbit_3d, plot_ground_track, EARTH,
)

# Define an ISS-like low Earth orbit.
elem = OrbitalElements.from_degrees(
    a=EARTH.radius + 420, e=0.0006, i=51.64, raan=60, argp=0, nu=0,
)

period = orbital_period(elem.a, EARTH.mu)
result = propagate(elem, duration_s=3 * period,
                   cfg=PropagatorConfig(use_j2=True))

plot_orbit_3d(result)
plot_ground_track(result)
```

## 📐 Physics & Numerics

The state vector **x** = [r, v] evolves under:

```
r̈ = -μ·r/|r|³  +  a_J2(r)  +  a_drag(r, v)
```

| Term | Model |
|---|---|
| Two-body | Newtonian point-mass gravity |
| J2 | Second zonal harmonic, `J2 = 1.0826×10⁻³` |
| Drag | Exponential atmosphere (7.249 km scale height), co-rotating |

Integration uses SciPy's `solve_ivp` with the 8th-order `DOP853` scheme. The
specific-energy diagnostic provides a built-in check on integration quality:
in a pure two-body run the drift stays near machine precision; with J2 the
small periodic variation is the real physics, not numerical error.

## ⚠️ Assumptions & Limitations

- Bound orbits only (`0 ≤ e < 1`); no hyperbolic/parabolic trajectories.
- Spherical-cap exponential atmosphere — engineering approximation, best in
  the ~150–1000 km regime.
- Earth orientation simplified to uniform rotation (no nutation/precession).
- No third-body, SRP, or higher-order gravity terms (extensible by design).

## 📁 Project Structure

```
satellite-orbit-simulator/
├── satellite_orbit_simulator.py     # Library + demos
├── satellite_orbit_simulator.ipynb  # Colab notebook (cell-by-cell)
├── requirements.txt
├── README.md
└── LICENSE
```

## 📜 License

MIT — see [LICENSE](LICENSE).
