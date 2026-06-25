# 🛰️ Satellite Orbit Simulator

[![tests](https://github.com/seshankbagavath/satellite-orbit-simulator/actions/workflows/tests.yml/badge.svg)](https://github.com/seshankbagavath/satellite-orbit-simulator/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)


A high-fidelity orbital trajectory propagator written in Python. It integrates the
equations of motion for a satellite around a central body using the two-body model
augmented with **J2 oblateness** and **atmospheric drag** perturbations, then
visualizes the results with publication-quality charts.

> **Difficulty:** Advanced · **Core topics:** Orbital mechanics, numerical methods

## 🌐 Live Interactive Demo

**[▶ Launch the web app](https://seshankbagavath.github.io/satellite-orbit-simulator/)**

A zero-install, browser-based orbit explorer: drag sliders to reshape the orbit
in real time, load real satellites (ISS, Hubble, GPS, Molniya, geostationary,
polar), and toggle the J2 precession effect — all rendered live in 3D. The
orbital physics is ported from the validated Python engine and runs entirely in
your browser (no backend).

---

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
- **SGP4 validation** — propagate real TLEs with the industry-standard SGP4
  propagator and quantify accuracy against the numerical integrator (~3.4 km
  RMS over 24 h for the ISS).

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

## ✅ Validation Against Real Satellite Data (SGP4)

The numerical propagator is validated against **SGP4**, the industry-standard
analytic propagator, using **real Two-Line Element (TLE)** data. Both
propagators are seeded from the *same* epoch state, so the growing difference
isolates how the force model diverges from SGP4 over time.

For a 24-hour ISS propagation:

| Configuration | RMS position error vs SGP4 |
|---|---:|
| Two-body + **J2** | **~3.4 km** |
| Two-body only (no J2) | ~621 km |

Including the J2 oblateness term improves agreement by roughly **180×**,
concretely demonstrating why it is the dominant perturbation in low Earth
orbit. The residual few-km drift reflects effects SGP4 models that this
propagator does not (atmospheric drag, higher-order harmonics).

| Trajectory Overlay | Error Growth |
|---|---|
| ![overlay](fig_val_overlay.png) | ![error](fig_val_error.png) |

```python
from orbit_validation import run_validation_demo

# Live fetch from Celestrak; falls back to an embedded TLE if offline.
run_validation_demo(catalog_number=25544, duration_hours=24)
```

Run `orbit_validation.ipynb` for the full pipeline: TLE ingestion → SGP4
reference → numerical propagation → quantified comparison.

## ⚠️ Assumptions & Limitations

- Bound orbits only (`0 ≤ e < 1`); no hyperbolic/parabolic trajectories.
- Spherical-cap exponential atmosphere — engineering approximation, best in
  the ~150–1000 km regime.
- Earth orientation simplified to uniform rotation (no nutation/precession).
- No third-body, SRP, or higher-order gravity terms (extensible by design).

## 🧪 Tests

The engine ships with a `pytest` suite (20 tests) covering element↔state
round trips, known orbital values (ISS period, geostationary period, Kepler's
third law), conservation laws (energy and angular momentum in the two-body
model), J2 perturbation behavior, and input validation. Continuous integration
runs them on Python 3.10–3.12 via GitHub Actions on every push.

```bash
pip install -r requirements.txt pytest
pytest
```

## 📁 Project Structure

```
satellite-orbit-simulator/
├── .github/workflows/
│   └── tests.yml                # CI: runs pytest on Python 3.10–3.12
├── docs/
│   └── index.html               # Interactive web app (GitHub Pages)
├── tests/
│   └── test_orbit_simulator.py  # pytest suite (20 tests)
├── satellite_orbit_simulator.py     # Core library + demos
├── satellite_orbit_simulator.ipynb  # Core Colab notebook
├── orbit_validation.py              # Tier-1: SGP4 validation upgrade
├── orbit_validation.ipynb           # Validation Colab notebook
├── pytest.ini
├── requirements.txt
├── README.md
└── LICENSE
```

The web app lives in `docs/` because GitHub Pages can serve a site directly
from that folder — see deployment steps below.

## 📜 License

MIT — see [LICENSE](LICENSE).
