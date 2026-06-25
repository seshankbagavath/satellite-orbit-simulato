# ==============================================================================
# SATELLITE ORBIT SIMULATOR
# ------------------------------------------------------------------------------
# A high-fidelity orbital trajectory propagator built around the two-body
# problem with optional J2 oblateness and atmospheric drag perturbations.
#
# Author:  <Your Name>
# License: MIT
#
# The notebook is organized into self-contained CELLS. In Google Colab, each
# "# === CELL N ===" header marks a new cell. Run them top to bottom.
# ==============================================================================


# ==============================================================================
# === CELL 1 — ENVIRONMENT SETUP & IMPORTS ===
# ==============================================================================
# Colab ships with numpy/scipy/matplotlib, so installs are usually unnecessary.
# Uncomment the line below if running in a fresh environment.
# !pip install numpy scipy matplotlib --quiet

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

# Reproducible, clean plotting defaults.
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titleweight": "bold",
})

np.set_printoptions(precision=6, suppress=True)
warnings.filterwarnings("ignore", category=UserWarning)

print("Environment ready. NumPy", np.__version__)


# ==============================================================================
# === CELL 2 — PHYSICAL CONSTANTS & CENTRAL-BODY MODEL ===
# ==============================================================================
@dataclass(frozen=True)
class CentralBody:
    """Container for the gravitational parameters of a central body.

    Attributes
    ----------
    name : str
        Human-readable identifier (e.g. "Earth").
    mu : float
        Standard gravitational parameter G*M  [km^3 / s^2].
    radius : float
        Mean equatorial radius  [km].
    j2 : float
        Second zonal harmonic coefficient (dimensionless oblateness term).
    omega : float
        Sidereal rotation rate  [rad / s]. Used for atmospheric co-rotation.
    """
    name: str
    mu: float
    radius: float
    j2: float
    omega: float


# Canonical Earth model (WGS-84 / EGM-derived values).
EARTH = CentralBody(
    name="Earth",
    mu=398_600.4418,      # km^3 / s^2
    radius=6_378.137,     # km
    j2=1.08262668e-3,     # dimensionless
    omega=7.2921159e-5,   # rad / s
)


# ==============================================================================
# === CELL 3 — ORBITAL ELEMENTS <-> STATE VECTOR CONVERSIONS ===
# ==============================================================================
@dataclass
class OrbitalElements:
    """Classical (Keplerian) orbital elements.

    Angles are stored in **radians** internally. Use ``from_degrees`` to build
    an instance from the more familiar degree-based inputs.
    """
    a: float       # semi-major axis            [km]
    e: float       # eccentricity               [-]
    i: float       # inclination                [rad]
    raan: float    # right ascension of asc.node[rad]
    argp: float    # argument of periapsis      [rad]
    nu: float      # true anomaly               [rad]

    @classmethod
    def from_degrees(cls, a: float, e: float, i: float,
                     raan: float, argp: float, nu: float) -> "OrbitalElements":
        """Construct from degrees; validates physical ranges."""
        if a <= 0:
            raise ValueError(f"Semi-major axis must be positive, got {a}.")
        if not (0.0 <= e < 1.0):
            raise ValueError(
                f"This simulator supports elliptical/circular orbits only "
                f"(0 <= e < 1); got e={e}."
            )
        return cls(
            a=float(a),
            e=float(e),
            i=np.radians(i),
            raan=np.radians(raan),
            argp=np.radians(argp),
            nu=np.radians(nu),
        )


def elements_to_state(elem: OrbitalElements, mu: float) -> NDArray[np.float64]:
    """Convert Keplerian elements to an ECI position/velocity state vector.

    Parameters
    ----------
    elem : OrbitalElements
    mu : float
        Gravitational parameter of the central body [km^3/s^2].

    Returns
    -------
    state : ndarray, shape (6,)
        [x, y, z, vx, vy, vz] in the Earth-Centered Inertial frame
        (km and km/s).
    """
    a, e, i = elem.a, elem.e, elem.i
    raan, argp, nu = elem.raan, elem.argp, elem.nu

    # Semi-latus rectum and orbital radius at the given true anomaly.
    p = a * (1.0 - e**2)
    r = p / (1.0 + e * np.cos(nu))

    # Position & velocity in the perifocal (PQW) frame.
    r_pqw = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    v_scale = np.sqrt(mu / p)
    v_pqw = np.array([-v_scale * np.sin(nu),
                      v_scale * (e + np.cos(nu)),
                      0.0])

    # Build the 3-1-3 rotation matrix PQW -> ECI.
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(argp), np.sin(argp)

    R = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci,  sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si,                 cw * si,                  ci],
    ])

    return np.concatenate([R @ r_pqw, R @ v_pqw])


def state_to_elements(state: NDArray[np.float64], mu: float) -> OrbitalElements:
    """Invert a state vector back into classical elements (for diagnostics)."""
    r_vec = state[:3]
    v_vec = state[3:]
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    # Angular momentum and node vectors.
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    n_vec = np.cross([0.0, 0.0, 1.0], h_vec)
    n = np.linalg.norm(n_vec)

    # Eccentricity vector.
    e_vec = (np.cross(v_vec, h_vec) / mu) - (r_vec / r)
    e = np.linalg.norm(e_vec)

    # Specific energy -> semi-major axis.
    energy = v**2 / 2.0 - mu / r
    a = -mu / (2.0 * energy)

    i = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))

    # Guard near-singular cases (equatorial / circular) with tiny-vector checks.
    raan = np.arccos(np.clip(n_vec[0] / n, -1.0, 1.0)) if n > 1e-12 else 0.0
    if n > 1e-12 and n_vec[1] < 0:
        raan = 2 * np.pi - raan

    if n > 1e-12 and e > 1e-12:
        argp = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1.0, 1.0))
        if e_vec[2] < 0:
            argp = 2 * np.pi - argp
    else:
        argp = 0.0

    if e > 1e-12:
        nu = np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1.0, 1.0))
        if np.dot(r_vec, v_vec) < 0:
            nu = 2 * np.pi - nu
    else:
        nu = 0.0

    return OrbitalElements(a=a, e=e, i=i, raan=raan, argp=argp, nu=nu)


# ==============================================================================
# === CELL 4 — PERTURBATION MODELS (J2 OBLATENESS + ATMOSPHERIC DRAG) ===
# ==============================================================================
def accel_j2(r_vec: NDArray[np.float64], body: CentralBody) -> NDArray[np.float64]:
    """Acceleration due to the J2 zonal harmonic (Earth's equatorial bulge).

    Returns a 3-vector [km/s^2]. This is the dominant non-spherical
    perturbation for low/medium Earth orbits and drives nodal regression
    and apsidal precession.
    """
    x, y, z = r_vec
    r = np.linalg.norm(r_vec)
    factor = 1.5 * body.j2 * body.mu * body.radius**2 / r**5
    z2_r2 = (z * z) / (r * r)

    ax = factor * x * (5.0 * z2_r2 - 1.0)
    ay = factor * y * (5.0 * z2_r2 - 1.0)
    az = factor * z * (5.0 * z2_r2 - 3.0)
    return np.array([ax, ay, az])


def accel_drag(r_vec: NDArray[np.float64], v_vec: NDArray[np.float64],
               body: CentralBody, bc: float) -> NDArray[np.float64]:
    """Atmospheric drag using an exponential density model.

    Parameters
    ----------
    bc : float
        Ballistic coefficient = (Cd * A / m)  [m^2 / kg]. Larger => more drag.

    Notes
    -----
    Uses a simple exponential atmosphere referenced to 7.249 km scale height,
    valid as an engineering approximation in the 150–1000 km regime. The
    relative velocity accounts for atmospheric co-rotation with the planet.
    """
    altitude = np.linalg.norm(r_vec) - body.radius
    if altitude <= 0:
        return np.zeros(3)  # below the surface: no meaningful drag term

    # Exponential atmosphere: rho0 at reference altitude h0.
    rho0 = 1.225e9        # kg/km^3 at sea level (1.225 kg/m^3 converted)
    h0 = 0.0              # reference altitude [km]
    scale_height = 7.249  # [km]
    rho = rho0 * np.exp(-(altitude - h0) / scale_height)

    # Velocity relative to the co-rotating atmosphere.
    omega_vec = np.array([0.0, 0.0, body.omega])
    v_rel = v_vec - np.cross(omega_vec, r_vec)
    v_rel_mag = np.linalg.norm(v_rel)

    # a_drag = -0.5 * rho * BC * |v_rel| * v_rel.  BC in m^2/kg -> km^2/kg *1e-6.
    bc_km = bc * 1e-6
    return -0.5 * rho * bc_km * v_rel_mag * v_rel


# ==============================================================================
# === CELL 5 — EQUATION OF MOTION & NUMERICAL PROPAGATOR ===
# ==============================================================================
@dataclass
class PropagatorConfig:
    """Toggles and parameters for the force model."""
    use_j2: bool = True
    use_drag: bool = False
    ballistic_coeff: float = 0.02   # Cd*A/m [m^2/kg]; typical small-sat value
    rtol: float = 1e-9
    atol: float = 1e-9
    method: str = "DOP853"          # high-order explicit Runge-Kutta


def equation_of_motion(t: float, state: NDArray[np.float64],
                       body: CentralBody, cfg: PropagatorConfig
                       ) -> NDArray[np.float64]:
    """Right-hand side of the orbital ODE: returns d(state)/dt.

    state = [x, y, z, vx, vy, vz]
    """
    r_vec = state[:3]
    v_vec = state[3:]
    r = np.linalg.norm(r_vec)

    # Two-body (Keplerian) acceleration: the dominant term.
    a_total = -body.mu * r_vec / r**3

    if cfg.use_j2:
        a_total = a_total + accel_j2(r_vec, body)
    if cfg.use_drag:
        a_total = a_total + accel_drag(r_vec, v_vec, body, cfg.ballistic_coeff)

    return np.concatenate([v_vec, a_total])


def _reentry_event(body: CentralBody) -> Callable:
    """Build a solve_ivp event that fires when the satellite hits the surface."""
    def event(t, state, *args):
        return np.linalg.norm(state[:3]) - body.radius
    event.terminal = True
    event.direction = -1
    return event


@dataclass
class PropagationResult:
    """Structured output of a propagation run."""
    t: NDArray[np.float64]               # time samples [s]
    states: NDArray[np.float64]          # shape (6, N): rows x,y,z,vx,vy,vz
    body: CentralBody
    reentered: bool = False
    raw: object = field(default=None, repr=False)  # full solve_ivp object

    @property
    def positions(self) -> NDArray[np.float64]:
        return self.states[:3, :]

    @property
    def velocities(self) -> NDArray[np.float64]:
        return self.states[3:, :]

    @property
    def altitudes(self) -> NDArray[np.float64]:
        return np.linalg.norm(self.positions, axis=0) - self.body.radius


def propagate_from_state(state0: NDArray[np.float64],
                         duration_s: float,
                         body: CentralBody = EARTH,
                         cfg: Optional[PropagatorConfig] = None,
                         n_samples: int = 2000) -> PropagationResult:
    """Numerically propagate forward in time from a raw ECI state vector.

    This is the low-level entry point. It accepts a 6-element state
    [x, y, z, vx, vy, vz] (km, km/s) directly, which is what you want when the
    initial condition comes from an external source such as an SGP4 ephemeris
    rather than from classical elements.

    Parameters
    ----------
    state0 : ndarray, shape (6,)
        Initial [position, velocity] in the inertial frame (km, km/s).
    duration_s : float
        Propagation horizon in seconds (must be > 0).
    body : CentralBody
        Central gravitating body. Defaults to Earth.
    cfg : PropagatorConfig, optional
        Force-model configuration. Defaults to two-body + J2.
    n_samples : int
        Number of evenly spaced output samples to return.

    Returns
    -------
    PropagationResult

    Raises
    ------
    ValueError
        On invalid inputs.
    RuntimeError
        If the underlying integrator fails to converge.
    """
    state0 = np.asarray(state0, dtype=float).reshape(-1)
    if state0.shape[0] != 6:
        raise ValueError(
            f"state0 must have 6 elements [x,y,z,vx,vy,vz], got {state0.shape[0]}."
        )
    if duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}.")
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")

    cfg = cfg or PropagatorConfig()

    # Sanity check: the initial position must clear the surface.
    if np.linalg.norm(state0[:3]) <= body.radius:
        raise ValueError(
            "Initial position is inside the central body. "
            "Increase the periapsis altitude."
        )

    t_eval = np.linspace(0.0, duration_s, n_samples)

    try:
        sol = solve_ivp(
            fun=equation_of_motion,
            t_span=(0.0, duration_s),
            y0=state0,
            method=cfg.method,
            t_eval=t_eval,
            args=(body, cfg),
            rtol=cfg.rtol,
            atol=cfg.atol,
            events=_reentry_event(body),
            max_step=duration_s / 50.0,  # cap step to keep drag sampling sane
        )
    except Exception as exc:  # numerical blow-up, bad RHS, etc.
        raise RuntimeError(f"Integration failed: {exc}") from exc

    if not sol.success:
        raise RuntimeError(f"Integrator did not converge: {sol.message}")

    reentered = sol.t_events is not None and len(sol.t_events[0]) > 0
    if reentered:
        print(f"[!] Satellite re-entered at t = {sol.t_events[0][0]/3600:.2f} h "
              f"(altitude reached 0 km).")

    return PropagationResult(
        t=sol.t,
        states=sol.y,
        body=body,
        reentered=reentered,
        raw=sol,
    )


def propagate(elem: OrbitalElements,
              duration_s: float,
              body: CentralBody = EARTH,
              cfg: Optional[PropagatorConfig] = None,
              n_samples: int = 2000) -> PropagationResult:
    """Numerically propagate an orbit forward in time from classical elements.

    Thin convenience wrapper over :func:`propagate_from_state` that first
    converts Keplerian elements into an ECI state vector.

    Parameters
    ----------
    elem : OrbitalElements
        Initial orbital state.
    duration_s : float
        Propagation horizon in seconds (must be > 0).
    body : CentralBody
        Central gravitating body. Defaults to Earth.
    cfg : PropagatorConfig, optional
        Force-model configuration. Defaults to two-body + J2.
    n_samples : int
        Number of evenly spaced output samples to return.

    Returns
    -------
    PropagationResult
    """
    state0 = elements_to_state(elem, body.mu)
    return propagate_from_state(state0, duration_s, body=body, cfg=cfg,
                                n_samples=n_samples)


# ==============================================================================
# === CELL 6 — ANALYSIS UTILITIES (PERIOD, ENERGY, GROUND TRACK) ===
# ==============================================================================
def orbital_period(a: float, mu: float) -> float:
    """Keplerian period [s] from the semi-major axis."""
    return 2.0 * np.pi * np.sqrt(a**3 / mu)


def specific_energy(result: PropagationResult) -> NDArray[np.float64]:
    """Specific orbital energy time-series [km^2/s^2].

    For a pure two-body model this is constant; deviations reveal both
    perturbation effects and numerical integration error.
    """
    r = np.linalg.norm(result.positions, axis=0)
    v = np.linalg.norm(result.velocities, axis=0)
    return v**2 / 2.0 - result.body.mu / r


def ground_track(result: PropagationResult) -> Tuple[NDArray, NDArray]:
    """Compute sub-satellite latitude/longitude including Earth rotation.

    Returns
    -------
    lat_deg, lon_deg : ndarray
        Geodetic-approximate latitude and longitude in degrees,
        longitude wrapped to [-180, 180].
    """
    x, y, z = result.positions
    r = np.linalg.norm(result.positions, axis=0)

    lat = np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0)))

    # Account for the planet rotating beneath the inertial trajectory.
    theta = result.body.omega * result.t
    lon_inertial = np.arctan2(y, x)
    lon = np.degrees(lon_inertial - theta)
    lon = (lon + 180.0) % 360.0 - 180.0  # wrap to [-180, 180]
    return lat, lon


# ==============================================================================
# === CELL 7 — VISUALIZATION SUITE ===
# ==============================================================================
def _draw_sphere(ax, radius: float, color: str = "#2a6fb0", alpha: float = 0.25):
    """Render a translucent wireframe sphere for the central body."""
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = radius * np.outer(np.cos(u), np.sin(v))
    ys = radius * np.outer(np.sin(u), np.sin(v))
    zs = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color=color, alpha=alpha,
                    linewidth=0, antialiased=True, zorder=0)


def plot_orbit_3d(result: PropagationResult, title: str = "Orbital Trajectory"):
    """Render the full 3D trajectory around the central body."""
    pos = result.positions
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    _draw_sphere(ax, result.body.radius)

    # Color the path by time to convey direction of motion.
    t_norm = (result.t - result.t[0]) / (result.t[-1] - result.t[0] + 1e-9)
    ax.scatter(pos[0], pos[1], pos[2], c=t_norm, cmap="plasma",
               s=2, alpha=0.9)
    ax.plot(pos[0], pos[1], pos[2], color="#d1495b", lw=0.4, alpha=0.5)

    ax.scatter(*pos[:, 0], color="lime", s=60, marker="o",
               label="Start", edgecolor="k", zorder=5)
    ax.scatter(*pos[:, -1], color="red", s=60, marker="X",
               label="End", edgecolor="k", zorder=5)

    ax.set_xlabel("X [km]"); ax.set_ylabel("Y [km]"); ax.set_zlabel("Z [km]")
    ax.set_title(f"{title}\n{result.body.name}-centered inertial frame")
    ax.legend(loc="upper right")

    # Force equal aspect ratio so the orbit isn't visually distorted.
    max_range = np.max(np.abs(pos)) * 1.05
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    plt.show()


def plot_ground_track(result: PropagationResult):
    """Plot the sub-satellite ground track on a flat lat/lon grid."""
    lat, lon = ground_track(result)
    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Break the line where it wraps around the +/-180 seam to avoid streaks.
    seam = np.where(np.abs(np.diff(lon)) > 180.0)[0]
    lon_plot = np.insert(lon.astype(float), seam + 1, np.nan)
    lat_plot = np.insert(lat.astype(float), seam + 1, np.nan)

    ax.plot(lon_plot, lat_plot, color="#d1495b", lw=1.2)
    ax.scatter(lon[0], lat[0], color="lime", s=60, edgecolor="k",
               zorder=5, label="Start")
    ax.scatter(lon[-1], lat[-1], color="red", s=60, marker="X",
               edgecolor="k", zorder=5, label="End")

    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.set_xlabel("Longitude [deg]"); ax.set_ylabel("Latitude [deg]")
    ax.set_title("Sub-Satellite Ground Track")
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_altitude_and_energy(result: PropagationResult):
    """Side-by-side altitude profile and energy-conservation diagnostic."""
    t_hours = result.t / 3600.0
    alt = result.altitudes
    energy = specific_energy(result)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(t_hours, alt, color="#2a6fb0", lw=1.4)
    ax1.fill_between(t_hours, alt, alpha=0.12, color="#2a6fb0")
    ax1.set_xlabel("Time [hours]"); ax1.set_ylabel("Altitude [km]")
    ax1.set_title("Altitude Profile")

    # Energy drift relative to the initial value (in parts, log-friendly).
    drift = energy - energy[0]
    ax2.plot(t_hours, drift, color="#7a2f8f", lw=1.4)
    ax2.set_xlabel("Time [hours]")
    ax2.set_ylabel(r"$\Delta$ Specific Energy [km$^2$/s$^2$]")
    ax2.set_title("Energy Conservation Diagnostic")
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.tight_layout()
    plt.show()


# ==============================================================================
# === CELL 8 — DEMO 1: LOW EARTH ORBIT (ISS-LIKE) WITH J2 ===
# ==============================================================================
def run_demo_leo():
    """Propagate an ISS-like LEO orbit for ~3 hours and visualize it."""
    print("=" * 70)
    print("DEMO 1 — Low Earth Orbit (ISS-like), two-body + J2")
    print("=" * 70)

    iss = OrbitalElements.from_degrees(
        a=EARTH.radius + 420.0,  # ~420 km altitude
        e=0.0006,
        i=51.64,                 # ISS inclination
        raan=60.0,
        argp=0.0,
        nu=0.0,
    )

    period = orbital_period(iss.a, EARTH.mu)
    print(f"Orbital period: {period/60:.2f} min")

    cfg = PropagatorConfig(use_j2=True, use_drag=False)
    result = propagate(iss, duration_s=3 * period, body=EARTH, cfg=cfg,
                       n_samples=3000)

    plot_orbit_3d(result, title="ISS-like LEO Trajectory")
    plot_ground_track(result)
    plot_altitude_and_energy(result)
    return result


# ==============================================================================
# === CELL 9 — DEMO 2: MOLNIYA (HIGHLY ECCENTRIC) ORBIT ===
# ==============================================================================
def run_demo_molniya():
    """Propagate a Molniya orbit — a classic highly-eccentric, critically
    inclined orbit used for high-latitude communications."""
    print("=" * 70)
    print("DEMO 2 — Molniya Orbit (e≈0.74, i=63.4°), two-body + J2")
    print("=" * 70)

    molniya = OrbitalElements.from_degrees(
        a=26_600.0,
        e=0.74,
        i=63.4,      # critical inclination: argp drift ~ 0
        raan=90.0,
        argp=270.0,
        nu=0.0,
    )

    period = orbital_period(molniya.a, EARTH.mu)
    print(f"Orbital period: {period/3600:.2f} h")

    result = propagate(molniya, duration_s=2 * period, body=EARTH,
                       cfg=PropagatorConfig(use_j2=True), n_samples=3000)

    plot_orbit_3d(result, title="Molniya Orbit")
    plot_ground_track(result)
    plot_altitude_and_energy(result)
    return result


# ==============================================================================
# === CELL 10 — DEMO 3: DRAG-DRIVEN ORBITAL DECAY ===
# ==============================================================================
def run_demo_decay():
    """Show a very low orbit decaying under atmospheric drag until re-entry."""
    print("=" * 70)
    print("DEMO 3 — Orbital Decay under Atmospheric Drag")
    print("=" * 70)

    low = OrbitalElements.from_degrees(
        a=EARTH.radius + 180.0,  # very low, drag-dominated
        e=0.001,
        i=51.6,
        raan=0.0,
        argp=0.0,
        nu=0.0,
    )

    # High ballistic coefficient (large area-to-mass) accelerates decay so the
    # effect is visible within a reasonable propagation window.
    cfg = PropagatorConfig(use_j2=True, use_drag=True, ballistic_coeff=0.5)
    result = propagate(low, duration_s=5 * 86_400.0, body=EARTH, cfg=cfg,
                       n_samples=4000)

    plot_orbit_3d(result, title="Decaying Low Orbit")
    plot_altitude_and_energy(result)

    if result.reentered:
        print("Result: satellite re-entered the atmosphere, as expected.")
    else:
        print("Result: orbit still decaying at end of window.")
    return result


# ==============================================================================
# === CELL 11 — MAIN ENTRY POINT ===
# ==============================================================================
if __name__ == "__main__":
    leo = run_demo_leo()
    molniya = run_demo_molniya()
    decay = run_demo_decay()
    print("\nAll demos complete.")
