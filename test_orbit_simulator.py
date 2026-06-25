"""
Test suite for the Satellite Orbit Simulator.

These tests verify the orbital-mechanics engine against known analytic results
and internal consistency properties:

  * Keplerian element <-> state-vector round trips (machine precision).
  * Known orbital values (ISS period, geostationary period, circular velocity).
  * Conservation laws (energy and angular momentum in the two-body model).
  * Perturbation behavior (J2 produces the expected nodal regression).
  * Input validation and error handling.

Run with:  pytest -v
"""

import numpy as np
import pytest

from satellite_orbit_simulator import (
    EARTH,
    OrbitalElements,
    PropagatorConfig,
    elements_to_state,
    state_to_elements,
    orbital_period,
    specific_energy,
    accel_j2,
    ground_track,
    propagate,
    propagate_from_state,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def iss_elements():
    """An ISS-like low Earth orbit."""
    return OrbitalElements.from_degrees(
        a=EARTH.radius + 420.0, e=0.0006, i=51.64, raan=60.0, argp=0.0, nu=0.0
    )


@pytest.fixture
def circular_leo():
    """A simple circular equatorial-ish orbit, handy for analytic checks."""
    return OrbitalElements.from_degrees(
        a=EARTH.radius + 500.0, e=0.0, i=30.0, raan=0.0, argp=0.0, nu=0.0
    )


# ---------------------------------------------------------------------------
# Element <-> state conversions
# ---------------------------------------------------------------------------
class TestConversions:
    def test_roundtrip_recovers_elements(self, iss_elements):
        """elements -> state -> elements should return the original."""
        state = elements_to_state(iss_elements, EARTH.mu)
        recovered = state_to_elements(state, EARTH.mu)
        assert recovered.a == pytest.approx(iss_elements.a, rel=1e-9)
        assert recovered.e == pytest.approx(iss_elements.e, abs=1e-9)
        assert recovered.i == pytest.approx(iss_elements.i, abs=1e-9)

    def test_state_vector_has_six_components(self, iss_elements):
        state = elements_to_state(iss_elements, EARTH.mu)
        assert state.shape == (6,)

    def test_position_magnitude_matches_orbit_radius(self, circular_leo):
        """At true anomaly 0 (periapsis), |r| should equal a(1-e)."""
        state = elements_to_state(circular_leo, EARTH.mu)
        r = np.linalg.norm(state[:3])
        expected = circular_leo.a * (1 - circular_leo.e)
        assert r == pytest.approx(expected, rel=1e-9)

    def test_circular_orbit_speed(self, circular_leo):
        """A circular orbit's speed must equal sqrt(mu/a)."""
        state = elements_to_state(circular_leo, EARTH.mu)
        v = np.linalg.norm(state[3:])
        expected = np.sqrt(EARTH.mu / circular_leo.a)
        assert v == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Known analytic values
# ---------------------------------------------------------------------------
class TestKnownValues:
    def test_iss_period_about_93_minutes(self, iss_elements):
        period_min = orbital_period(iss_elements.a, EARTH.mu) / 60.0
        assert 92.0 < period_min < 94.0

    def test_geostationary_period_is_one_sidereal_day(self):
        """A geostationary orbit's period should be ~23.93 hours."""
        a_geo = 42164.0  # km, standard GEO semi-major axis
        period_hours = orbital_period(a_geo, EARTH.mu) / 3600.0
        assert period_hours == pytest.approx(23.93, abs=0.05)

    def test_period_scales_with_a_three_halves(self):
        """Kepler's third law: T proportional to a^(3/2)."""
        a1, a2 = 7000.0, 7000.0 * (2 ** (2 / 3))  # a2 chosen so T2 = 2*T1
        t1 = orbital_period(a1, EARTH.mu)
        t2 = orbital_period(a2, EARTH.mu)
        assert t2 / t1 == pytest.approx(2.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Conservation laws (two-body model)
# ---------------------------------------------------------------------------
class TestConservation:
    def test_energy_conserved_two_body(self, circular_leo):
        """With no perturbations, specific energy must stay essentially constant."""
        cfg = PropagatorConfig(use_j2=False, use_drag=False)
        period = orbital_period(circular_leo.a, EARTH.mu)
        result = propagate(circular_leo, 2 * period, cfg=cfg, n_samples=400)
        energy = specific_energy(result)
        drift = abs(energy.max() - energy.min())
        # Drift should be at the level of integrator tolerance, not physics.
        assert drift < 1e-4

    def test_angular_momentum_conserved_two_body(self, circular_leo):
        """Specific angular momentum |r x v| is constant in the two-body model."""
        cfg = PropagatorConfig(use_j2=False, use_drag=False)
        period = orbital_period(circular_leo.a, EARTH.mu)
        result = propagate(circular_leo, period, cfg=cfg, n_samples=300)
        r = result.positions.T
        v = result.velocities.T
        h = np.linalg.norm(np.cross(r, v), axis=1)
        assert np.std(h) / np.mean(h) < 1e-6

    def test_circular_orbit_stays_circular(self, circular_leo):
        """A circular two-body orbit should keep near-constant altitude."""
        cfg = PropagatorConfig(use_j2=False, use_drag=False)
        period = orbital_period(circular_leo.a, EARTH.mu)
        result = propagate(circular_leo, period, cfg=cfg, n_samples=300)
        alt = result.altitudes
        assert (alt.max() - alt.min()) < 1.0  # within 1 km


# ---------------------------------------------------------------------------
# Perturbations
# ---------------------------------------------------------------------------
class TestPerturbations:
    def test_j2_acceleration_points_inward_ish(self):
        """J2 acceleration should be small relative to the central term."""
        r_vec = np.array([EARTH.radius + 500.0, 0.0, 0.0])
        a_j2 = accel_j2(r_vec, EARTH)
        a_central = EARTH.mu / np.linalg.norm(r_vec) ** 2
        # J2 is a small correction: orders of magnitude below the main term.
        assert np.linalg.norm(a_j2) < 0.01 * a_central

    def test_j2_causes_nodal_regression(self):
        """For a prograde orbit, J2 should make the RAAN drift over time.

        We propagate with and without J2 and confirm the orbit planes differ.
        """
        elem = OrbitalElements.from_degrees(
            a=EARTH.radius + 600.0, e=0.001, i=51.6, raan=0.0, argp=0.0, nu=0.0
        )
        period = orbital_period(elem.a, EARTH.mu)
        r_j2 = propagate(elem, 15 * period,
                         cfg=PropagatorConfig(use_j2=True), n_samples=600)
        r_no = propagate(elem, 15 * period,
                         cfg=PropagatorConfig(use_j2=False), n_samples=600)
        # The angular-momentum direction (orbit normal) should rotate under J2
        # but stay fixed without it.
        def normal(res):
            r = res.positions[:, -1]
            v = res.velocities[:, -1]
            h = np.cross(r, v)
            return h / np.linalg.norm(h)
        # Without J2 the normal at start vs end barely moves.
        n0 = np.cross(r_no.positions[:, 0], r_no.velocities[:, 0])
        n0 = n0 / np.linalg.norm(n0)
        drift_no = np.linalg.norm(normal(r_no) - n0)
        drift_j2 = np.linalg.norm(normal(r_j2) - n0)
        assert drift_j2 > drift_no


# ---------------------------------------------------------------------------
# Ground track
# ---------------------------------------------------------------------------
class TestGroundTrack:
    def test_latitude_bounded_by_inclination(self, iss_elements):
        """Sub-satellite latitude can't exceed the orbit inclination."""
        period = orbital_period(iss_elements.a, EARTH.mu)
        result = propagate(iss_elements, period, n_samples=400)
        lat, lon = ground_track(result)
        inc_deg = np.degrees(iss_elements.i)
        assert lat.max() <= inc_deg + 1.0
        assert lat.min() >= -inc_deg - 1.0

    def test_longitude_within_bounds(self, iss_elements):
        period = orbital_period(iss_elements.a, EARTH.mu)
        result = propagate(iss_elements, period, n_samples=400)
        lat, lon = ground_track(result)
        assert lon.max() <= 180.0
        assert lon.min() >= -180.0


# ---------------------------------------------------------------------------
# Input validation & error handling
# ---------------------------------------------------------------------------
class TestValidation:
    def test_negative_semimajor_axis_rejected(self):
        with pytest.raises(ValueError):
            OrbitalElements.from_degrees(a=-1.0, e=0.0, i=0.0,
                                         raan=0.0, argp=0.0, nu=0.0)

    def test_hyperbolic_eccentricity_rejected(self):
        with pytest.raises(ValueError):
            OrbitalElements.from_degrees(a=7000.0, e=1.5, i=0.0,
                                         raan=0.0, argp=0.0, nu=0.0)

    def test_negative_duration_rejected(self, iss_elements):
        with pytest.raises(ValueError):
            propagate(iss_elements, duration_s=-100.0)

    def test_too_few_samples_rejected(self, iss_elements):
        with pytest.raises(ValueError):
            propagate(iss_elements, duration_s=1000.0, n_samples=1)

    def test_subsurface_state_rejected(self):
        """A state vector inside the Earth should raise."""
        bad_state = np.array([100.0, 0.0, 0.0, 0.0, 7.0, 0.0])  # 100 km from center
        with pytest.raises(ValueError):
            propagate_from_state(bad_state, duration_s=1000.0)

    def test_wrong_state_length_rejected(self):
        with pytest.raises(ValueError):
            propagate_from_state(np.array([1.0, 2.0, 3.0]), duration_s=1000.0)
