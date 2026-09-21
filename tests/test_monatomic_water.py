"""MonatomicWater tests with analytic and independent scalar energy oracles."""
from itertools import combinations
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_pdf import MonatomicWater

# Molinero-Moore two-body parameters, written out independently of the module
# under test (Angstrom, kcal/mol) for the analytic oracles below.
_A, _B, _EPS, _SIGMA, _CUT = 7.049556277, 0.6022245584, 6.189, 2.3925, 1.8


def _phi2(r):
    """mW two-body potential phi(r) in kcal/mol at distance r (Angstrom), float64."""
    rr = r / _SIGMA
    if rr >= _CUT:
        return 0.0
    return _A * _EPS * (_B * rr**-4 - 1.0) * math.exp(1.0 / (rr - _CUT))


def _dphi2(r, h=1e-6):
    """Central finite-difference slope of phi at r (float64; error ~1e-9)."""
    return (_phi2(r + h) - _phi2(r - h)) / (2.0 * h)


def _pair_config(r):
    """Two particles a distance r apart along x, far from the box walls."""
    return jnp.array([[5.0, 5.0, 5.0], [5.0 + r, 5.0, 5.0]])


def _reference_energy(x, box_length, min_distance=0.0, linearize_below=None):
    """Scalar float64 oracle for the equations in docs/monatomic_water.md.

    Enumerate unique pairs and, for each centre, unordered neighbour pairs.
    This deliberately uses neither production helpers/constants nor JAX's
    vectorized distance tensors, masking, or autodiff. Fixtures have distinct
    particles, so the physical angle needs no numerical norm regularization.
    Clipping and linearization apply only to the two-body distances.
    """
    x = np.asarray(x, dtype=np.float64)

    def displacement(i, j):
        delta = x[j] - x[i]
        return delta - box_length * np.rint(delta / box_length)

    energy = 0.0
    for i, j in combinations(range(len(x)), 2):
        r = max(float(np.linalg.norm(displacement(i, j))), min_distance)
        if linearize_below is not None and r < linearize_below:
            energy += _phi2(linearize_below) + (r - linearize_below) * _dphi2(linearize_below)
        else:
            energy += _phi2(r)

    cos0 = math.cos(math.radians(109.47))
    for centre in range(len(x)):
        neighbours = [j for j in range(len(x)) if j != centre]
        for j, k in combinations(neighbours, 2):
            a, b = displacement(centre, j), displacement(centre, k)
            ra, rb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            if ra >= _CUT * _SIGMA or rb >= _CUT * _SIGMA:
                continue
            cosine = float(np.dot(a, b)) / (ra * rb)
            radial = math.exp(1.2 / (ra / _SIGMA - _CUT) + 1.2 / (rb / _SIGMA - _CUT))
            energy += 23.15 * _EPS * (cosine - cos0)**2 * radial
    return energy


def _x(n=8, L=6.0, key=0, batch=4):
    return jax.random.uniform(jax.random.PRNGKey(key), (batch, n, 3), minval=0.0, maxval=L)


def _lattice_config(spacing=3.5, jitter=0.1, key=7):
    """8 well-separated particles (jittered 2x2x2 grid), neighbours within the
    cutoff. The exact mW potential is singular at true overlap, so the
    finite-gradient claim only holds for a non-overlapping config like this."""
    g = jnp.stack(
        jnp.meshgrid(jnp.arange(2), jnp.arange(2), jnp.arange(2), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3).astype(jnp.float32) * spacing
    return g + jitter * jax.random.normal(jax.random.PRNGKey(key), g.shape)


class TestInvariances:
    def test_output_shape(self):
        mw = MonatomicWater(n_particles=8, box_length=6.0)
        assert mw(_x()).shape == (4,)

    def test_wrong_shape_raises(self):
        mw = MonatomicWater(n_particles=8, box_length=6.0)
        with pytest.raises(ValueError, match=r"x.shape\[-2:\]"):
            mw(jnp.zeros((8, 2)))

    def test_translation_invariance(self):
        """Minimum-image energy is invariant under a global shift."""
        mw = MonatomicWater(n_particles=8, box_length=6.0, beta=0.5)
        x = _x(batch=1)[0]
        assert jnp.allclose(mw(x), mw(x + jnp.array([1.3, -0.7, 0.4])), atol=1e-3)

    def test_permutation_invariance(self):
        mw = MonatomicWater(n_particles=8, box_length=6.0, beta=0.5)
        x = _x(batch=1)[0]
        perm = jnp.array([3, 1, 4, 0, 2, 5, 7, 6])
        assert jnp.allclose(mw(x), mw(x[perm]), atol=1e-3)

    def test_pbc_box_shift(self):
        """Shifting one particle by a box vector leaves the energy unchanged."""
        L = 6.0
        mw = MonatomicWater(n_particles=8, box_length=L, beta=0.5)
        x = _x(batch=1)[0]
        x_shift = x.at[0].add(jnp.array([L, 0.0, 0.0]))
        assert jnp.allclose(mw(x), mw(x_shift), atol=1e-3)


class TestNumerics:
    def test_dim_property(self):
        assert MonatomicWater(n_particles=8, box_length=6.0).dim == 24

    def test_beta_scaling(self):
        x = _x(batch=1)[0]
        mw1 = MonatomicWater(n_particles=8, box_length=6.0, beta=1.0)
        mw2 = MonatomicWater(n_particles=8, box_length=6.0, beta=2.0)
        assert jnp.allclose(mw2(x), 2.0 * mw1(x), atol=1e-4)

    def test_gradient_finite(self):
        """Exact potential: gradient is finite for a non-overlapping config."""
        mw = MonatomicWater(n_particles=8, box_length=10.0, beta=0.5)
        x = _lattice_config()
        grad = jax.grad(mw)(x)
        assert grad.shape == (8, 3)
        assert bool(jnp.all(jnp.isfinite(grad)))

    def test_gradient_finite_with_overlap(self):
        """Training regime: a near-coincident pair would NaN the gradient via
        the 3-body sqrt and the 2-body core. With min_distance>0 (clips the
        core) and the eps-floored 3-body norm, the gradient stays finite -- the
        precondition for reverse-KL training. Cf. nflojax GNN coincident test.
        """
        x = _lattice_config().at[1].set(_lattice_config()[0] + 0.01)
        mw = MonatomicWater(n_particles=8, box_length=10.0, beta=0.5, min_distance=0.3)
        grad = jax.grad(mw)(x)
        assert bool(jnp.all(jnp.isfinite(grad)))

    def test_jit(self):
        mw = MonatomicWater(n_particles=8, box_length=6.0)
        x = _x(batch=1)[0]
        assert jnp.allclose(jax.jit(mw.__call__)(x), mw(x), atol=1e-4)

    def test_log_normalization_raises(self):
        with pytest.raises(NotImplementedError):
            MonatomicWater(n_particles=8, box_length=6.0).log_normalization()

    def test_two_body_cutoff(self):
        """A pair beyond the cutoff (1.8*sigma) contributes 0; within, nonzero."""
        mw = MonatomicWater(n_particles=2, box_length=20.0, beta=1.0)
        far = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])    # 5 > 1.8*sigma (4.31)
        near = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])   # 3 < 4.31
        assert jnp.allclose(mw(far), 0.0, atol=1e-6)
        assert float(mw(near)) != 0.0 and bool(jnp.isfinite(mw(near)))

    def test_three_body_minimal_at_tetrahedral(self):
        """The 3-body term is 0 at 109.47 deg and positive otherwise, so log p
        (= -beta*U) is maximal at the tetrahedral angle. The outer pair is kept
        beyond the cutoff so only the 3-body term varies with the angle.
        """
        import math

        sigma = 2.3925
        d = 1.3 * sigma                         # centre-neighbour: within cutoff
        mw = MonatomicWater(n_particles=3, box_length=40.0, beta=1.0)

        def cfg(theta_deg):
            t = math.radians(theta_deg)
            return jnp.array([
                [0.0, 0.0, 0.0],
                [d, 0.0, 0.0],
                [d * math.cos(t), d * math.sin(t), 0.0],
            ])

        e_tet = float(mw(cfg(109.47)))
        assert e_tet > float(mw(cfg(95.0)))
        assert e_tet > float(mw(cfg(130.0)))


class TestLinearize:
    """``linearize_below`` softens the two-body core for training.

    Below ``r_lin`` the two-body potential is its tangent line in r,
    ``phi(r_lin) + (r - r_lin) * phi'(r_lin)``; the squared distance is first
    clipped to ``min_distance**2``; the three-body term is untouched. Two
    particles have no three-body term, so a pair isolates the two-body part.
    """

    LIN = 1.2   # Linearization distance (Angstrom).

    @pytest.mark.parametrize("r", [0.6, 0.9, 1.1])
    def test_two_body_is_tangent_line_below(self, r):
        """Analytic oracle: tangent line of the hand-written phi at r_lin.

        Bug class: linearising in r**2 instead of r, or using the slope with
        respect to r**2 (off by a factor 2 * r_lin).
        """
        mw = MonatomicWater(n_particles=2, box_length=20.0, beta=1.0,
                            linearize_below=self.LIN)
        u = float(-mw(_pair_config(r)))
        expected = _phi2(self.LIN) + (r - self.LIN) * _dphi2(self.LIN)
        assert math.isclose(u, expected, rel_tol=1e-4)

    def test_unchanged_above_linearize_point(self):
        """With every pair farther apart than r_lin the energy is the exact one."""
        x = _lattice_config()                     # all pair distances > 3 Angstrom
        exact = MonatomicWater(n_particles=8, box_length=10.0, beta=0.5)
        soft = MonatomicWater(n_particles=8, box_length=10.0, beta=0.5,
                              linearize_below=self.LIN)
        assert jnp.allclose(soft(x), exact(x), rtol=1e-6)

    def test_slope_below_equals_phi_prime_at_linearize_point(self):
        """The force below r_lin is constant and equals phi'(r_lin): the
        potential is C1 at r_lin. Checked with autodiff against the hand-written
        finite-difference slope.
        """
        mw = MonatomicWater(n_particles=2, box_length=20.0, beta=1.0,
                            linearize_below=self.LIN)
        slope = jax.grad(lambda r: -mw(_pair_config(r)))
        for r in (0.7, 1.0, 1.19):
            assert math.isclose(float(slope(r)), _dphi2(self.LIN), rel_tol=1e-3)

    def test_value_continuous_at_linearize_point(self):
        """No jump at r_lin: values just below and above agree to O(delta)."""
        mw = MonatomicWater(n_particles=2, box_length=20.0, beta=1.0,
                            linearize_below=self.LIN)
        delta = 1e-3
        below = float(-mw(_pair_config(self.LIN - delta)))
        above = float(-mw(_pair_config(self.LIN + delta)))
        assert abs(below - above) < 2.0 * delta * abs(_dphi2(self.LIN)) + 1e-2

    def test_clip_applies_before_linearisation(self):
        """Below min_distance the energy is constant: clipping happens first."""
        mw = MonatomicWater(n_particles=2, box_length=20.0, beta=1.0,
                            min_distance=0.3, linearize_below=self.LIN)
        at_clip = float(-mw(_pair_config(0.3)))
        inside = float(-mw(_pair_config(0.1)))
        assert math.isclose(inside, at_clip, rel_tol=1e-6)
        expected = _phi2(self.LIN) + (0.3 - self.LIN) * _dphi2(self.LIN)
        assert math.isclose(at_clip, expected, rel_tol=1e-4)

    def test_gradient_finite_at_coincidence_training_config(self):
        """Training settings (min_distance=0.01, linearize_below=1.2)
        keep energy and gradient finite for exactly coincident particles, the
        precondition for reverse-KL training."""
        x = _lattice_config()
        x = x.at[1].set(x[0])
        mw = MonatomicWater(n_particles=8, box_length=10.0, beta=0.5,
                            min_distance=0.01, linearize_below=self.LIN)
        assert bool(jnp.isfinite(mw(x)))
        assert bool(jnp.all(jnp.isfinite(jax.grad(mw)(x))))

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_nonpositive_linearize_below_raises(self, bad):
        with pytest.raises(ValueError, match="linearize_below"):
            MonatomicWater(n_particles=2, box_length=20.0, linearize_below=bad)


@pytest.mark.parametrize("mind,lin", [
    (0.0, None), (0.1, None), (0.8, None),
    (0.0, 1.2), (0.01, 1.2), (0.01, 2.0), (0.3, 1.2), (0.8, 1.2),
])
class TestScalarReference:
    """Independent sums protect pair/triplet counting and softening semantics."""

    @staticmethod
    def points():
        # The first pair spans the periodic seam, 0.6 Angstrom apart: below
        # every linearization point and the active 0.8-Angstrom clip. The last
        # particle is outside all cutoffs; the others form nonzero triplets.
        return np.array([[0.2, 0.3, 0.4], [11.6, 0.3, 0.4],
                         [2.3, 1.5, 0.6], [2.8, 3.1, 1.0],
                         [7.0, 7.5, 8.0]], dtype=np.float32)

    def test_batched_energy_matches_scalar_sum(self, mind, lin):
        points = self.points()
        shifted = points.copy()
        shifted[2] += [0.2, -0.1, 0.3]
        batch = np.stack([points, shifted])
        dist = MonatomicWater(n_particles=5, box_length=12.0, beta=0.7,
                              min_distance=mind, linearize_below=lin)
        expected = [-0.7 * _reference_energy(x, 12.0, mind, lin) for x in batch]

        # Float64 reference vs float32 JAX, including the steep repulsive core.
        np.testing.assert_allclose(jax.jit(dist.__call__)(jnp.asarray(batch)),
                                   expected, rtol=2e-5, atol=2e-4)

    def test_gradient_matches_reference_finite_differences(self, mind, lin):
        x = self.points().astype(np.float64)
        dist = MonatomicWater(n_particles=5, box_length=12.0, beta=0.7,
                              min_distance=mind, linearize_below=lin)
        expected = np.zeros_like(x)
        # Perturb the float64 oracle, away from cutoff and clipping boundaries;
        # do not finite-difference float32 energies with a tiny step.
        for index in np.ndindex(x.shape):
            delta = np.zeros_like(x)
            delta[index] = 1e-4
            expected[index] = -0.7 * (
                _reference_energy(x + delta, 12.0, mind, lin)
                - _reference_energy(x - delta, 12.0, mind, lin)
            ) / 2e-4
        actual = jax.grad(dist)(jnp.asarray(x, dtype=jnp.float32))
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-3)


def test_isolated_right_angle_has_known_three_body_energy():
    """Only the central angle interacts: catch dropped or doubled triplets.

    The two neighbours are sqrt(2)*r apart, beyond the cutoff. With a right
    angle the angular factor is cos(theta_0)**2, giving an analytic reference
    for both the scalar oracle and the full production energy.
    """
    r = 3.5
    x = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0], [0.0, r, 0.0]])
    angular = (23.15 * _EPS * math.cos(math.radians(109.47))**2
               * math.exp(2 * 1.2 / (r / _SIGMA - _CUT)))
    expected = 2 * _phi2(r) + angular
    assert angular > 0.01  # The fixture must detect a missing three-body term.
    assert _reference_energy(x, 20.0) == pytest.approx(expected, rel=1e-12)
    dist = MonatomicWater(n_particles=3, box_length=20.0)
    assert float(-dist(jnp.asarray(x))) == pytest.approx(expected, rel=1e-5)
