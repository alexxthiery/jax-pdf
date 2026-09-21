"""Periodic Lennard-Jones tests with analytic and scalar reference oracles."""

from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_pdf import PeriodicLennardJones


def _reference_energy(x, box_length, cutoff, epsilon, sigma, lambda_lj,
                      min_distance, linearize_below, shift_energy):
    """Float64 pair sum of the potential in docs/periodic_lennard_jones.md.

    Scalar distances, explicit pair enumeration and an analytic radial slope
    provide an oracle independent of the JAX distance matrix and autodiff.
    Clipping precedes linearization and truncation; the shift is the original
    soft-core potential at the cutoff. No production energy helper is used.
    """
    def potential(r):
        g = (r / sigma)**6 + 0.5 * (1.0 - lambda_lj)**2
        return 4 * lambda_lj * epsilon * (1 / g**2 - 1 / g)

    def slope(r):
        g = (r / sigma)**6 + 0.5 * (1.0 - lambda_lj)**2
        dg = 6 * r**5 / sigma**6
        return 4 * lambda_lj * epsilon * dg * (g - 2) / g**3

    x = np.asarray(x, dtype=np.float64)
    shift = potential(cutoff) if shift_energy else 0.0
    energy = 0.0
    for i, j in combinations(range(len(x)), 2):
        delta = x[j] - x[i]
        delta -= box_length * np.rint(delta / box_length)
        r = max(float(np.linalg.norm(delta)), min_distance)
        if r > cutoff:
            continue
        if linearize_below is not None and r < linearize_below:
            pair = potential(linearize_below) + (r - linearize_below) * slope(linearize_below)
        else:
            pair = potential(r)
        energy += pair - shift
    return energy


def _hard_lj(n_particles, box_length, **kw):
    """Standard (lambda=1), unshifted, un-clipped instance for clean checks."""
    return PeriodicLennardJones(
        n_particles=n_particles,
        spatial_dim=3,
        box_length=box_length,
        shift_energy=False,
        min_distance=0.0,
        **kw,
    )


class TestConstruction:
    def test_from_density_matches_cubic_box(self):
        dist = PeriodicLennardJones.from_density(256, density=1.28)
        # (256 / 1.28) ** (1/3) = 200 ** (1/3)
        assert jnp.allclose(dist.box_length, 200.0 ** (1.0 / 3.0))

    def test_dim(self):
        assert PeriodicLennardJones(n_particles=108).dim == 324

    def test_validation_n_particles(self):
        with pytest.raises(ValueError, match="n_particles must be >= 2"):
            PeriodicLennardJones(n_particles=1)

    def test_validation_box_length(self):
        with pytest.raises(ValueError, match="box_length must be positive"):
            PeriodicLennardJones(box_length=0.0)

    def test_validation_lambda(self):
        with pytest.raises(ValueError, match="lambda_lj must be in"):
            PeriodicLennardJones(lambda_lj=1.5)


class TestPairPotential:
    def test_zero_at_sigma(self):
        """Hard LJ pair energy is exactly 0 at r = sigma (large box, no cutoff effect)."""
        dist = _hard_lj(2, box_length=100.0, cutoff=50.0)
        x = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # sigma = 1
        assert jnp.allclose(dist(x), 0.0, atol=1e-6)

    def test_minimum_at_rmin(self):
        """Pair energy is -epsilon at r = 2^(1/6) sigma; log p = -beta*(-eps)."""
        dist = _hard_lj(2, box_length=100.0, cutoff=50.0, epsilon=1.0, beta=1.0)
        rmin = 2.0 ** (1.0 / 6.0)
        x = jnp.array([[0.0, 0.0, 0.0], [rmin, 0.0, 0.0]])
        assert jnp.allclose(dist(x), 1.0, atol=1e-6)


class TestPeriodicity:
    def test_minimum_image(self):
        """Particles near opposite faces interact across the boundary."""
        L = 6.0
        near = _hard_lj(2, box_length=L, cutoff=2.0)
        # True min-image separation is 1.2 (0.6 + 0.6 across the seam); chosen
        # near the potential well where the energy is O(1) and well-conditioned.
        x_wrapped = jnp.array([[0.6, 0.0, 0.0], [L - 0.6, 0.0, 0.0]])
        x_direct = jnp.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
        assert jnp.allclose(near(x_wrapped), near(x_direct), atol=1e-5)

    def test_image_shift_invariance(self):
        """Shifting one particle by a full box vector leaves the energy unchanged."""
        L = 7.0
        dist = _hard_lj(3, box_length=L, cutoff=3.0)
        key = jax.random.PRNGKey(0)
        x = jax.random.uniform(key, (3, 3), minval=0.0, maxval=L)
        x_img = x.at[1].add(jnp.array([L, 0.0, 0.0]))
        assert jnp.allclose(dist(x), dist(x_img), atol=1e-5)

    def test_translation_invariance(self):
        L = 7.0
        dist = _hard_lj(4, box_length=L, cutoff=3.0)
        key = jax.random.PRNGKey(1)
        x = jax.random.uniform(key, (4, 3), minval=0.0, maxval=L)
        assert jnp.allclose(dist(x), dist(x + jnp.array([2.3, -1.1, 0.7])), atol=1e-5)

    def test_permutation_invariance(self):
        L = 7.0
        dist = _hard_lj(4, box_length=L, cutoff=3.0)
        key = jax.random.PRNGKey(2)
        x = jax.random.uniform(key, (4, 3), minval=0.0, maxval=L)
        x_swap = x.at[0].set(x[3]).at[3].set(x[0])
        assert jnp.allclose(dist(x), dist(x_swap), atol=1e-6)


class TestCutoffAndShift:
    def test_no_interaction_beyond_cutoff(self):
        """A pair separated beyond the cutoff (but within L/2) contributes nothing."""
        L = 12.0
        dist = PeriodicLennardJones(
            n_particles=2, spatial_dim=3, box_length=L, cutoff=2.7, shift_energy=True
        )
        x = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])  # 3.0 > cutoff 2.7
        assert jnp.allclose(dist(x), 0.0, atol=1e-7)

    def test_shift_makes_potential_continuous_at_cutoff(self):
        """With shift, energy -> 0 as r -> cutoff from below."""
        L = 12.0
        dist = PeriodicLennardJones(
            n_particles=2, spatial_dim=3, box_length=L, cutoff=2.7, shift_energy=True
        )
        x_just_inside = jnp.array([[0.0, 0.0, 0.0], [2.699, 0.0, 0.0]])
        assert jnp.allclose(dist(x_just_inside), 0.0, atol=1e-4)


class TestNumerics:
    def test_beta_scaling(self):
        L = 7.0
        key = jax.random.PRNGKey(3)
        x = jax.random.uniform(key, (4, 3), minval=0.0, maxval=L)
        lp1 = _hard_lj(4, box_length=L, cutoff=3.0, beta=1.0)(x)
        lp2 = _hard_lj(4, box_length=L, cutoff=3.0, beta=2.0)(x)
        assert jnp.allclose(lp2, 2.0 * lp1, atol=1e-6)

    def test_soft_core_finite_at_overlap(self):
        """lambda_lj < 1 keeps the log-density finite when particles coincide."""
        dist = PeriodicLennardJones(
            n_particles=3, spatial_dim=3, box_length=8.0, lambda_lj=0.5
        )
        x = jnp.zeros((3, 3))  # all coincident
        assert jnp.isfinite(dist(x))

    def test_log_normalization_raises(self):
        with pytest.raises(NotImplementedError):
            PeriodicLennardJones().log_normalization()


@pytest.mark.parametrize("lam,shift,mind,lin", [
    (1.0, True, 0.0, None), (1.0, False, 0.0, None),
    (0.5, True, 0.0, None), (1.0, True, 0.1, None),
    (1.0, True, 0.0, 0.8), (1.0, True, 0.7, 0.8),
    (0.5, False, 0.7, 0.8),
])
class TestScalarReference:
    """Pair counting, periodicity and softening checked without external code."""

    @staticmethod
    def points():
        # Pair 0-1 has minimum-image distance 0.6, exercising both the active
        # 0.7 clip and the 0.8 linearization. Other pairs straddle the cutoff.
        return np.array([[0.2, 0.3, 0.4], [3.6, 0.3, 0.4],
                         [1.4, 1.2, 0.8], [2.2, 2.6, 2.9]], dtype=np.float32)

    @staticmethod
    def parameters(lam, shift, mind, lin):
        return dict(box_length=4.0, cutoff=1.8, epsilon=1.3, sigma=0.9,
                    lambda_lj=lam, shift_energy=shift, min_distance=mind,
                    linearize_below=lin)

    def test_batched_energy_matches_scalar_sum(self, lam, shift, mind, lin):
        parameters = self.parameters(lam, shift, mind, lin)
        x = self.points()
        perturbed = x.copy()
        perturbed[2] += [0.2, -0.1, 0.3]
        batch = np.stack([x, perturbed])
        dist = PeriodicLennardJones(n_particles=4, beta=0.7, **parameters)
        expected = [-0.7 * _reference_energy(y, **parameters) for y in batch]

        np.testing.assert_allclose(jax.jit(dist.__call__)(jnp.asarray(batch)),
                                   expected, rtol=2e-5, atol=2e-4)

    def test_gradient_matches_reference_finite_differences(self, lam, shift, mind, lin):
        parameters = self.parameters(lam, shift, mind, lin)
        x = self.points().astype(np.float64)
        dist = PeriodicLennardJones(n_particles=4, beta=0.7, **parameters)
        expected = np.zeros_like(x)
        # Float64 differences away from switching surfaces, compared with
        # float32 autodiff; the absolute tolerance covers near-zero forces.
        for index in np.ndindex(x.shape):
            delta = np.zeros_like(x)
            delta[index] = 1e-4
            expected[index] = -0.7 * (
                _reference_energy(x + delta, **parameters)
                - _reference_energy(x - delta, **parameters)
            ) / 2e-4
        actual = jax.grad(dist)(jnp.asarray(x, dtype=jnp.float32))
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-3)


def test_scalar_reference_has_known_lj_minimum():
    """Anchor the reference itself: the unshifted hard-LJ minimum is -eps."""
    sigma, epsilon = 0.9, 1.3
    x = [[0.0, 0.0, 0.0], [2**(1/6) * sigma, 0.0, 0.0]]
    energy = _reference_energy(x, box_length=10.0, cutoff=4.0,
                               epsilon=epsilon, sigma=sigma, lambda_lj=1.0,
                               min_distance=0.0, linearize_below=None,
                               shift_energy=False)
    assert energy == pytest.approx(-epsilon, rel=1e-12)
