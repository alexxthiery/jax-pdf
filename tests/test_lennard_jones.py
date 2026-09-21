"""Tests for LennardJones distribution."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_pdf import LennardJones


class TestLennardJones:
    @pytest.mark.parametrize("relative_distance", [1.0, 1.25, 2.0])
    def test_nondefault_length_scale_value_and_force(self, relative_distance):
        """Default rm=1 cannot detect an implementation that ignores rm.

        A single pair isolates the 12-6 potential and its analytic derivative.
        The tiny squared-distance regularizer changes these values below the
        absolute tolerance at the chosen nonzero distances.
        """
        rm, epsilon, beta = 1.4, 1.8, 0.7
        r = rm * relative_distance
        dist = LennardJones(n_particles=2, rm=rm, epsilon=epsilon, beta=beta,
                            trap_scale=0.0)
        x = jnp.array([[0., 0., 0.], [r, 0., 0.]])
        q6 = relative_distance**-6
        expected = -beta * epsilon * (q6**2 - 2 * q6)
        slope = 12 * epsilon / r * (q6 - q6**2)
        value, gradient = jax.jit(jax.value_and_grad(dist))(x)
        assert float(value) == pytest.approx(expected, rel=2e-5, abs=2e-6)
        np.testing.assert_allclose(gradient, [[beta * slope, 0, 0], [-beta * slope, 0, 0]],
                                   rtol=2e-5, atol=3e-6)

    @pytest.mark.parametrize("mode,trap_energy", [("individual", 5.0), ("com", 4.0)])
    def test_trap_modes_have_known_energy(self, mode, trap_energy):
        """Two points at (+/-1,2) distinguish individual and COM confinement."""
        x = jnp.array([[1., 2.], [-1., 2.]])
        free = LennardJones(n_particles=2, spatial_dim=2, trap_scale=0.0, beta=0.6)
        trapped = free.replace(trap_scale=0.7, trap_mode=mode)
        assert float(trapped(x) - free(x)) == pytest.approx(-0.6 * 0.7 * trap_energy,
                                                           abs=2e-6)

    def test_dim_lj13(self):
        assert LennardJones().dim == 39

    def test_dim_lj55(self):
        assert LennardJones(n_particles=55).dim == 165

    def test_dim_custom(self):
        assert LennardJones(n_particles=7, spatial_dim=2).dim == 14

    def test_validation_n_particles(self):
        with pytest.raises(ValueError, match="n_particles must be >= 2"):
            LennardJones(n_particles=1)

    def test_validation_spatial_dim(self):
        with pytest.raises(ValueError, match="spatial_dim must be >= 1"):
            LennardJones(spatial_dim=0)

    def test_validation_epsilon(self):
        with pytest.raises(ValueError, match="epsilon must be positive"):
            LennardJones(epsilon=-1.0)

    def test_validation_rm(self):
        with pytest.raises(ValueError, match="rm must be positive"):
            LennardJones(rm=0.0)

    def test_validation_beta(self):
        with pytest.raises(ValueError, match="beta must be positive"):
            LennardJones(beta=-1.0)

    def test_lj_minimum_at_rm(self):
        """Two particles at distance rm: LJ pair energy = -epsilon."""
        dist = LennardJones(
            n_particles=2, spatial_dim=3, trap_scale=0.0
        )
        x = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        log_p = dist(x)
        # U_LJ = eps * ((rm/rm)^12 - 2*(rm/rm)^6) = eps*(1 - 2) = -eps
        # log p = -beta * (-eps) = eps
        assert jnp.allclose(log_p, 1.0, atol=1e-4)

    def test_permutation_invariance(self):
        """Swapping particles does not change energy."""
        dist = LennardJones(n_particles=4, spatial_dim=3)
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (4, 3))
        # Swap particles 0 and 2.
        x_swapped = x.at[0].set(x[2]).at[2].set(x[0])
        assert jnp.allclose(dist(x), dist(x_swapped), atol=1e-6)

    def test_translation_invariance_without_trap(self):
        """Without trap, shifting all particles preserves energy."""
        dist = LennardJones(
            n_particles=3, spatial_dim=2, trap_scale=0.0
        )
        key = jax.random.PRNGKey(4)
        x = jax.random.normal(key, (3, 2))
        shift = jnp.array([5.0, -3.0])
        x_shifted = x + shift  # broadcasts across particle axis
        assert jnp.allclose(dist(x), dist(x_shifted), atol=1e-5)

    def test_trap_breaks_translation_invariance(self):
        """With trap, shifting all particles changes energy."""
        dist = LennardJones(
            n_particles=3, spatial_dim=2, trap_scale=1.0
        )
        key = jax.random.PRNGKey(4)
        x = jax.random.normal(key, (3, 2))
        shift = jnp.array([5.0, -3.0])
        x_shifted = x + shift
        assert not jnp.allclose(dist(x), dist(x_shifted), atol=1e-3)

    def test_trap_zero_at_origin(self):
        """Trap contributes nothing when all particles are at origin."""
        x = jnp.zeros((3, 3))
        dist = LennardJones(n_particles=3, trap_scale=1.0)
        dist_no_trap = LennardJones(n_particles=3, trap_scale=0.0)
        assert jnp.allclose(dist(x), dist_no_trap(x), atol=1e-6)

    def test_trap_penalizes_distance_from_origin(self):
        """Trap increases energy for particles far from origin."""
        dist = LennardJones(n_particles=3, spatial_dim=2, trap_scale=1.0)
        # Deterministic positions near the LJ pair minimum so u_lj stays O(1)
        # and the trap difference isn't lost to float32 cancellation.
        x_near = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])
        x_far = x_near + 10.0
        # Pair distances are the same, but trap penalizes x_far more
        dist_no_trap = LennardJones(
            n_particles=3, spatial_dim=2, trap_scale=0.0
        )
        trap_near = dist_no_trap(x_near) - dist(x_near)
        trap_far = dist_no_trap(x_far) - dist(x_far)
        assert trap_far > trap_near

    def test_beta_scaling(self):
        """log p at beta=2 should be 2x log p at beta=1."""
        key = jax.random.PRNGKey(1)
        x = jax.random.normal(key, (3, 2))
        lp1 = LennardJones(
            n_particles=3, spatial_dim=2, beta=1.0
        )(x)
        lp2 = LennardJones(
            n_particles=3, spatial_dim=2, beta=2.0
        )(x)
        assert jnp.allclose(lp2, 2.0 * lp1, atol=1e-6)

    def test_gradient_finite_lj55(self):
        dist = LennardJones(n_particles=55)
        key = jax.random.PRNGKey(3)
        x = jax.random.normal(key, (55, 3))
        grad = jax.grad(dist)(x)
        assert grad.shape == (55, 3)
        assert jnp.all(jnp.isfinite(grad))

    def test_log_normalization_raises(self):
        with pytest.raises(NotImplementedError):
            LennardJones().log_normalization()
