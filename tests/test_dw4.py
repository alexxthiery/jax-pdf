"""Tests for DW4 distribution."""

import jax
import jax.numpy as jnp
import pytest

from jax_pdf import DW4


class TestDW4:
    def test_dim(self):
        assert DW4().dim == 8

    def test_beta_validation(self):
        with pytest.raises(ValueError, match="beta must be positive"):
            DW4(beta=-1.0)

    def test_output_shape_single(self):
        dist = DW4()
        x = jnp.ones(8)
        assert dist(x).shape == ()

    def test_output_shape_batch(self):
        dist = DW4()
        x = jnp.ones((5, 8))
        assert dist(x).shape == (5,)

    def test_permutation_invariance(self):
        """Swapping two particles does not change the energy."""
        dist = DW4()
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (8,))
        # Swap particles 0 and 2 (coords 0:2 <-> 4:6)
        x_swapped = x.at[0:2].set(x[4:6]).at[4:6].set(x[0:2])
        assert jnp.allclose(dist(x), dist(x_swapped), atol=1e-6)

    def test_translation_invariance_without_trap(self):
        """Without trap, shifting all particles preserves energy."""
        dist = DW4(trap_scale=0.0)
        key = jax.random.PRNGKey(1)
        x = jax.random.normal(key, (8,))
        shift = jnp.array([3.0, -2.0])
        x_shifted = x + jnp.tile(shift, 4)
        assert jnp.allclose(dist(x), dist(x_shifted), atol=1e-6)

    def test_trap_breaks_translation_invariance(self):
        """With trap, shifting all particles changes energy."""
        dist = DW4(trap_scale=1.0)
        key = jax.random.PRNGKey(1)
        x = jax.random.normal(key, (8,))
        shift = jnp.array([3.0, -2.0])
        x_shifted = x + jnp.tile(shift, 4)
        assert not jnp.allclose(dist(x), dist(x_shifted), atol=1e-3)

    def test_trap_penalizes_distance_from_origin(self):
        """Trap increases energy for particles far from the origin."""
        x_near = jnp.array([0.1, 0.0, -0.1, 0.0, 0.0, 0.1, 0.0, -0.1])
        x_far = x_near + 10.0
        dist = DW4(trap_scale=1.0)
        # Far from origin should have lower log-density
        assert dist(x_near) > dist(x_far)

    def test_trap_zero_at_origin(self):
        """Trap contributes nothing when all particles are at the origin."""
        x = jnp.zeros(8)
        dist = DW4(trap_scale=1.0)
        dist_no_trap = DW4(trap_scale=0.0)
        assert jnp.allclose(dist(x), dist_no_trap(x), atol=1e-6)

    def test_known_pair_energy(self):
        """Two particles at known distance, other two coincident at origin."""
        dist = DW4(c=0.0, trap_scale=0.0)
        d = 5.0
        x = jnp.array([0.0, 0.0, d, 0.0, 0.0, 0.0, 0.0, 0.0])
        log_p = dist(x)
        # Distances: (0,1)=5, (0,2)=0~eps, (0,3)=0~eps, (1,2)=5, (1,3)=5, (2,3)=0~eps
        r_01 = 5.0
        r_eps = jnp.sqrt(1e-8)
        v = lambda r: 0.9 * (r - 4.0) ** 4 + (-4.0) * (r - 4.0) ** 2
        expected_energy = 3 * v(r_01) + 3 * v(r_eps)
        assert jnp.allclose(log_p, -expected_energy, atol=1e-3)

    def test_beta_scaling(self):
        """log p at beta=2 should be 2x log p at beta=1."""
        key = jax.random.PRNGKey(2)
        x = jax.random.normal(key, (8,))
        lp1 = DW4(beta=1.0)(x)
        lp2 = DW4(beta=2.0)(x)
        assert jnp.allclose(lp2, 2.0 * lp1, atol=1e-6)

    def test_gradient_finite(self):
        dist = DW4()
        key = jax.random.PRNGKey(3)
        x = jax.random.normal(key, (8,))
        grad = jax.grad(dist)(x)
        assert jnp.all(jnp.isfinite(grad))

    def test_log_normalization_raises(self):
        with pytest.raises(NotImplementedError):
            DW4().log_normalization()
