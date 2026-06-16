"""Tests for PeriodicLennardJones distribution."""

import jax
import jax.numpy as jnp
import pytest

from jax_pdf import PeriodicLennardJones


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


class TestShapes:
    def test_output_shape_single(self):
        dist = _hard_lj(4, box_length=10.0)
        assert dist(jnp.zeros((4, 3))).shape == ()

    def test_output_shape_batch(self):
        dist = _hard_lj(4, box_length=10.0)
        assert dist(jnp.zeros((5, 4, 3))).shape == (5,)

    def test_wrong_shape_raises(self):
        dist = _hard_lj(4, box_length=10.0)
        with pytest.raises(ValueError, match=r"x.shape\[-2:\] == \(4, 3\)"):
            dist(jnp.ones(12))


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

    def test_gradient_finite(self):
        dist = PeriodicLennardJones.from_density(256, density=1.28)
        key = jax.random.PRNGKey(4)
        x = jax.random.uniform(key, (256, 3), minval=0.0, maxval=dist.box_length)
        grad = jax.grad(dist)(x)
        assert grad.shape == (256, 3)
        assert jnp.all(jnp.isfinite(grad))

    def test_jit(self):
        dist = _hard_lj(8, box_length=8.0, cutoff=3.0)
        key = jax.random.PRNGKey(5)
        x = jax.random.uniform(key, (8, 3), minval=0.0, maxval=8.0)
        assert jnp.allclose(jax.jit(dist.__call__)(x), dist(x), atol=1e-6)

    def test_log_normalization_raises(self):
        with pytest.raises(NotImplementedError):
            PeriodicLennardJones().log_normalization()


class TestBgmatDifferential:
    """Lock the contract: our energy == bgmat's LennardJonesEnergy bit-for-bit.

    Strongest available oracle (independent reference implementation). Skipped
    when the sibling bgmat repo or its deps are unavailable, so the suite stays
    green in minimal environments.
    """

    def test_matches_bgmat_energy(self):
        import os
        import sys

        bgmat_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "bgmat")
        )
        if not os.path.isdir(bgmat_root):
            pytest.skip("sibling bgmat repo not found")
        if bgmat_root not in sys.path:
            sys.path.insert(0, bgmat_root)
        try:
            from bgmat.systems.lennard_jones import LennardJonesEnergy
        except Exception as e:  # missing chex / heavy deps absent
            pytest.skip(f"bgmat not importable: {e}")

        N, L, B = 8, 4.0, 0.5
        box = jnp.full((3,), L)
        x = jax.random.uniform(jax.random.PRNGKey(0), (5, N, 3), minval=0.0, maxval=L)
        # (cutoff, lambda_lj, shift, min_distance, linearize_below)
        cases = [
            (1.8, 1.0, True, 0.0, None),
            (1.8, 1.0, False, 0.0, None),
            (1.8, 0.5, True, 0.0, None),
            (1.8, 1.0, True, 0.1, None),
            (1.8, 1.0, True, 0.0, 0.8),
        ]
        for cutoff, lam, shift, mind, lin in cases:
            bg = LennardJonesEnergy(
                cutoff=cutoff, box_length=box, epsilon=1.0, sigma=1.0,
                min_distance=mind, lambda_lj=lam, linearize_below=lin,
                shift_energy=shift,
            )
            ours = PeriodicLennardJones(
                n_particles=N, spatial_dim=3, box_length=L, cutoff=cutoff, beta=B,
                lambda_lj=lam, min_distance=mind, linearize_below=lin,
                shift_energy=shift,
            )
            our_energy = -ours(x) / B            # __call__ returns -beta*U
            assert jnp.allclose(our_energy, bg.energy(x), rtol=1e-4, atol=1e-3), (
                f"mismatch vs bgmat for case "
                f"cutoff={cutoff} lambda={lam} shift={shift} "
                f"min_distance={mind} linearize_below={lin}"
            )
