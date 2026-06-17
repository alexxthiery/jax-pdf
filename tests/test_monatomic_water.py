"""Tests for MonatomicWater (Stillinger-Weber mW), incl. bgmat differential."""
import jax
import jax.numpy as jnp
import pytest

from jax_pdf import MonatomicWater


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


class TestBgmatDifferential:
    """Lock the 2-body + 3-body form against bgmat's MonatomicWaterEnergy."""

    def test_matches_bgmat_energy(self):
        import os
        import sys

        root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "bgmat")
        )
        if not os.path.isdir(root):
            pytest.skip("sibling bgmat repo not found")
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from bgmat.systems.monatomic_water import MonatomicWaterEnergy
        except Exception as e:
            pytest.skip(f"bgmat not importable: {e}")

        N, L, beta = 8, 6.0, 0.5
        box = jnp.full((3,), L)
        x = jax.random.uniform(jax.random.PRNGKey(1), (4, N, 3), minval=0.0, maxval=L)
        for mind in (0.0, 0.1):
            bg = MonatomicWaterEnergy(box_length=box, min_distance=mind)
            ours = MonatomicWater(n_particles=N, box_length=L, beta=beta, min_distance=mind)
            our_energy = -ours(x) / beta            # __call__ returns -beta*U
            assert jnp.allclose(
                our_energy, bg.energy(x), rtol=1e-3, atol=1e-2
            ), f"mismatch vs bgmat, min_distance={mind}"
