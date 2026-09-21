"""Tests for LatticePhiFour.

Oracles are independent of the implementation: closed forms for fields whose
energy can be written down by hand, the analytic gradient of the action, the
symmetries of the action, and `PhiFour` as a second implementation of the same
model in one dimension.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_pdf import LatticePhiFour, PhiFour


def action(dist, x):
    """The action S(x) = -log p(x), as the tests speak of it."""
    return -dist(x)


class TestClosedForms:

    @pytest.mark.parametrize("shape", [(8,), (4, 5), (2, 3, 4)])
    @pytest.mark.parametrize("m", [0.0, 1.0, -0.6])
    def test_constant_field(self, shape, m):
        """
        Claim: a constant field has no coupling energy, so only the local terms
        remain: S = V u (m^2 - a^2)^2 - h V m over V sites.
        Bugs: a coupling term that does not vanish on a constant field (a
        misplaced roll), a volume factor counted per axis, a dropped field term.
        Oracle: the closed form written out.
        """
        u, a, kappa, h = 0.7, 1.3, 2.1, 0.4
        dist = LatticePhiFour(u=u, a=a, kappa=kappa, h=h, lattice_shape=shape)
        volume = int(np.prod(shape))

        got = action(dist, jnp.full((dist.dim,), m))
        expected = volume * u * (m**2 - a**2) ** 2 - h * volume * m

        assert float(got) == pytest.approx(expected, rel=1e-5, abs=1e-6)

    def test_one_excited_site_counts_all_its_bonds(self):
        """
        Claim: on a 2D lattice each site meets 4 bonds, so lifting one site to t
        from a zero field costs (kappa / 2) * 4 * t^2 in the coupling.
        Bugs: bonds counted along one axis only, or counted twice.
        Oracle: the energy written out term by term.
        """
        shape, t = (4, 5), 0.8
        u, a, kappa = 0.5, 1.0, 3.0
        dist = LatticePhiFour(u=u, a=a, kappa=kappa, h=0.0, lattice_shape=shape)
        volume = int(np.prod(shape))

        got = action(dist, jnp.zeros(dist.dim).at[0].set(t))
        quartic = u * ((t**2 - a**2) ** 2 + (volume - 1) * a**4)
        coupling = kappa / 2 * (2 * len(shape)) * t**2

        assert float(got) == pytest.approx(quartic + coupling, rel=1e-5)

    def test_a_side_of_two_carries_two_bonds(self):
        """
        Claim: on a periodic ring of length 2 the two sites are joined twice,
        once each way around, as in `PhiFour`.
        Bug: silently counting one bond, which halves the coupling on the
        smallest lattice.
        Oracle: the energy of a two-site field written out.
        """
        u, a, kappa, t = 1.0, 1.0, 2.0, 0.5
        dist = LatticePhiFour(u=u, a=a, kappa=kappa, h=0.0, lattice_shape=(2,))

        got = action(dist, jnp.array([t, -t]))
        quartic = 2 * u * (t**2 - a**2) ** 2
        coupling = kappa / 2 * 2 * (2 * t) ** 2          # two bonds, each (t - (-t))^2

        assert float(got) == pytest.approx(quartic + coupling, rel=1e-5)


class TestBarrier:

    @pytest.mark.parametrize("shape", [(8,), (4, 4), (6, 6)])
    def test_uniform_barrier_scales_with_the_number_of_sites(self, shape):
        """
        Claim (docs/lattice_phi_four.md): at h = 0 the barrier along the uniform
        field is V u a^4, so it grows in proportion to the lattice, unlike
        `PhiFour` whose normalization keeps it fixed.
        Bug: a volume factor that cancels, which would silently make every
        lattice equally hard.
        Oracle: the closed form.
        """
        u, a = 0.8, 1.1
        dist = LatticePhiFour(u=u, a=a, kappa=1.5, h=0.0, lattice_shape=shape)
        volume = int(np.prod(shape))

        barrier = action(dist, jnp.zeros(dist.dim)) - action(dist, jnp.full((dist.dim,), a))

        assert float(barrier) == pytest.approx(volume * u * a**4, rel=1e-5)


class TestSymmetries:

    def test_sign_symmetry_holds_at_zero_field(self):
        """
        Claim: at h = 0 the action is invariant under x -> -x, which is what
        makes the two phases carry equal mass.
        Bug: an odd term left in the local potential.
        Oracle: the action at the mirrored field.
        """
        dist = LatticePhiFour(u=0.6, a=1.2, kappa=1.7, h=0.0, lattice_shape=(3, 4))
        x = jnp.asarray(np.random.default_rng(0).standard_normal(dist.dim), jnp.float32)

        assert float(dist(x)) == pytest.approx(float(dist(-x)), rel=1e-6)

    def test_a_nonzero_field_breaks_the_sign_symmetry(self):
        """Fixture strength: with h != 0 the mirrored field has another energy."""
        dist = LatticePhiFour(u=0.6, a=1.2, kappa=1.7, h=0.5, lattice_shape=(3, 4))
        x = jnp.asarray(np.random.default_rng(1).standard_normal(dist.dim), jnp.float32)

        assert abs(float(dist(x)) - float(dist(-x))) > 1e-3

    @pytest.mark.parametrize("axis", [0, 1])
    def test_translation_invariance(self, axis):
        """
        Claim: the lattice is periodic, so shifting the field along any axis
        leaves the action unchanged.
        Bug: a boundary term, or an axis handled differently from the others.
        Oracle: the action of the rolled field.
        """
        shape = (3, 4)
        dist = LatticePhiFour(u=0.6, a=1.2, kappa=1.7, h=0.3, lattice_shape=shape)
        x = np.random.default_rng(2).standard_normal(shape).astype(np.float32)
        shifted = np.roll(x, 1, axis=axis)

        assert float(dist(jnp.asarray(x.ravel()))) == pytest.approx(
            float(dist(jnp.asarray(shifted.ravel()))), rel=1e-6
        )


class TestParameterRedundancy:

    @pytest.mark.parametrize("u,a,kappa,h,shape", [
        (0.7, 1.3, 2.1, 0.4, (4, 5)),
        (2.0, 0.6, 0.3, -1.1, (6,)),
        (0.2, 2.5, 1.0, 0.0, (3, 3, 3)),
    ])
    def test_the_well_location_is_a_choice_of_field_units(self, u, a, kappa, h, shape):
        """
        Claim: x = a y turns S(x; u, a, kappa, h) into S(y; u a^4, 1, kappa a^2, h a),
        so the four parameters are three plus a scale, and a study may fix a = 1.
        Bug: a term whose power of the field does not match its parameter, which
        would break the scaling.
        Oracle: the rescaled distribution evaluated at the rescaled field.
        """
        full = LatticePhiFour(u=u, a=a, kappa=kappa, h=h, lattice_shape=shape)
        unit = LatticePhiFour(u=u * a**4, a=1.0, kappa=kappa * a**2, h=h * a,
                              lattice_shape=shape)
        x = np.random.default_rng(5).standard_normal((6, full.dim)).astype(np.float32) * a

        np.testing.assert_allclose(np.asarray(full(jnp.asarray(x))),
                                   np.asarray(unit(jnp.asarray(x / a))), rtol=1e-5)


class TestGradient:

    def test_gradient_matches_the_analytic_expression(self):
        """
        Claim: d log p / d x_v = -(4 u x_v (x_v^2 - a^2) + kappa sum_w (x_v - x_w) - h).
        Bug: a factor of two from the bond convention, or a sign.
        Oracle: the discrete Laplacian written with rolls in both directions,
        which does not share the implementation's bond formulation.
        """
        shape = (4, 5)
        u, a, kappa, h = 0.9, 1.1, 1.4, 0.2
        dist = LatticePhiFour(u=u, a=a, kappa=kappa, h=h, lattice_shape=shape)
        x = np.random.default_rng(3).standard_normal(shape).astype(np.float32)

        got = jax.grad(dist)(jnp.asarray(x.ravel()))
        laplacian = sum(2 * x - np.roll(x, 1, ax) - np.roll(x, -1, ax)
                        for ax in range(len(shape)))
        expected = -(4 * u * x * (x**2 - a**2) + kappa * laplacian - h)

        np.testing.assert_allclose(np.asarray(got), expected.ravel(), rtol=1e-4, atol=1e-5)


class TestAgreementWithPhiFour:

    @pytest.mark.parametrize("n,a_jp,b,beta", [(12, 0.1, 0.0, 1.0), (16, 0.05, 0.3, 1.5)])
    def test_matches_phi_four_on_a_ring(self, n, a_jp, b, beta):
        """
        Claim: in one dimension this is `PhiFour` with kappa = beta c,
        u = beta / (4c), a = 1, h = -beta b / c, where c = a_jp * n.
        Bug: a factor in any term, or a different bond convention.
        Oracle: `PhiFour` itself, a separate implementation.
        """
        c = a_jp * n
        chain = LatticePhiFour(u=beta / (4 * c), a=1.0, kappa=beta * c,
                               h=-beta * b / c, lattice_shape=(n,))
        reference = PhiFour(a=a_jp, b=b, dim_grid=n, beta=beta, periodic=True)
        x = jnp.asarray(np.random.default_rng(4).standard_normal((5, n)), jnp.float32)

        np.testing.assert_allclose(np.asarray(chain(x)), np.asarray(reference(x)), rtol=1e-5)


class TestInterface:

    @pytest.mark.parametrize("shape,dim", [((8,), 8), ((4, 5), 20), ((2, 3, 4), 24)])
    def test_dim_is_the_number_of_sites(self, shape, dim):
        assert LatticePhiFour(lattice_shape=shape).dim == dim

    def test_log_normalization_raises(self):
        with pytest.raises(NotImplementedError):
            LatticePhiFour().log_normalization()

    @pytest.mark.parametrize("kwargs,message", [
        ({"u": 0.0}, "u must be positive"),
        ({"a": -1.0}, "a must be positive"),
        ({"kappa": 0.0}, "kappa must be positive"),
        ({"lattice_shape": ()}, "at least one axis"),
        ({"lattice_shape": (4, 1)}, "every lattice side must be >= 2"),
    ])
    def test_validation(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            LatticePhiFour(**kwargs)
