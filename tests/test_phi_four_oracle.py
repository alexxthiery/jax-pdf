"""Tests for PhiFourChainOracle.

Oracles are independent of the transfer operator: dense quadrature over every
configuration of a tiny chain, the closed form of the Gaussian chain that
u = 0 leaves behind, the sign symmetry of the action, and exact enumeration of
the sampler's own law. The quadrature here is written in NumPy from the action
itself, so it shares no code with the implementation under test.
"""

import numpy as np
import pytest

from jax_pdf import LatticePhiFour, PhiFour
from jax_pdf import phi_four_oracle as oracle_module
from jax_pdf.phi_four_oracle import PhiFourChainOracle


def trapezoid_grid(n_grid, bound):
    grid = np.linspace(-bound, bound, n_grid)
    weights = np.full(n_grid, grid[1] - grid[0])
    weights[0] = weights[-1] = (grid[1] - grid[0]) / 2
    return grid, weights


def chain_action(x, u, a, kappa, h, periodic):
    """S(x) for a chain, written from the definition in NumPy."""
    quartic = u * np.sum((x**2 - a**2) ** 2, axis=-1)
    if periodic:
        bonds = np.sum((x - np.roll(x, -1, axis=-1)) ** 2, axis=-1)
    else:
        zero = np.zeros_like(x[..., :1])
        bonds = np.sum(np.diff(np.concatenate([zero, x, zero], axis=-1), axis=-1) ** 2, axis=-1)
    return quartic + kappa / 2 * bonds - h * np.sum(x, axis=-1)


def dense_configurations(n_sites, grid, weights):
    """Every configuration of a tiny chain, with its quadrature weight."""
    mesh = np.meshgrid(*([grid] * n_sites), indexing="ij")
    x = np.stack([m.ravel() for m in mesh], axis=-1)
    wmesh = np.meshgrid(*([weights] * n_sites), indexing="ij")
    w = np.prod(np.stack([m.ravel() for m in wmesh], axis=-1), axis=-1)
    return x, w


def dense_log_partition(n_sites, params, periodic, grid, weights):
    x, w = dense_configurations(n_sites, grid, weights)
    log_weight = np.log(w) - chain_action(x, *params, periodic)
    top = log_weight.max()
    return top + np.log(np.sum(np.exp(log_weight - top)))


def gaussian_chain(n_sites, kappa, h):
    """The u = 0 chain with Dirichlet ends: precision, mean, covariance, log Z."""
    precision = kappa * (2 * np.eye(n_sites)
                         - np.eye(n_sites, k=1) - np.eye(n_sites, k=-1))
    covariance = np.linalg.inv(precision)
    mean = covariance @ (h * np.ones(n_sites))
    sign, logdet = np.linalg.slogdet(precision)
    log_z = 0.5 * n_sites * np.log(2 * np.pi) - 0.5 * logdet + 0.5 * h * np.sum(mean)
    return precision, mean, covariance, log_z


class TestLogPartition:

    @pytest.mark.parametrize("n_sites", [2, 3])
    @pytest.mark.parametrize("periodic", [True, False])
    def test_matches_dense_quadrature(self, n_sites, periodic):
        """
        Claim: the transfer operator computes the same integral as summing over
        every configuration, on the same grid and quadrature rule.
        Bugs: a local factor applied once too often or too few times, a bond
        missed at the seam of a ring, a boundary bond dropped.
        Oracle: the dense sum, written from the action.
        """
        params = (0.8, 1.0, 1.5, 0.3)
        n_grid, bound = 121, 3.0
        grid, weights = trapezoid_grid(n_grid, bound)
        oracle = PhiFourChainOracle(*params, n_sites=n_sites, periodic=periodic,
                                    n_grid=n_grid, bound=bound)

        expected = dense_log_partition(n_sites, params, periodic, grid, weights)

        assert oracle.log_partition() == pytest.approx(expected, rel=1e-10)

    def test_gaussian_chain_closed_form(self):
        """
        Claim: with u = 0 the chain is Gaussian, and the operator reproduces its
        closed-form log Z, including the external field's contribution.
        Bug: a missing 2 pi, a determinant of the wrong matrix, a field term
        that does not enter the normalization.
        Oracle: (N/2) log(2 pi) - (1/2) log det P + (1/2) h' P^-1 h.
        """
        n_sites, kappa, h = 4, 2.0, 0.3
        _, _, _, expected = gaussian_chain(n_sites, kappa, h)
        oracle = PhiFourChainOracle(u=0.0, a=1.0, kappa=kappa, h=h, n_sites=n_sites,
                                    periodic=False, n_grid=2001, bound=8.0)

        assert oracle.log_partition() == pytest.approx(expected, rel=1e-6)


class TestObservables:

    def test_site_marginal_matches_dense_quadrature(self):
        """
        Claim: the marginal of a site is the dense sum over the other sites.
        Bug: forward and backward messages combined with a factor counted twice
        at the site itself.
        Oracle: the dense configuration table, marginalized.
        """
        params, n_sites = (0.8, 1.0, 1.5, 0.3), 3
        n_grid, bound = 61, 3.0
        grid, weights = trapezoid_grid(n_grid, bound)
        oracle = PhiFourChainOracle(*params, n_sites=n_sites, periodic=False,
                                    n_grid=n_grid, bound=bound)

        x, w = dense_configurations(n_sites, grid, weights)
        density = np.exp(-chain_action(x, *params, False)) * w
        expected = density.reshape((n_grid,) * n_sites).sum(axis=(1, 2)) / weights
        expected /= np.trapezoid(expected, grid)

        np.testing.assert_allclose(oracle.site_marginal(site=0), expected, rtol=1e-8)

    @pytest.mark.parametrize("separation", [1, 2])
    def test_two_point_matches_dense_quadrature(self, separation):
        """
        Claim: E[x_i x_j] for sites a given number of bonds apart.
        Bug: the separation walked in the wrong direction, or the pair joint
        built from mismatched message lengths.
        Oracle: the dense configuration table.
        """
        params, n_sites = (0.8, 1.0, 1.5, 0.3), 3
        n_grid, bound = 61, 3.0
        grid, weights = trapezoid_grid(n_grid, bound)
        oracle = PhiFourChainOracle(*params, n_sites=n_sites, periodic=False,
                                    n_grid=n_grid, bound=bound)

        x, w = dense_configurations(n_sites, grid, weights)
        p = w * np.exp(-chain_action(x, *params, False))
        p /= p.sum()
        expected = float(np.sum(p * x[:, 0] * x[:, separation]))

        assert oracle.two_point(separation, site=0) == pytest.approx(expected, rel=1e-8)

    def test_gaussian_chain_moments(self):
        """
        Claim: mean and correlations agree with the Gaussian chain's own.
        Oracle: the inverse precision matrix.
        """
        n_sites, kappa, h = 4, 2.0, 0.3
        _, mean, covariance, _ = gaussian_chain(n_sites, kappa, h)
        oracle = PhiFourChainOracle(u=0.0, a=1.0, kappa=kappa, h=h, n_sites=n_sites,
                                    periodic=False, n_grid=2001, bound=8.0)

        assert oracle.mean_field(site=1) == pytest.approx(mean[1], rel=1e-5)
        assert oracle.two_point(1, site=1) == pytest.approx(
            covariance[1, 2] + mean[1] * mean[2], rel=1e-5
        )

    def test_sign_symmetry_at_zero_field(self):
        """
        Claim: at h = 0 the marginal is even and the mean field is exactly zero,
        which is the exact check this target offers a sampler.
        Bug: an asymmetric grid, or a field term that survives h = 0.
        Oracle: the mirrored marginal.
        """
        oracle = PhiFourChainOracle(u=1.0, a=1.0, kappa=1.5, h=0.0, n_sites=8,
                                    periodic=True, n_grid=401, bound=3.0)
        marginal = oracle.site_marginal()

        np.testing.assert_allclose(marginal, marginal[::-1], rtol=1e-12, atol=1e-14)
        assert oracle.mean_field() == pytest.approx(0.0, abs=1e-12)

    def test_every_site_of_a_ring_has_the_same_marginal(self):
        """
        Claim: a ring is translation invariant, so the site index does not matter.
        Bug: messages of the wrong length, which would break at the seam.
        Oracle: the marginal at another site.
        """
        oracle = PhiFourChainOracle(u=0.8, a=1.0, kappa=1.5, h=0.3, n_sites=6,
                                    periodic=True, n_grid=201, bound=3.0)

        np.testing.assert_allclose(oracle.site_marginal(site=0),
                                   oracle.site_marginal(site=4), rtol=1e-10)

    def test_dirichlet_ends_are_not_equivalent(self):
        """Fixture strength: with pinned ends the sites genuinely differ."""
        oracle = PhiFourChainOracle(u=0.8, a=1.0, kappa=1.5, h=0.0, n_sites=6,
                                    periodic=False, n_grid=201, bound=3.0)

        assert not np.allclose(oracle.site_marginal(site=0),
                               oracle.site_marginal(site=2), rtol=1e-3)


class TestCorrelationLength:

    @pytest.mark.parametrize("u,kappa,expected_decay", [(1.0, 2.0, None), (2.0, 2.0, None)])
    def test_matches_the_decay_of_the_correlation_function(self, u, kappa, expected_decay):
        """
        Claim: the correlation length is 1 / log(lambda_1 / lambda_2) from the
        transfer spectrum, and it governs the decay of E[x_i x_{i+r}].
        Bug: the wrong pair of eigenvalues, or a reciprocal the wrong way up.
        Oracle: the decay measured from two_point at two separations, which
        uses the eigenvectors and the pair joint, not the gap.
        """
        oracle = PhiFourChainOracle(u=u, a=1.0, kappa=kappa, h=0.0, n_sites=96,
                                    periodic=True, n_grid=301, bound=3.0)

        near, far = oracle.two_point(4), oracle.two_point(12)
        from_decay = (12 - 4) / np.log(near / far)

        assert oracle.correlation_length() == pytest.approx(from_decay, rel=1e-3)

    def test_stronger_coupling_orders_the_chain(self):
        """
        Claim: the correlation length grows with the coupling, which is what
        makes a chain of fixed length behave as one domain or as many.
        Oracle: monotonicity, an invariant of the model rather than of the code.
        """
        lengths = [PhiFourChainOracle(u=1.0, a=1.0, kappa=kappa, h=0.0, n_sites=64,
                                      periodic=True, n_grid=201, bound=3.0).correlation_length()
                   for kappa in (1.0, 2.0, 4.0)]

        assert lengths[0] < lengths[1] < lengths[2]


class UniformStub:
    """An rng whose uniforms are prescribed, so a draw can be checked by hand."""

    def __init__(self, values):
        self.values = list(values)

    def random(self, size):
        n = int(np.prod(size))
        out = np.array([self.values.pop(0) for _ in range(n)])
        return out.reshape(size)


class TestSampler:

    @pytest.mark.parametrize("periodic", [True, False])
    def test_the_sampler_law_is_the_chain(self, periodic):
        """
        Claim: the conditionals the sampler draws from multiply out to the exact
        discretized law of the chain.
        Bug: a message of the wrong length, a conditional that forgets the ring
        closes on the first site, a missing quadrature weight.
        Oracle: exact enumeration of all n_grid^n_sites configurations against
        the dense weight table. No sampling is involved.
        """
        params, n_sites, n_grid, bound = (0.9, 1.0, 1.2, 0.2), 3, 5, 2.0
        grid, weights = trapezoid_grid(n_grid, bound)
        oracle = PhiFourChainOracle(*params, n_sites=n_sites, periodic=periodic,
                                    n_grid=n_grid, bound=bound)

        first = oracle._first_site_probabilities()
        law = np.zeros((n_grid,) * n_sites)
        for j0 in range(n_grid):
            for j1 in range(n_grid):
                p1 = oracle._conditional(1, np.array([j0]), np.array([j0]))[0, j1]
                for j2 in range(n_grid):
                    p2 = oracle._conditional(2, np.array([j0]), np.array([j1]))[0, j2]
                    law[j0, j1, j2] = first[j0] * p1 * p2

        x, w = dense_configurations(n_sites, grid, weights)
        expected = w * np.exp(-chain_action(x, *params, periodic))
        expected = (expected / expected.sum()).reshape((n_grid,) * n_sites)

        assert law.sum() == pytest.approx(1.0, rel=1e-12)
        np.testing.assert_allclose(law, expected, rtol=1e-10)

    def test_a_draw_follows_the_conditionals_for_given_uniforms(self):
        """
        Claim: `sample` maps its uniforms through the conditionals by inverse
        CDF, one site at a time, and jitters inside the chosen cell.
        Bug: uniforms consumed in another order, or a site drawn from the wrong
        conditional. Oracle: the same inverse CDF applied by hand.
        """
        params, n_sites, n_grid, bound = (0.9, 1.0, 1.2, 0.0), 3, 5, 2.0
        oracle = PhiFourChainOracle(*params, n_sites=n_sites, periodic=True,
                                    n_grid=n_grid, bound=bound)
        choices, jitters = [0.05, 0.5, 0.95], [0.25, 0.5, 0.75]

        drawn = oracle.sample(UniformStub(choices + jitters), 1)[0]

        first = np.searchsorted(np.cumsum(oracle._first_site_probabilities()), choices[0])
        cells = [first]
        for site in (1, 2):
            cond = oracle._conditional(site, np.array([first]), np.array([cells[-1]]))[0]
            cells.append(int(np.searchsorted(np.cumsum(cond), choices[site])))
        spacing = oracle.grid[1] - oracle.grid[0]
        expected = [oracle.grid[c] + (j - 0.5) * spacing for c, j in zip(cells, jitters)]

        np.testing.assert_allclose(drawn, expected, rtol=1e-12)

    def test_sample_shape_and_support(self):
        oracle = PhiFourChainOracle(u=0.9, a=1.0, kappa=1.2, n_sites=5, periodic=True,
                                    n_grid=51, bound=3.0)

        x = oracle.sample(np.random.default_rng(0), 32)

        assert x.shape == (32, 5)
        assert np.all(np.abs(x) <= 3.0 + 1e-9)

    @pytest.mark.parametrize("byte_limit", [399, 400])
    def test_sampling_a_ring_guards_its_memory(self, monkeypatch, byte_limit):
        """Two 5x5 float64 transfer powers need 400 bytes: exactly at the limit.

        Lower the budget, not the work performed by the sampler, to exercise
        both sides of the guard without allocating large transfer matrices.
        A broken guard therefore fails an assertion instead of risking an OOM.
        """
        monkeypatch.setattr(oracle_module, "_MAX_POWER_BYTES", byte_limit)
        oracle = PhiFourChainOracle(n_sites=3, periodic=True, n_grid=5, bound=3.0)

        if byte_limit < 400:
            with pytest.raises(MemoryError, match="n_grid"):
                oracle.sample(np.random.default_rng(0), 1)
        else:
            assert oracle.sample(np.random.default_rng(0), 1).shape == (1, 3)


class TestGeneralLocalPotential:
    """The tempered path of a chain stays in this family, with coefficients the
    (u, a, h) form cannot express, which is what from_coefficients is for."""

    def test_coefficients_reproduce_the_well_form(self):
        """
        Claim: u (x^2 - a^2)^2 - h x is the local potential with coefficients
        (u, -2 u a^2, -h, u a^4), so the two constructors agree.
        Bug: a factor or a sign in the expansion, or the constant dropped,
        which would shift log Z by n_sites times it.
        Oracle: the (u, a, h) constructor itself.
        """
        u, a, h, kappa, n = 0.8, 1.2, 0.3, 1.5, 6
        direct = PhiFourChainOracle(u=u, a=a, kappa=kappa, h=h, n_sites=n,
                                    periodic=True, n_grid=201, bound=3.0)
        expanded = PhiFourChainOracle.from_coefficients(
            quartic=u, quadratic=-2 * u * a**2, linear=-h, constant=u * a**4,
            kappa=kappa, n_sites=n, periodic=True, n_grid=201, bound=3.0)

        assert expanded.log_partition() == pytest.approx(direct.log_partition(), rel=1e-12)
        np.testing.assert_allclose(expanded.site_marginal(), direct.site_marginal(), rtol=1e-12)

    @pytest.mark.parametrize("periodic", [True, False])
    def test_a_positive_quadratic_term_is_solved(self, periodic):
        """
        Claim: a chain whose local potential has a positive quadratic term, as
        every early level of a tempered path does, is solved like any other.
        The (u, a, h) form cannot express it, since -2 u a^2 is never positive.
        Oracle: dense quadrature over every configuration of a three-site chain.
        """
        quartic, quadratic, linear, kappa, n = 0.4, 1.3, -0.2, 1.1, 3
        n_grid, bound = 121, 3.0
        grid, weights = trapezoid_grid(n_grid, bound)
        oracle = PhiFourChainOracle.from_coefficients(
            quartic=quartic, quadratic=quadratic, linear=linear, kappa=kappa,
            n_sites=n, periodic=periodic, n_grid=n_grid, bound=bound)

        x, w = dense_configurations(n, grid, weights)
        local = np.sum(quartic * x**4 + quadratic * x**2 + linear * x, axis=-1)
        if periodic:
            bonds = np.sum((x - np.roll(x, -1, axis=-1)) ** 2, axis=-1)
        else:
            zero = np.zeros_like(x[..., :1])
            bonds = np.sum(np.diff(np.concatenate([zero, x, zero], axis=-1), axis=-1) ** 2, axis=-1)
        log_weight = np.log(w) - (local + kappa / 2 * bonds)
        top = log_weight.max()
        expected = top + np.log(np.sum(np.exp(log_weight - top)))

        assert oracle.log_partition() == pytest.approx(expected, rel=1e-10)

    def test_a_quartic_of_zero_needs_a_positive_quadratic(self):
        """Without a quartic term the integral diverges unless the quadratic
        confines it, so the constructor refuses the unbounded case."""
        with pytest.raises(ValueError, match="quartic"):
            PhiFourChainOracle.from_coefficients(quartic=0.0, quadratic=-1.0, linear=0.0,
                                                 kappa=1.0, n_sites=4)
        assert PhiFourChainOracle.from_coefficients(quartic=0.0, quadratic=1.0, linear=0.0,
                                                    kappa=1.0, n_sites=4, n_grid=51).log_partition()

    def test_pinned_ends_confine_where_a_ring_does_not(self):
        """
        Claim: with no local confinement the integral converges on a chain with
        pinned ends, whose end bonds hold the field, and diverges on a ring,
        whose uniform mode is free. The check must see the boundary condition.
        Bug: one rule for both, which would either refuse a solvable chain (the
        Gaussian chain that u = 0 leaves) or accept a divergent ring.
        Oracle: the Gaussian chain's closed-form log Z, which exists only in
        the first case.
        """
        chain = PhiFourChainOracle.from_coefficients(quartic=0.0, quadratic=0.0, linear=0.0,
                                                     kappa=1.0, n_sites=4, periodic=False,
                                                     n_grid=401, bound=8.0)
        _, _, _, expected = gaussian_chain(4, 1.0, 0.0)

        assert chain.log_partition() == pytest.approx(expected, rel=1e-6)
        with pytest.raises(ValueError, match="positive definite"):
            PhiFourChainOracle.from_coefficients(quartic=0.0, quadratic=0.0, linear=0.0,
                                                 kappa=1.0, n_sites=4, periodic=True)

    def test_the_check_weighs_the_local_term_as_the_energy_does(self):
        """
        Claim: the energy carries `quadratic * x^2` while a quadratic form
        carries `x^2 / 2`, so the check adds twice the coefficient.
        Bug: the factor dropped, which accepts a divergent chain. On four
        pinned sites at kappa = 1 the Laplacian's smallest eigenvalue is 0.382,
        so quadratic = -0.2 is unbounded (0.382 - 0.4 < 0) while a check
        missing the factor would see 0.182 and accept it.
        Oracle: the eigenvalue arithmetic, written out above.
        """
        with pytest.raises(ValueError, match="positive definite"):
            PhiFourChainOracle.from_coefficients(quartic=0.0, quadratic=-0.2, linear=0.0,
                                                 kappa=1.0, n_sites=4, periodic=False)
        accepted = PhiFourChainOracle.from_coefficients(quartic=0.0, quadratic=-0.15, linear=0.0,
                                                        kappa=1.0, n_sites=4, periodic=False,
                                                        n_grid=401, bound=8.0)
        assert np.isfinite(accepted.log_partition())

    def test_the_report_keeps_the_coefficients(self):
        """discretization_report rebuilds the chain, so it must carry them."""
        oracle = PhiFourChainOracle.from_coefficients(quartic=0.4, quadratic=1.3, linear=-0.2,
                                                      kappa=1.1, n_sites=4, n_grid=101, bound=3.0)

        report = oracle.discretization_report()

        assert report["log_partition"] == pytest.approx(oracle.log_partition(), rel=1e-12)
        assert abs(report["refinement_change"]) < 1e-6


class TestFromDistribution:

    def test_from_lattice_phi_four(self):
        dist = LatticePhiFour(u=0.7, a=1.3, kappa=2.1, h=0.4, lattice_shape=(6,))

        oracle = PhiFourChainOracle.from_distribution(dist, n_grid=51, bound=3.0)

        assert (oracle.u, oracle.a, oracle.kappa, oracle.h) == (0.7, 1.3, 2.1, 0.4)
        assert oracle.n_sites == 6 and oracle.periodic

    def test_from_phi_four_uses_the_documented_mapping(self):
        """
        Claim: PhiFour maps through kappa = beta c, u = beta / (4c), a = 1,
        h = -beta b / c with c = a * dim_grid.
        Oracle: an oracle built from the equivalent LatticePhiFour, which must
        give the same partition function.
        """
        n, a_jp, b, beta = 6, 0.1, 0.3, 1.5
        c = a_jp * n
        chain = PhiFour(a=a_jp, b=b, dim_grid=n, beta=beta, periodic=True)

        oracle = PhiFourChainOracle.from_distribution(chain, n_grid=201, bound=3.0)
        equivalent = PhiFourChainOracle(u=beta / (4 * c), a=1.0, kappa=beta * c,
                                        h=-beta * b / c, n_sites=n, periodic=True,
                                        n_grid=201, bound=3.0)

        assert oracle.log_partition() == pytest.approx(equivalent.log_partition(), rel=1e-12)

    def test_dirichlet_phi_four_is_carried_over(self):
        oracle = PhiFourChainOracle.from_distribution(PhiFour(dim_grid=5, periodic=False),
                                                      n_grid=51, bound=3.0)

        assert not oracle.periodic

    @pytest.mark.parametrize("dist", [LatticePhiFour(lattice_shape=(4, 4)), "not a distribution"])
    def test_rejects_what_it_cannot_solve(self, dist):
        """A chain oracle must refuse a lattice with more than one axis."""
        with pytest.raises(TypeError):
            PhiFourChainOracle.from_distribution(dist)


class TestDiscretizationReport:

    def test_an_adequate_grid_reports_small_errors(self):
        """
        Claim: the report measures the discretization error rather than assuming
        it, so on a grid that resolves the chain every difference is tiny.
        Oracle: the report's own refinements, which are independent computations.
        """
        oracle = PhiFourChainOracle(u=1.0, a=1.0, kappa=1.5, h=0.0, n_sites=8,
                                    periodic=True, n_grid=201, bound=4.0)

        report = oracle.discretization_report()

        assert abs(report["refinement_change"]) < 1e-6
        assert abs(report["extent_change"]) < 1e-6
        assert report["tail_density"] < 1e-12

    def test_a_narrow_grid_is_flagged(self):
        """
        Fixture strength: a grid that cuts the distribution off shows up in the
        report, so a user cannot mistake a truncated answer for an exact one.
        """
        oracle = PhiFourChainOracle(u=1.0, a=1.0, kappa=1.5, h=0.0, n_sites=8,
                                    periodic=True, n_grid=201, bound=0.6)

        report = oracle.discretization_report()

        assert abs(report["extent_change"]) > 1e-3
        assert report["tail_density"] > 1e-2


class TestValidation:

    @pytest.mark.parametrize("kwargs,message", [
        ({"u": -1.0}, "u must be non-negative"),
        ({"a": 0.0}, "a must be positive"),
        ({"kappa": 0.0}, "kappa must be positive"),
        ({"n_sites": 1}, "n_sites must be >= 2"),
        ({"n_grid": 2}, "n_grid must be >= 3"),
        ({"bound": 0.0}, "bound must be positive"),
    ])
    def test_validation(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            PhiFourChainOracle(**kwargs)
