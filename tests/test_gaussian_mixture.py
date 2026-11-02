"""Tests for the GaussianMixture target.

Oracles are written from the definition of a Gaussian mixture in float64 NumPy
(explicit quadratic form, `slogdet`, a plain weighted sum of component
densities), which is independent of the implementation's Cholesky solve and
log-space arithmetic. No test averages samples.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_pdf import GaussianMixture

# Three components in four dimensions, correlated and of unequal width: enough
# structure that a transposed Cholesky, a dropped log-determinant or a wrong
# component axis all change the answer.
WEIGHTS = np.array([0.5, 0.3, 0.2])
MEANS = np.array([
    [-2.0, 0.0, 1.0, 0.5],
    [2.0, 1.0, -1.0, 0.0],
    [0.0, -2.0, 0.5, 1.5],
])
_FACTORS = np.array([
    [[1.0, 0.0, 0.0, 0.0], [0.5, 1.2, 0.0, 0.0], [0.0, -0.4, 0.8, 0.0], [0.2, 0.1, 0.3, 1.1]],
    [[0.7, 0.0, 0.0, 0.0], [-0.3, 0.9, 0.0, 0.0], [0.1, 0.2, 1.3, 0.0], [0.0, -0.5, 0.2, 0.6]],
    [[1.4, 0.0, 0.0, 0.0], [0.2, 0.5, 0.0, 0.0], [-0.6, 0.3, 1.0, 0.0], [0.4, 0.0, -0.2, 0.9]],
])
COVS = np.einsum("kij,klj->kil", _FACTORS, _FACTORS)
POINTS = np.linspace(-2.5, 2.5, 20).reshape(5, 4)


def _oracle_components(x, means=MEANS, covs=COVS):
    """log N(x; mu_c, Sigma_c) per component, straight from the definition."""
    x = np.asarray(x, np.float64)
    terms = []
    for mean, cov in zip(np.asarray(means, np.float64), np.asarray(covs, np.float64)):
        diff = x - mean
        quadratic = np.einsum("...i,ij,...j->...", diff, np.linalg.inv(cov), diff)
        log_det = np.linalg.slogdet(cov)[1]
        terms.append(-0.5 * quadratic - 0.5 * log_det
                     - 0.5 * x.shape[-1] * np.log(2.0 * np.pi))
    return np.stack(terms, axis=-1)


def _oracle_log_prob(x, weights=WEIGHTS, **kwargs):
    """log sum_c w_c N(x; mu_c, Sigma_c), summed outside log space on purpose."""
    return np.log(np.sum(np.asarray(weights, np.float64)
                         * np.exp(_oracle_components(x, **kwargs)), axis=-1))


@pytest.fixture(scope="module")
def mixture():
    return GaussianMixture.from_covariance(WEIGHTS, MEANS, COVS)


class TestDensity:
    def test_component_log_prob_matches_the_definition(self, mixture):
        """Each component is the Gaussian its mean and covariance describe."""
        actual = mixture.component_log_prob(jnp.asarray(POINTS))
        assert actual.shape == (5, 3)
        np.testing.assert_allclose(actual, _oracle_components(POINTS), rtol=2e-5, atol=2e-5)

    def test_call_matches_the_definition(self, mixture):
        """The mixture is the weighted sum of its components."""
        actual = mixture(jnp.asarray(POINTS))
        np.testing.assert_allclose(actual, _oracle_log_prob(POINTS), rtol=2e-5, atol=2e-5)

    def test_single_component_is_a_multivariate_normal(self):
        """K = 1 reduces to one Gaussian, log-determinant and all."""
        one = GaussianMixture.from_covariance(WEIGHTS[:1], MEANS[:1], COVS[:1])
        expected = (np.log(WEIGHTS[0])
                    + _oracle_components(POINTS, MEANS[:1], COVS[:1])[..., 0])
        np.testing.assert_allclose(one(jnp.asarray(POINTS)), expected, rtol=2e-5, atol=2e-5)

    def test_identical_components_collapse_to_one_gaussian(self):
        """Splitting one Gaussian's weight across copies cannot change the density.

        Oracle: an algebraic identity, independent of any numeric reference.
        """
        mean, cov = MEANS[0], COVS[0]
        split = GaussianMixture.from_covariance(
            [0.3, 0.7], np.stack([mean, mean]), np.stack([cov, cov]))
        whole = GaussianMixture.from_covariance([1.0], mean[None], cov[None])
        np.testing.assert_allclose(split(jnp.asarray(POINTS)), whole(jnp.asarray(POINTS)),
                                   rtol=1e-6, atol=1e-6)

    def test_a_zero_weight_drops_a_component_without_nan(self, mixture):
        """log 0 must leave the sum finite, not poison it.

        Bug it catches: a logsumexp over weights that produces NaN from -inf,
        or validation that forbids switching a component off.
        """
        muted = GaussianMixture.from_covariance([0.5, 0.0, 0.2], MEANS, COVS)
        kept = GaussianMixture.from_covariance([0.5, 0.2], MEANS[[0, 2]], COVS[[0, 2]])
        actual = muted(jnp.asarray(POINTS))
        assert jnp.all(jnp.isfinite(actual))
        np.testing.assert_allclose(actual, kept(jnp.asarray(POINTS)), rtol=1e-6, atol=1e-6)


class TestNormalization:
    def test_log_normalization_is_the_log_total_weight(self):
        """Weights are not normalized for the caller; log Z reports their sum."""
        unnormalized = GaussianMixture.from_covariance([2.0, 1.0, 1.0], MEANS, COVS)
        assert float(unnormalized.log_normalization()) == pytest.approx(np.log(4.0), rel=1e-6)

    def test_normalized_weights_give_zero(self, mixture):
        assert float(mixture.log_normalization()) == pytest.approx(0.0, abs=1e-6)

    def test_density_integrates_to_the_total_weight(self):
        """Quadrature in one dimension ties __call__ to log_normalization."""
        line = GaussianMixture.from_isotropic_variance([0.7, 0.4], [[-1.5], [2.0]], 0.5)
        grid = jnp.linspace(-12.0, 12.0, 200_001)
        integral = jnp.trapezoid(jnp.exp(line(grid[:, None])), grid)
        assert float(jnp.log(integral)) == pytest.approx(float(line.log_normalization()),
                                                         abs=1e-5)


class TestResponsibilities:
    def test_responsibilities_are_the_normalized_component_terms(self, mixture):
        """Oracle: the posterior over components, formed outside log space."""
        weighted = WEIGHTS * np.exp(_oracle_components(POINTS))
        expected = weighted / weighted.sum(axis=-1, keepdims=True)
        actual = jnp.exp(mixture.log_responsibilities(jnp.asarray(POINTS)))
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)

    def test_responsibilities_sum_to_one(self, mixture):
        total = jnp.sum(jnp.exp(mixture.log_responsibilities(jnp.asarray(POINTS))), axis=-1)
        np.testing.assert_allclose(total, 1.0, atol=1e-6)

    def test_a_symmetric_pair_is_undecided_at_its_midpoint(self):
        """Equal weights and equal covariance: the midpoint splits exactly."""
        pair = GaussianMixture.from_isotropic_variance([0.5, 0.5], [[-1.0, 0.0], [1.0, 0.0]], 0.3)
        actual = jnp.exp(pair.log_responsibilities(jnp.zeros(2)))
        np.testing.assert_allclose(actual, 0.5, atol=1e-6)


class TestConstructors:
    def test_isotropic_matches_a_scaled_identity_covariance(self):
        iso = GaussianMixture.from_isotropic_variance(WEIGHTS, MEANS, 0.25)
        same = GaussianMixture.from_covariance(
            WEIGHTS, MEANS, np.broadcast_to(0.25 * np.eye(4), (3, 4, 4)))
        np.testing.assert_allclose(iso(jnp.asarray(POINTS)), same(jnp.asarray(POINTS)),
                                   rtol=1e-6, atol=1e-6)

    def test_isotropic_accepts_one_variance_per_component(self):
        variances = np.array([0.1, 0.5, 1.0])
        iso = GaussianMixture.from_isotropic_variance(WEIGHTS, MEANS, variances)
        same = GaussianMixture.from_covariance(
            WEIGHTS, MEANS, variances[:, None, None] * np.eye(4))
        np.testing.assert_allclose(iso(jnp.asarray(POINTS)), same(jnp.asarray(POINTS)),
                                   rtol=1e-6, atol=1e-6)

    def test_diagonal_matches_a_diagonal_covariance(self):
        variances = np.array([[0.1, 0.2, 0.3, 0.4], [1.0, 0.5, 0.25, 2.0],
                              [0.7, 0.7, 0.7, 0.7]])
        diag = GaussianMixture.from_diagonal_variances(WEIGHTS, MEANS, variances)
        same = GaussianMixture.from_covariance(
            WEIGHTS, MEANS, np.stack([np.diag(v) for v in variances]))
        np.testing.assert_allclose(diag(jnp.asarray(POINTS)), same(jnp.asarray(POINTS)),
                                   rtol=1e-6, atol=1e-6)


class TestMoments:
    def test_mean_and_covariance_match_the_definition(self, mixture):
        """Oracle: E[x] and E[xx'] - E[x]E[x]' summed over components by hand."""
        p = WEIGHTS / WEIGHTS.sum()
        mean = p @ MEANS
        second = np.einsum("k,kij->ij", p, COVS) + np.einsum("k,ki,kj->ij", p, MEANS, MEANS)
        np.testing.assert_allclose(mixture.mean(), mean, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(mixture.covariance(), second - np.outer(mean, mean),
                                   rtol=1e-5, atol=1e-5)


class TestSample:
    def test_an_identity_scale_returns_the_raw_normals(self):
        """With unit scale and zero means the sampler is its own innovations.

        Pins the key contract: components from the first split, normals from the
        second. Oracle: those normals, written out here.
        """
        plain = GaussianMixture(weights=jnp.ones(3) / 3.0,
                                means=jnp.zeros((3, 4)),
                                scale_tril=jnp.broadcast_to(jnp.eye(4), (3, 4, 4)))
        key = jax.random.PRNGKey(0)
        expected = jax.random.normal(jax.random.split(key)[1], (6, 4))
        np.testing.assert_array_equal(plain.sample(key, 6), expected)

    def test_draws_whiten_back_to_their_innovations(self, mixture):
        """x = mu_c + L_c z must invert to z under the drawn component's L_c.

        Bug it catches: the scale applied transposed, or the mean of the wrong
        component added. Oracle: the innovations from the second key.
        """
        key = jax.random.PRNGKey(3)
        x, component = mixture.sample_with_component(key, 32)
        innovations = jax.random.normal(jax.random.split(key)[1], (32, 4))
        recovered = jnp.linalg.solve(mixture.scale_tril[component],
                                     (x - mixture.means[component])[..., None])[..., 0]
        np.testing.assert_allclose(recovered, innovations, rtol=1e-4, atol=1e-4)

    def test_a_zero_weight_component_is_never_drawn(self):
        """Oracle: the exact support of the categorical law, not its frequencies."""
        muted = GaussianMixture.from_isotropic_variance([0.0, 1.0, 0.0], MEANS, 0.25)
        _, component = muted.sample_with_component(jax.random.PRNGKey(1), 64)
        assert jnp.all(component == 1)


class TestValidation:
    def test_rejects_a_negative_weight(self):
        with pytest.raises(ValueError, match="weights must be"):
            GaussianMixture.from_isotropic_variance([0.5, -0.1, 0.6], MEANS, 0.25)

    def test_rejects_weights_that_sum_to_zero(self):
        with pytest.raises(ValueError, match="weights must sum"):
            GaussianMixture.from_isotropic_variance([0.0, 0.0, 0.0], MEANS, 0.25)

    def test_rejects_a_component_count_mismatch(self):
        with pytest.raises(ValueError, match="one weight per component"):
            GaussianMixture.from_isotropic_variance([0.5, 0.5], MEANS, 0.25)

    def test_rejects_a_wrong_scale_tril_shape(self):
        with pytest.raises(ValueError, match="scale_tril must be"):
            GaussianMixture(weights=jnp.asarray(WEIGHTS), means=jnp.asarray(MEANS),
                            scale_tril=jnp.ones((3, 4, 5)))

    def test_reports_a_covariance_that_is_not_positive_definite(self):
        """Cholesky returns NaN instead of raising, so the error must name the cause.

        Bug it catches: the triangularity check tripping on the NaNs first and
        telling the caller to use from_covariance, which is what they just did.
        """
        indefinite = np.array([[[1.0, 2.0], [2.0, 1.0]]])   # eigenvalues 3 and -1
        with pytest.raises(ValueError, match="positive definite"):
            GaussianMixture.from_covariance([1.0], np.zeros((1, 2)), indefinite)

    def test_rejects_an_entry_above_the_diagonal(self):
        upper = np.array(np.linalg.cholesky(COVS))
        upper[0, 0, 1] = 0.5
        with pytest.raises(ValueError, match="lower triangular"):
            GaussianMixture(weights=jnp.asarray(WEIGHTS), means=jnp.asarray(MEANS),
                            scale_tril=jnp.asarray(upper))

    def test_rejects_a_non_positive_diagonal(self):
        singular = np.array(np.linalg.cholesky(COVS))
        singular[1, 2, 2] = 0.0
        with pytest.raises(ValueError, match="positive diagonal"):
            GaussianMixture(weights=jnp.asarray(WEIGHTS), means=jnp.asarray(MEANS),
                            scale_tril=jnp.asarray(singular))

    def test_validation_is_skipped_under_tracing(self):
        """A tracer cannot be compared, so a mixture built inside jit is unchecked.

        This documents the cost of the static/traced split rather than blessing
        it. Oracle: the density the formula gives for the valid twin.
        """
        traced = jax.jit(lambda w: GaussianMixture.from_isotropic_variance(w, MEANS, 0.25)(POINTS[0]))(
            jnp.asarray([0.5, 0.3, 0.2]))
        assert jnp.isfinite(traced)
        assert float(traced) == pytest.approx(
            float(GaussianMixture.from_isotropic_variance(WEIGHTS, MEANS, 0.25)(jnp.asarray(POINTS[0]))),
            rel=1e-6)


class TestShape:
    def test_component_axis_is_appended(self, mixture):
        """Batch axes are preserved and the component axis is last."""
        batch = jnp.asarray(np.broadcast_to(POINTS, (2, 3, 5, 4)))
        assert mixture.component_log_prob(batch).shape == (2, 3, 5, 3)
        assert mixture.log_responsibilities(batch).shape == (2, 3, 5, 3)
        assert mixture(batch).shape == (2, 3, 5)

    def test_shape_properties(self, mixture):
        assert (mixture.n_components, mixture.dim) == (3, 4)
