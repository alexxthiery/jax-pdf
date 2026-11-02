"""Gaussian mixture: the standard multimodal target, exact in every quantity.

Unlike the physical targets here, nothing about a Gaussian mixture is
approximate. The density, its normalizing constant, its moments, its posterior
over components and its sampler are all closed form, which is what makes it the
reference case for a sampler study: an estimate can be scored against truth
rather than against another estimate.

The density over `R^dim` is

    p(x) = sum_c w_c N(x; mu_c, Sigma_c),    Sigma_c = L_c L_c^T

with `L_c` lower triangular. Following this package's convention `__call__`
returns the density as written, so the weights are not normalized for the
caller and `log_normalization()` reports `log sum_c w_c`.

The scale is stored as its Cholesky factor rather than as `Sigma_c`, so
evaluation is one triangular solve with no factorization inside the call and
none repeated under `grad` or `vmap`. The three `from_*` constructors all take
the same thing, the component covariance, at whatever shape it is held in, and
direct construction from `scale_tril` stays the primary path.
"""

import math

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct
from jax import Array
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp

from jax_pdf._validation import check_event_shape, is_concrete

_LOG_2PI = math.log(2.0 * math.pi)


@struct.dataclass
class GaussianMixture:
    """Mixture of Gaussians with per-component full covariance.

    Attributes:
        weights: Component weights `w_c`, shape `(n_components,)`. Must be
            non-negative with a positive sum; they need not sum to one, and a
            zero switches a component off.
        means: Component means `mu_c`, shape `(n_components, dim)`.
        scale_tril: Cholesky factors `L_c` with `Sigma_c = L_c L_c^T`, shape
            `(n_components, dim, dim)`, lower triangular with a positive
            diagonal.
    """

    weights: Array
    means: Array
    scale_tril: Array

    def __post_init__(self):
        # vmap probes the pytree with object placeholders, which carry no shape
        # and no value. Arrays and tracers still expose shapes, so shape checks
        # stay active for them while numeric checks wait for concrete values.
        arrays = (self.weights, self.means, self.scale_tril)
        if all(type(a) is not object for a in arrays):
            if jnp.ndim(self.weights) != 1:
                raise ValueError(
                    f"GaussianMixture: weights must be (n_components,), got shape "
                    f"{jnp.shape(self.weights)}.")
            if jnp.ndim(self.means) != 2:
                raise ValueError(
                    f"GaussianMixture: means must be (n_components, dim), got shape "
                    f"{jnp.shape(self.means)}.")
            n_components, dim = jnp.shape(self.means)
            if jnp.shape(self.weights)[0] != n_components:
                raise ValueError(
                    f"GaussianMixture: one weight per component expected, got "
                    f"{jnp.shape(self.weights)[0]} weights for {n_components} means.")
            if jnp.shape(self.scale_tril) != (n_components, dim, dim):
                raise ValueError(
                    f"GaussianMixture: scale_tril must be "
                    f"{(n_components, dim, dim)}, got shape {jnp.shape(self.scale_tril)}.")
        if is_concrete(self.weights) and jnp.ndim(self.weights) == 1:
            if bool(jnp.any(self.weights < 0)):
                raise ValueError(
                    f"GaussianMixture: weights must be non-negative, got {self.weights}.")
            if not bool(jnp.sum(self.weights) > 0):
                raise ValueError(
                    f"GaussianMixture: weights must sum to a positive value, got "
                    f"{self.weights}.")
        if is_concrete(self.scale_tril) and jnp.ndim(self.scale_tril) == 3:
            if not bool(jnp.all(jnp.isfinite(self.scale_tril))):
                raise ValueError(
                    "GaussianMixture: scale_tril is not finite. A NaN factor is what "
                    "jnp.linalg.cholesky returns for a covariance that is not positive "
                    "definite, so check the covariances given to from_covariance.")
            if not bool(jnp.allclose(self.scale_tril, jnp.tril(self.scale_tril))):
                raise ValueError(
                    "GaussianMixture: scale_tril must be lower triangular; found a "
                    "nonzero entry above the diagonal. Pass covariances to "
                    "GaussianMixture.from_covariance instead.")
            diagonal = jnp.diagonal(self.scale_tril, axis1=-2, axis2=-1)
            if not bool(jnp.all(diagonal > 0)):
                raise ValueError(
                    f"GaussianMixture: scale_tril must have a positive diagonal, got "
                    f"diagonals {diagonal}.")

    # ---------------------------------------------------------------- building
    @classmethod
    def from_covariance(cls, weights, means, covariances) -> "GaussianMixture":
        """Build from covariance matrices, factorizing them once here.

        Args:
            weights: Shape `(n_components,)`.
            means: Shape `(n_components, dim)`.
            covariances: Shape `(n_components, dim, dim)`, each positive
                definite.

        Returns:
            The mixture, holding the Cholesky factors.
        """
        return cls(weights=jnp.asarray(weights), means=jnp.asarray(means),
                   scale_tril=jnp.linalg.cholesky(jnp.asarray(covariances)))

    @classmethod
    def from_isotropic_variance(cls, weights, means, variance) -> "GaussianMixture":
        """Build with `Sigma_c = variance_c * I`.

        Args:
            weights: Shape `(n_components,)`.
            means: Shape `(n_components, dim)`.
            variance: A scalar shared by every component, or one variance per
                component, shape `(n_components,)`.

        Returns:
            The mixture.
        """
        means = jnp.asarray(means)
        n_components, dim = jnp.shape(means)
        scale = jnp.sqrt(jnp.broadcast_to(jnp.asarray(variance), (n_components,)))
        return cls(weights=jnp.asarray(weights), means=means,
                   scale_tril=scale[:, None, None] * jnp.eye(dim, dtype=scale.dtype))

    @classmethod
    def from_diagonal_variances(cls, weights, means, variances) -> "GaussianMixture":
        """Build with a diagonal covariance per component.

        Args:
            weights: Shape `(n_components,)`.
            means: Shape `(n_components, dim)`.
            variances: Per-coordinate variances, shape `(n_components, dim)`.

        Returns:
            The mixture.
        """
        scale = jnp.sqrt(jnp.asarray(variances))
        return cls(weights=jnp.asarray(weights), means=jnp.asarray(means),
                   scale_tril=jax.vmap(jnp.diag)(scale))

    # ----------------------------------------------------------------- shape
    @property
    def n_components(self) -> int:
        return int(jnp.shape(self.means)[0])

    @property
    def dim(self) -> int:
        return int(jnp.shape(self.means)[1])

    # --------------------------------------------------------------- density
    def component_log_prob(self, x: Array) -> Array:
        """Normalized log density of each component separately.

        Each term is `-||L_c^{-1}(x - mu_c)||^2 / 2 - log det L_c
        - (dim / 2) log 2 pi`, the Gaussian written through its Cholesky factor.

        Args:
            x: Input point(s) of shape `(..., dim)`.

        Returns:
            Log densities of shape `(..., n_components)`.

        Raises:
            ValueError: If the input does not have trailing shape `(dim,)`.
        """
        check_event_shape(x, (self.dim,))
        difference = x[..., None, :] - self.means
        whitened = solve_triangular(self.scale_tril, difference[..., None], lower=True)
        log_det = jnp.sum(
            jnp.log(jnp.diagonal(self.scale_tril, axis1=-2, axis2=-1)), axis=-1)
        return (-0.5 * jnp.sum(whitened[..., 0] ** 2, axis=-1)
                - log_det - 0.5 * self.dim * _LOG_2PI)

    def __call__(self, x: Array) -> Array:
        """Evaluate the mixture log-density, with the weights as given.

        Args:
            x: Input point(s) of shape `(..., dim)`.

        Returns:
            Log-density of shape `(...)`.

        Raises:
            ValueError: If the input does not have trailing shape `(dim,)`.
        """
        return logsumexp(jnp.log(self.weights) + self.component_log_prob(x), axis=-1)

    def log_normalization(self) -> Array:
        """Log normalizing constant, exactly `log sum_c w_c`.

        Returns:
            Scalar `log Z`; zero when the weights sum to one.
        """
        return jnp.log(jnp.sum(self.weights))

    def log_responsibilities(self, x: Array) -> Array:
        """Log posterior over components, `log P(c | x)`.

        `argmax` of this is the component a point is attributed to, which is how
        a recovered mode weight is measured without hand-placing a boundary.

        Args:
            x: Input point(s) of shape `(..., dim)`.

        Returns:
            Log probabilities of shape `(..., n_components)`, summing to one
            over the last axis.
        """
        joint = jnp.log(self.weights) + self.component_log_prob(x)
        return joint - logsumexp(joint, axis=-1, keepdims=True)

    # -------------------------------------------------------- moments, draws
    def _probabilities(self) -> Array:
        return self.weights / jnp.sum(self.weights)

    def mean(self) -> Array:
        """Exact mean `sum_c p_c mu_c`, shape `(dim,)`."""
        return self._probabilities() @ self.means

    def covariance(self) -> Array:
        """Exact covariance, shape `(dim, dim)`.

        The law of total covariance: the weighted mean of the component
        covariances plus the covariance of the means.
        """
        probabilities = self._probabilities()
        within = jnp.einsum("k,kij->ij", probabilities,
                            self.scale_tril @ jnp.swapaxes(self.scale_tril, -1, -2))
        deviation = self.means - self.mean()
        return within + jnp.einsum("k,ki,kj->ij", probabilities, deviation, deviation)

    def sample(self, key: Array, n: int) -> Array:
        """Draw exact samples.

        Args:
            key: JAX PRNG key.
            n: Number of samples.

        Returns:
            Samples of shape `(n, dim)`.
        """
        return self.sample_with_component(key, n)[0]

    def sample_with_component(self, key: Array, n: int) -> tuple[Array, Array]:
        """Draw exact samples and report which component produced each one.

        The labels are the ground truth a mode-recovery measurement is scored
        against, and the check that `log_responsibilities` attributes points the
        way the mixture generated them.

        Args:
            key: JAX PRNG key. The component labels come from the first half of
                its split and the Gaussian innovations from the second.
            n: Number of samples.

        Returns:
            Samples of shape `(n, dim)` and component indices of shape `(n,)`.
        """
        key_component, key_normal = jr.split(key)
        component = jr.choice(key_component, self.n_components, shape=(n,),
                              p=self._probabilities())
        innovation = jr.normal(key_normal, (n, self.dim))
        return (self.means[component]
                + jnp.einsum("nij,nj->ni", self.scale_tril[component], innovation),
                component)
