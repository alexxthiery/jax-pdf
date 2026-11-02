# GaussianMixture

A mixture of Gaussians: the standard multimodal test target, and the only one here in which nothing is approximate. The density, its normalizing constant, its moments, its posterior over components and its sampler are all closed form, so an estimate can be scored against truth rather than against another estimate. Mode weights, in particular, are known exactly, which is what makes this the reference case for asking whether a sampler finds every mode and in the right proportion.

## Mathematical definition

The density over $\mathbb{R}^{d}$ is

$$
p(x) = \sum_{c=1}^{K} w_c \, N(x; \mu_c, \Sigma_c),
\qquad
\Sigma_c = L_c L_c^{\top},
$$

with $L_c$ lower triangular with a positive diagonal. Following this package's convention, `__call__` returns the density as written, so the weights are not normalized for the caller and

$$
\log Z = \log \sum_{c=1}^{K} w_c ,
$$

which is zero when the weights sum to one. A weight of zero switches a component off and leaves the density finite.

Each component is evaluated through its Cholesky factor,

$$
\log N(x; \mu_c, \Sigma_c)
= -\tfrac{1}{2} \lVert L_c^{-1} (x - \mu_c) \rVert^2
- \log \det L_c
- \tfrac{d}{2} \log 2\pi ,
$$

so one triangular solve evaluates every component, with no factorization inside the call and none repeated under `grad` or `vmap`. The scale is stored as $L_c$ for that reason; `from_covariance` factorizes $\Sigma_c$ once at construction.

The posterior over components,

$$
P(c \mid x) = \frac{w_c \, N(x; \mu_c, \Sigma_c)}{\sum_{c'} w_{c'} N(x; \mu_{c'}, \Sigma_{c'})} ,
$$

is returned in log space by `log_responsibilities`. Its `argmax` attributes a point to a component, which is how a recovered mode weight is measured without hand-placing a boundary. The mean and covariance are exact, the latter by the law of total covariance:

$$
\mathbb{E}[x] = \sum_c p_c \mu_c ,
\qquad
\operatorname{Cov}[x] = \sum_c p_c \Sigma_c + \sum_c p_c (\mu_c - \mathbb{E}[x])(\mu_c - \mathbb{E}[x])^{\top} ,
$$

with $p_c = w_c / \sum_{c'} w_{c'}$.

## How hard the target is

For two components sharing an isotropic covariance $\sigma^2 I$, the Bayes boundary is the perpendicular bisector of the means, and a draw from one component lands on the far side with probability

$$
\Phi\left( - \frac{\lVert \mu_2 - \mu_1 \rVert}{2 \sigma} \right) .
$$

That is the whole difficulty knob, and in high dimension a separation that looks mild per coordinate is not. Means at $0$ and $\tfrac{3}{4} \mathbf{1}_d$ with $\sigma^2 = 1/16$ are $\tfrac{3}{4}\sqrt{d}$ apart, or $3\sqrt{d}$ standard deviations, so the overlap is $\Phi(-1.5\sqrt{d})$: about $10^{-9}$ at $d = 16$ and $10^{-17}$ at $d = 32$. The components are then disjoint for any finite sample, and a sampler that misses one cannot discover it from the other.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weights` | required | Component weights $w_c$, shape `(n_components,)`; non-negative with a positive sum, not required to sum to one |
| `means` | required | Component means $\mu_c$, shape `(n_components, dim)` |
| `scale_tril` | required | Cholesky factors $L_c$, shape `(n_components, dim, dim)`, lower triangular with a positive diagonal |

`n_components` and `dim` are read off `means`, so there is no size parameter to keep consistent with the arrays.

## Usage

Input shape is `(..., dim)`; the output drops the last axis. The three constructors all take the same thing, the component covariance, at the shape it is held in: full matrices, per-coordinate variances, or one variance per component. Each names its quantity, since no validation can catch a variance passed where a scale belongs. Direct construction from `scale_tril` remains the primary path.

```python
import jax
import jax.numpy as jnp
from jax_pdf import GaussianMixture

# Two equally weighted modes in 32 dimensions, isotropic with variance 1/16
d = 32
gm = GaussianMixture.from_isotropic_variance(
    weights=[0.5, 0.5],
    means=[jnp.zeros(d), jnp.full(d, 0.75)],
    variance=1 / 16,
)

print(gm.dim, gm.n_components)        # 32 2
print(gm.log_normalization())         # 0.0, the weights sum to one

# Correlated modes: pass covariances and they are factorized once
rotation = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(0), (3, 3)))[0]
scales = jnp.diag(jnp.array([1.0, 0.2, 0.05]))
gm3 = GaussianMixture.from_covariance(
    weights=[0.6, 0.4],
    means=jnp.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    covariances=jnp.stack([rotation @ scales @ rotation.T] * 2),
)

# One variance per component, or one per coordinate per component
GaussianMixture.from_isotropic_variance([0.5, 0.5], gm3.means, jnp.array([0.1, 1.0]))
GaussianMixture.from_diagonal_variances([0.5, 0.5], gm3.means, jnp.array([[0.1, 1.0, 0.5]] * 2))
```

Scoring a sampler's mode weights against the truth takes one call, and the labelled draws check the attribution itself:

```python
samples, truth = gm.sample_with_component(jax.random.PRNGKey(1), 10_000)
attributed = jnp.argmax(gm.log_responsibilities(samples), axis=-1)
print(jnp.mean(attributed == truth))     # 1.0: the modes are disjoint at d = 32

# the same attribution on another sampler's draws gives its mode weights, here a
# stand-in model that puts nine tenths of its mass on the first mode
model = GaussianMixture.from_isotropic_variance([0.9, 0.1], gm.means, 1 / 16)
model_draws = model.sample(jax.random.PRNGKey(2), 10_000)
print(jnp.mean(jnp.argmax(gm.log_responsibilities(model_draws), -1) == 1))   # about 0.1
```

## References

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, chapter 9. Springer.
- Midgley, L. I., Stimper, V., Simm, G. N. C., Scholkopf, B. and Hernandez-Lobato, J. M. (2023). Flow annealed importance sampling bootstrap. *International Conference on Learning Representations*. Uses a 40-component mixture in two dimensions as its main illustrative target.
