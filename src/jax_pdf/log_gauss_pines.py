"""Log Gaussian Cox Process on the Finnish Pines dataset."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from jax import Array

from jax_pdf import cox_process_utils as cp_utils

jax.config.update("jax_enable_x64", True)  # needed for LBFGS optimization

# Load pines data at module level (shared across instances)
_MODULE_PATH = os.path.dirname(__file__)
_PINES_PATH = os.path.join(_MODULE_PATH, "finpines.csv")
_pines_data = None


def _load_pines_data() -> Array:
    """Load Finnish pines dataset (cached)."""
    global _pines_data
    if _pines_data is None:
        if not os.path.exists(_PINES_PATH):
            raise FileNotFoundError(f"Dataset not found at {_PINES_PATH}")
        data = np.genfromtxt(_PINES_PATH, delimiter=",")
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(f"Expected (N, 2) shape, got {data.shape}")
        _pines_data = jnp.array(data)
    return _pines_data


@struct.dataclass
class LGCP:
    """Log Gaussian Cox Process (LGCP) on the Finnish Pines dataset.

    The LGCP models spatial point patterns as a Poisson process with
    log-intensity given by a Gaussian process:

        f ~ GP(mu, K)
        n_i | f ~ Poisson(A * exp(f_i))

    where f is the latent log-intensity field on a discretized grid,
    K is an exponential covariance kernel, and n_i are observed counts.

    This is an unnormalized posterior density used as an MCMC benchmark.
    The challenging geometry arises from the GP prior correlation structure.

    Attributes:
        grid_dim: Number of grid cells per dimension. Total latent
            dimension is grid_dim^2. Larger = finer resolution but
            more expensive. Default: 40.
        whitened: If True, parameterize in whitened space where the
            prior is standard normal (easier geometry for HMC).
            Default: False.
    """

    grid_dim: int = 40
    """Grid cells per dimension. Total dim = grid_dim^2."""

    whitened: bool = False
    """If True, use whitened (decorrelated) parameterization."""

    # Computed fields (set in __post_init__)
    _bin_counts: Array = struct.field(default=None, pytree_node=True)
    _cholesky_cov: Array = struct.field(default=None, pytree_node=True)
    _mean: float = struct.field(default=None, pytree_node=False)
    _bin_area: float = struct.field(default=None, pytree_node=False)
    _log_norm: float = struct.field(default=None, pytree_node=False)

    def __post_init__(self):
        if self.grid_dim <= 0:
            raise ValueError(f"grid_dim must be positive, got {self.grid_dim}")

        # Skip computation if already initialized (e.g., from replace())
        if self._bin_counts is not None:
            return

        # Load data and compute bin counts
        pines = _load_pines_data()
        bin_counts_2d = cp_utils.compute_bin_counts(pines, self.grid_dim)
        bin_counts = jnp.reshape(jnp.array(bin_counts_2d), (-1,))

        # Build covariance matrix and Cholesky factor
        grid_indices = cp_utils.make_grid_indices(self.grid_dim)
        signal_variance = 1.91
        length_scale = 1.0 / 33

        def kernel(x, y):
            return cp_utils.exponential_kernel(
                x, y, signal_variance, self.grid_dim, length_scale
            )

        gram = cp_utils.compute_gram_matrix(kernel, grid_indices)
        cholesky_cov = jnp.linalg.cholesky(gram)

        # GP mean and Poisson scaling
        mean = jnp.log(126.0) - 0.5 * signal_variance
        bin_area = 1.0 / (self.grid_dim ** 2)

        # Prior normalization constant
        num_latents = self.grid_dim ** 2
        log2pi = jnp.log(2.0 * jnp.pi)
        if self.whitened:
            log_norm = -0.5 * num_latents * log2pi
        else:
            half_log_det = jnp.sum(jnp.log(jnp.abs(jnp.diag(cholesky_cov))))
            log_norm = -0.5 * num_latents * log2pi - half_log_det

        # Set computed fields (frozen dataclass requires object.__setattr__)
        object.__setattr__(self, "_bin_counts", bin_counts)
        object.__setattr__(self, "_cholesky_cov", cholesky_cov)
        object.__setattr__(self, "_mean", float(mean))
        object.__setattr__(self, "_bin_area", float(bin_area))
        object.__setattr__(self, "_log_norm", float(log_norm))

    @property
    def dim(self) -> int:
        """Dimensionality of the latent field (grid_dim^2)."""
        return self.grid_dim ** 2

    @property
    def pines_points(self) -> Array:
        """Finnish pines dataset coordinates, shape (n_points, 2)."""
        return _load_pines_data()

    def __call__(self, x: Array) -> Array:
        """Evaluate log probability density (unnormalized posterior).

        Args:
            x: Latent field values of shape (..., dim). If whitened=True,
                these are whitened variables; otherwise, direct GP values.
                Supports batch dimensions.

        Returns:
            Log probability density of shape (...).
        """
        if self.whitened:
            latents = cp_utils.whiten_to_latent(x, self._mean, self._cholesky_cov)
            prior = -0.5 * jnp.sum(x ** 2, axis=-1) + self._log_norm
        else:
            latents = x
            white = cp_utils.latent_to_whiten(x, self._mean, self._cholesky_cov)
            prior = -0.5 * jnp.sum(white ** 2, axis=-1) + self._log_norm

        likelihood = cp_utils.poisson_log_likelihood(
            latents, self._bin_area, self._bin_counts
        )
        return prior + likelihood

    def log_normalization(self) -> Array:
        """Log normalizing constant of the prior (not the full posterior).

        Returns:
            Log normalization of the GP prior component.

        Note:
            The full LGCP posterior is unnormalized (intractable Z).
            This returns only the prior's normalization for diagnostics.
        """
        return jnp.array(self._log_norm)

    def map_estimate(
        self,
        x0: Array | None = None,
        max_iter: int = 500,
        tol: float = 1e-8,
    ) -> dict:
        """Compute Maximum A Posteriori (MAP) estimate via L-BFGS.

        Args:
            x0: Initial guess of shape (dim,). Defaults to zeros.
            max_iter: Maximum optimization iterations.
            tol: Gradient norm convergence tolerance.

        Returns:
            Dict with keys:
                - "x": MAP estimate of shape (dim,)
                - "x_history": optimization trajectory (n_iters+1, dim)
                - "loss_history": loss at each step (n_iters+1,)
                - "grad_norm_history": gradient norm at each step (n_iters+1,)
                - "n_iters": number of iterations taken
                - "converged": whether tolerance was reached
        """
        if x0 is None:
            x0 = jnp.zeros(self.dim, dtype=jnp.float64)

        if x0.shape != (self.dim,):
            raise ValueError(f"x0 must have shape ({self.dim},), got {x0.shape}")

        def loss_fn(x):
            return -self(x)

        grad_fn = jax.grad(loss_fn)
        optimizer = optax.lbfgs()
        opt_state = optimizer.init(x0)

        # Pre-allocate history arrays (max_iter + 1 for initial state)
        x_history = jnp.zeros((max_iter + 1, self.dim))
        loss_history = jnp.zeros(max_iter + 1)
        grad_norm_history = jnp.zeros(max_iter + 1)

        def cond_fun(state):
            _, _, _, _, grad_norm, i, _, _, _ = state
            return (grad_norm > tol) & (i < max_iter)

        def body_fun(state):
            x, opt_state, loss, grad, grad_norm, i, x_hist, loss_hist, grad_norm_hist = state

            # Store current state in history
            x_hist = x_hist.at[i].set(x)
            loss_hist = loss_hist.at[i].set(loss)
            grad_norm_hist = grad_norm_hist.at[i].set(grad_norm)

            # Optimization step
            updates, opt_state = optimizer.update(
                grad, opt_state, params=x, value=loss, grad=grad, value_fn=loss_fn
            )
            x_new = optax.apply_updates(x, updates)
            loss_new = loss_fn(x_new)
            grad_new = grad_fn(x_new)
            grad_norm_new = jnp.linalg.norm(grad_new)

            return (x_new, opt_state, loss_new, grad_new, grad_norm_new, i + 1,
                    x_hist, loss_hist, grad_norm_hist)

        @jax.jit
        def run_lbfgs(x0, opt_state, x_hist, loss_hist, grad_norm_hist):
            loss0 = loss_fn(x0)
            grad0 = grad_fn(x0)
            grad_norm0 = jnp.linalg.norm(grad0)

            init_state = (x0, opt_state, loss0, grad0, grad_norm0, 0,
                          x_hist, loss_hist, grad_norm_hist)
            final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)

            x_final, _, loss_final, _, grad_norm_final, n_iters, x_hist, loss_hist, grad_norm_hist = final_state

            # Store final state
            x_hist = x_hist.at[n_iters].set(x_final)
            loss_hist = loss_hist.at[n_iters].set(loss_final)
            grad_norm_hist = grad_norm_hist.at[n_iters].set(grad_norm_final)

            return x_final, n_iters, grad_norm_final, x_hist, loss_hist, grad_norm_hist

        x_map, n_iters, final_grad_norm, x_hist, loss_hist, grad_norm_hist = run_lbfgs(
            x0, opt_state, x_history, loss_history, grad_norm_history
        )

        # Slice history to actual length (n_iters + 1 points: initial + n_iters steps)
        actual_len = n_iters + 1
        return {
            "x": x_map,
            "x_history": x_hist[:actual_len],
            "loss_history": loss_hist[:actual_len],
            "grad_norm_history": grad_norm_hist[:actual_len],
            "n_iters": int(n_iters),
            "converged": bool(final_grad_norm <= tol),
        }

    def hessian_at(self, x: Array) -> Array:
        """Compute Hessian of negative log-density at a point.

        Args:
            x: Point of shape (dim,) (e.g., MAP estimate).

        Returns:
            Hessian matrix of shape (dim, dim).
        """
        if x.shape != (self.dim,):
            raise ValueError(f"x must have shape ({self.dim},), got {x.shape}")

        def loss_fn(x_):
            return -self(x_)

        return jax.hessian(loss_fn)(x)

    def laplace_approximation(
        self,
        x0: Array | None = None,
        max_iter: int = 500,
        tol: float = 1e-8,
    ) -> dict:
        """Compute Laplace approximation (Gaussian at MAP).

        Args:
            x0: Initial guess for MAP optimization.
            max_iter: Maximum optimization iterations.
            tol: Gradient norm convergence tolerance.

        Returns:
            Dict with keys:
                - "mu": MAP estimate (mean)
                - "precision": Hessian at MAP
                - "cov": Inverse Hessian (covariance)
                - "optimization": dict with optimization trajectory
                    - "x_history": trajectory (n_iters+1, dim)
                    - "loss_history": losses (n_iters+1,)
                    - "grad_norm_history": grad norms (n_iters+1,)
                    - "n_iters": iterations taken
                    - "converged": whether tolerance reached
        """
        map_result = self.map_estimate(x0=x0, max_iter=max_iter, tol=tol)
        x_map = map_result["x"]
        hessian = self.hessian_at(x_map)

        if not jnp.all(jnp.linalg.eigvalsh(hessian) > 0):
            raise ValueError("Hessian not positive definite at MAP")

        return {
            "mu": x_map,
            "precision": hessian,
            "cov": jnp.linalg.inv(hessian),
            "optimization": {
                "x_history": map_result["x_history"],
                "loss_history": map_result["loss_history"],
                "grad_norm_history": map_result["grad_norm_history"],
                "n_iters": map_result["n_iters"],
                "converged": map_result["converged"],
            },
        }
