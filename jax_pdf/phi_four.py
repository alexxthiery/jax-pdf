"""Phi-four lattice field theory as a 1D distribution."""

import jax.numpy as jnp
from flax import struct
from jax import Array


@struct.dataclass
class PhiFour:
    """Phi-four lattice field theory on a 1D lattice.

    The log-density is:
        log p(phi) = -beta * U(phi)

    where the energy combines nearest-neighbor coupling with a local
    double-well potential:
        U(phi) = c * sum_i (phi_{i+1} - phi_i)^2 / 2
                 + (1/c) * sum_i [(1 - phi_i^2)^2 / 4 + b * phi_i]

    with c = a * dim_grid.

    The double-well potential creates two energy minima (phi near +1 and -1).
    Nearest-neighbor coupling penalizes spatial variation, favoring smooth
    fields. When b = 0 the distribution is Z2-symmetric (invariant under
    phi -> -phi); nonzero b breaks this symmetry.

    Attributes:
        a: Coupling constant. Smaller values produce stronger correlation
            between neighbors. Default: 0.1.
        b: External field bias. Breaks Z2 symmetry when nonzero.
            Default: 0.0.
        dim_grid: Number of lattice sites (= distribution dimension).
            Default: 100.
        beta: Inverse temperature. Higher values sharpen the modes.
            Default: 1.0.
        periodic: Boundary conditions. False = Dirichlet (phi_0 = phi_{N+1} = 0),
            True = periodic (phi wraps around). Default: False.
    """

    a: float = 0.1
    """Coupling constant. Smaller = stronger neighbor correlation."""

    b: float = 0.0
    """External field bias. Breaks Z2 symmetry when nonzero."""

    dim_grid: int = 100
    """Number of lattice sites (= distribution dimension)."""

    beta: float = 1.0
    """Inverse temperature. Higher = sharper modes, harder sampling."""

    periodic: bool = False
    """False = Dirichlet BCs (zero boundary), True = periodic BCs."""

    def __post_init__(self):
        if self.a <= 0:
            raise ValueError(f"a must be positive, got {self.a}")
        if self.dim_grid < 2:
            raise ValueError(f"dim_grid must be >= 2, got {self.dim_grid}")
        if self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}")

    @property
    def dim(self) -> int:
        return self.dim_grid

    def __call__(self, x: Array) -> Array:
        """Evaluate unnormalized log-density.

        Args:
            x: Input point(s) of shape (..., dim_grid).

        Returns:
            Unnormalized log-density of shape (...).
        """
        coef = self.a * self.dim_grid

        # Pad for finite differences depending on boundary conditions
        if self.periodic:
            # Pad left only to avoid double-counting the wrap-around edge
            padded = jnp.concatenate([x[..., -1:], x], axis=-1)
        else:
            pad_width = ((0, 0),) * (x.ndim - 1) + ((1, 1),)
            padded = jnp.pad(x, pad_width)

        # Nearest-neighbor coupling: squared finite differences
        diffs = padded[..., 1:] - padded[..., :-1]
        grad_term = jnp.sum(diffs**2 / 2, axis=-1)

        # Local double-well potential
        potential = jnp.sum((1 - x**2) ** 2 / 4 + self.b * x, axis=-1)

        energy = coef * grad_term + potential / coef
        return -self.beta * energy

    def log_normalization(self) -> Array:
        """Not available: normalizing constant is intractable.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "PhiFour normalizing constant is intractable."
        )
