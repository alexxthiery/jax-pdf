"""Phi-four lattice field theory on a periodic lattice of any dimension."""

import math

import jax.numpy as jnp
from flax import struct
from jax import Array

from jax_pdf._validation import check_event_shape, is_concrete


@struct.dataclass
class LatticePhiFour:
    """Phi-four field on a periodic lattice, in the action's own parameters.

    The log-density is

        log p(x) = -S(x)

        S(x) = u * sum_v (x_v^2 - a^2)^2
               + (kappa / 2) * sum_<v,w> (x_v - x_w)^2
               - h * sum_v x_v

    where v runs over lattice sites and <v, w> over nearest-neighbour bonds,
    each counted once. The quartic term puts a double well at x_v = +a and
    x_v = -a on every site, the coupling penalizes variation between
    neighbours, so the field tends to be near +a or near -a across the whole
    lattice, and h tilts the two.

    Samplers must flip the entire field between those two states, which no
    local move does. At h = 0 the density is invariant under x -> -x, so the
    two states carry exactly equal mass, which is an exact check available to
    any sampler. Nonzero h breaks that symmetry and makes the masses unequal.

    The lattice is periodic along every axis. A side of length 2 therefore
    carries two bonds between its two sites, one each way around the ring.

    `PhiFour` is the same family restricted to a 1D chain with the wells
    pinned at +-1 and the quartic tied to the coupling. This class takes the
    action's parameters directly and lattices of any dimension, so a 2D
    lattice field theory is `lattice_shape=(L, L)`.

    Attributes:
        u: Quartic strength. Larger deepens the wells. Default: 1.0.
        a: Well location: the minima of the local term are at +-a.
            Default: 1.0.
        kappa: Nearest-neighbour coupling. Larger favours smoother fields
            and raises the cost of a domain wall. Default: 1.0.
        h: External field. Zero leaves the sign symmetry intact; nonzero
            tilts toward one sign. Default: 0.0.
        lattice_shape: Sides of the periodic lattice, e.g. (16, 16) for a
            2D lattice. Static: it sets the input shape. Default: (16, 16).
    """

    u: float = 1.0
    """Quartic strength. Larger = deeper wells."""

    a: float = 1.0
    """Well location: local minima at +-a."""

    kappa: float = 1.0
    """Nearest-neighbour coupling. Larger = smoother fields."""

    h: float = 0.0
    """External field. Nonzero breaks the sign symmetry."""

    lattice_shape: tuple = struct.field(pytree_node=False, default=(16, 16))
    """Sides of the periodic lattice. Static: it sets the input shape."""

    def __post_init__(self):
        if len(self.lattice_shape) == 0:
            raise ValueError("lattice_shape must have at least one axis, got ()")
        if any(side < 2 for side in self.lattice_shape):
            raise ValueError(
                f"every lattice side must be >= 2, got {self.lattice_shape}"
            )
        if is_concrete(self.u) and self.u <= 0:
            raise ValueError(f"u must be positive, got {self.u}")
        if is_concrete(self.a) and self.a <= 0:
            raise ValueError(f"a must be positive, got {self.a}")
        if is_concrete(self.kappa) and self.kappa <= 0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")

    @property
    def dim(self) -> int:
        """Number of lattice sites, which is the dimension of the input."""
        return math.prod(self.lattice_shape)

    def __call__(self, x: Array) -> Array:
        """Evaluate unnormalized log-density.

        Args:
            x: Field(s) of shape (..., dim), flattened in C order over
                `lattice_shape`.

        Returns:
            Unnormalized log-density of shape (...).

        Raises:
            ValueError: If the input does not have trailing shape (dim,).
        """
        check_event_shape(x, (self.dim,))
        shape = jnp.shape(x)
        field = x.reshape(shape[:-1] + tuple(self.lattice_shape))
        axes = tuple(range(-len(self.lattice_shape), 0))

        # Pairing every site with its successor along an axis walks each bond
        # of the periodic lattice exactly once.
        coupling = sum(jnp.sum((field - jnp.roll(field, -1, axis=ax)) ** 2, axis=axes)
                       for ax in axes)
        quartic = jnp.sum((field**2 - self.a**2) ** 2, axis=axes)
        magnetization = jnp.sum(field, axis=axes)

        return -(self.u * quartic + self.kappa / 2 * coupling - self.h * magnetization)

    def log_normalization(self) -> Array:
        """Not available: the normalizing constant is intractable.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "LatticePhiFour normalizing constant is intractable."
        )
