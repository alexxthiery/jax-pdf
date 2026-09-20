"""Reference values for a one-dimensional phi-four chain, by transfer operator.

NumPy, float64, and deliberately outside the distribution interface: the chain
is solved on a quadrature grid, so every number here is a controlled
approximation rather than a closed form, and `log_normalization` on the
distributions keeps its "exact or raise" contract.
"""

import numpy as np

from jax_pdf.lattice_phi_four import LatticePhiFour
from jax_pdf.phi_four import PhiFour

_MAX_POWER_BYTES = 2e9


class PhiFourChainOracle:
    """Ground truth for a phi-four chain on a quadrature grid.

    The chain has sites i = 0, ..., n_sites - 1 with

        S(x) = u sum_i (x_i^2 - a^2)^2
               + (kappa / 2) sum_bonds (x_i - x_j)^2
               - h sum_i x_i

    and either periodic bonds (a ring, n_sites bonds) or Dirichlet ends (the
    field is pinned to zero just outside each end, n_sites + 1 bonds). This is
    `LatticePhiFour` with a one-axis lattice, and `PhiFour` under the mapping
    in its documentation.

    Because the coupling is nearest-neighbour, the chain is a Markov chain in
    the site index, so discretizing the field on a grid turns every quantity
    into linear algebra on a transfer matrix: no Monte Carlo anywhere. The
    sampler is exact for the discretized chain, with no burn-in and no
    rejection.

    What is approximate: the field is represented on `n_grid` points spanning
    `[-bound, bound]` with the trapezoid rule. Away from the tails and with a
    fine enough grid the error is far below anything a sampler study needs;
    `discretization_report` measures it rather than assuming it.

    Attributes:
        u, a, kappa, h: Action parameters, as in `LatticePhiFour`. `u = 0` is
            allowed here, which makes the chain Gaussian and gives a closed
            form to check against.
        n_sites: Number of sites.
        periodic: Ring (True) or Dirichlet ends (False).
        n_grid: Number of quadrature points for the field at one site.
        bound: The grid spans [-bound, bound].
    """

    def __init__(self, u: float = 1.0, a: float = 1.0, kappa: float = 1.0, h: float = 0.0,
                 n_sites: int = 32, periodic: bool = True,
                 n_grid: int = 2001, bound: float = 3.0):
        """Build the transfer operator.

        Raises:
            ValueError: on a negative u, a non-positive a or kappa, fewer than
                two sites, a grid of fewer than three points, or a
                non-positive bound.
        """
        if u < 0:
            raise ValueError(f"u must be non-negative, got {u}")
        if a <= 0:
            raise ValueError(f"a must be positive, got {a}")
        if kappa <= 0:
            raise ValueError(f"kappa must be positive, got {kappa}")
        if n_sites < 2:
            raise ValueError(f"n_sites must be >= 2, got {n_sites}")
        if n_grid < 3:
            raise ValueError(f"n_grid must be >= 3, got {n_grid}")
        if bound <= 0:
            raise ValueError(f"bound must be positive, got {bound}")

        self.u, self.a, self.kappa, self.h = float(u), float(a), float(kappa), float(h)
        self.n_sites, self.periodic = int(n_sites), bool(periodic)
        self.n_grid, self.bound = int(n_grid), float(bound)

        self._grid = np.linspace(-self.bound, self.bound, self.n_grid)
        self._spacing = float(self._grid[1] - self._grid[0])
        self._weights = np.full(self.n_grid, self._spacing)
        self._weights[0] = self._weights[-1] = self._spacing / 2        # trapezoid rule

        # The local factor is shifted to its maximum before exponentiating, so
        # a deep well cannot overflow; the shift returns once per site in log Z.
        log_local = -self.u * (self._grid**2 - self.a**2) ** 2 + self.h * self._grid
        self._shift = float(log_local.max())
        self._site = self._weights * np.exp(log_local - self._shift)
        self._bond = np.exp(-self.kappa / 2 * (self._grid[:, None] - self._grid[None, :]) ** 2)
        self._end = np.exp(-self.kappa / 2 * self._grid**2)             # bond to a pinned end

        # On a ring every site meets exactly two bonds, so splitting its factor
        # between them makes the transfer matrix symmetric.
        self._ring = np.sqrt(self._site)[:, None] * self._bond * np.sqrt(self._site)[None, :]

        self._eigen = None
        self._powers = None
        self._messages = None

    @classmethod
    def from_distribution(cls, dist, **kwargs) -> "PhiFourChainOracle":
        """Build the oracle for a `PhiFour` or one-dimensional `LatticePhiFour`.

        Args:
            dist: The distribution to mirror. `PhiFour` is mapped through
                kappa = beta c, u = beta / (4c), a = 1, h = -beta b / c, with
                c = a * dim_grid.
            **kwargs: Passed on (`n_grid`, `bound`).

        Returns:
            An oracle for the same chain.

        Raises:
            TypeError: for any other distribution, or a lattice with more than
                one axis.
        """
        if isinstance(dist, LatticePhiFour):
            if len(dist.lattice_shape) != 1:
                raise TypeError(
                    "the chain oracle solves one-dimensional lattices, got "
                    f"lattice_shape {dist.lattice_shape}"
                )
            return cls(u=float(dist.u), a=float(dist.a), kappa=float(dist.kappa),
                       h=float(dist.h), n_sites=dist.lattice_shape[0], periodic=True, **kwargs)
        if isinstance(dist, PhiFour):
            c = float(dist.a) * dist.dim_grid
            return cls(u=float(dist.beta) / (4 * c), a=1.0, kappa=float(dist.beta) * c,
                       h=-float(dist.beta) * float(dist.b) / c, n_sites=dist.dim_grid,
                       periodic=dist.periodic, **kwargs)
        raise TypeError(
            f"expected PhiFour or a one-dimensional LatticePhiFour, got {type(dist).__name__}"
        )

    @property
    def grid(self) -> np.ndarray:
        """The quadrature points, shape (n_grid,)."""
        return self._grid

    def log_partition(self) -> float:
        """log Z of the chain, with Z the integral of exp(-S) over all fields."""
        if self.periodic:
            values = np.clip(np.linalg.eigvalsh(self._ring), 0.0, None)
            top = values.max()
            total = self.n_sites * np.log(top) + np.log(np.sum((values / top) ** self.n_sites))
        else:
            forward, scales = self._forward_messages()
            total = np.log(np.sum(forward[-1] * self._end)) + scales[-1]
        return float(total + self.n_sites * self._shift)

    def site_marginal(self, site: int = 0) -> np.ndarray:
        """Marginal density of one site, evaluated on the grid.

        Args:
            site: Which site. On a ring every site has the same marginal; with
                Dirichlet ends they differ.

        Returns:
            Density on the grid, shape (n_grid,), integrating to 1 under the
            trapezoid rule.
        """
        mass = self._site_mass(site)
        return mass / (self._weights * mass.sum())

    def mean_field(self, site: int = 0) -> float:
        """E[x_i] at one site. Exactly zero at h = 0, by the sign symmetry."""
        mass = self._site_mass(site)
        return float(np.sum(self._grid * mass) / mass.sum())

    def two_point(self, separation: int, site: int = 0) -> float:
        """E[x_i x_j] for sites i and j a given number of bonds apart.

        Raises:
            ValueError: if the separation leaves the chain.
        """
        self._check_site(site)
        if separation < 1:
            raise ValueError(f"separation must be >= 1, got {separation}")
        if self.periodic and separation >= self.n_sites:
            raise ValueError(
                f"separation must be < n_sites = {self.n_sites}, got {separation}"
            )
        if not self.periodic and site + separation > self.n_sites - 1:
            raise ValueError(
                f"site {site} plus separation {separation} leaves a chain of "
                f"{self.n_sites} sites"
            )

        if self.periodic:
            values, vectors = self._eigendecomposition()
            ratio = values / values.max()
            near = (vectors * ratio**separation) @ vectors.T
            far = (vectors * ratio ** (self.n_sites - separation)) @ vectors.T
            joint = near * far
            return float(self._grid @ joint @ self._grid / joint.sum())

        forward, _ = self._forward_messages()
        backward, _ = self._backward_messages()
        left, right = forward[site], backward[site + separation]
        walked = right.copy()
        for _ in range(separation):                      # apply C = B diag(site)
            walked = self._bond @ (self._site * walked)
        moment = self._bond @ (self._site * (self._grid * right))
        for _ in range(separation - 1):
            moment = self._bond @ (self._site * moment)
        return float(np.sum(self._grid * left * moment) / np.sum(left * walked))

    def correlation_length(self) -> float:
        """Bulk correlation length of the chain, in lattice spacings.

        The transfer operator's two largest eigenvalues give
        xi = 1 / log(lambda_1 / lambda_2), and connected correlations decay as
        exp(-r / xi). It is a property of the bulk, so it does not depend on
        the boundary condition, and on a chain shorter than a few xi the
        measured decay differs from it by finite-size effects.

        A chain of n_sites much longer than xi breaks into domains and its
        magnetization concentrates near zero; a chain shorter than xi behaves
        as one domain and its magnetization is bimodal. In one dimension there
        is no symmetry breaking in the long-chain limit, so this ratio, not the
        barrier alone, decides whether the target is genuinely two-moded.

        Returns:
            The correlation length, or inf if the top two eigenvalues coincide.
        """
        values = np.sort(np.clip(np.linalg.eigvalsh(self._ring), 0.0, None))[::-1]
        if values[1] <= 0:
            return float("inf")
        gap = np.log(values[0] / values[1])
        return float("inf") if gap <= 0 else float(1.0 / gap)

    def sample(self, rng, n: int) -> np.ndarray:
        """Draw fields from the discretized chain, exactly.

        Sites are drawn one at a time from their exact conditionals given the
        sites already drawn, so the samples are exact for the discretized
        measure, with no Metropolis step and no burn-in. The value at a site is
        a grid point jittered inside its cell, so the samples are continuous.

        Uniforms are taken as `rng.random((n, n_sites))` for the site choices,
        then `rng.random((n, n_sites))` for the jitter.

        Args:
            rng: A NumPy Generator.
            n: How many fields.

        Returns:
            Fields of shape (n, n_sites).

        Raises:
            MemoryError: if the transfer powers a ring needs would exceed about
                two gigabytes. Lower `n_grid`.
        """
        choices = rng.random((n, self.n_sites))
        jitter = rng.random((n, self.n_sites))

        cells = np.empty((n, self.n_sites), dtype=int)
        cells[:, 0] = np.searchsorted(np.cumsum(self._first_site_probabilities()), choices[:, 0])
        cells[:, 0] = np.clip(cells[:, 0], 0, self.n_grid - 1)
        for site in range(1, self.n_sites):
            table = self._conditional(site, cells[:, 0], cells[:, site - 1])
            drawn = (np.cumsum(table, axis=1) < choices[:, [site]]).sum(axis=1)
            cells[:, site] = np.clip(drawn, 0, self.n_grid - 1)

        values = self._grid[cells] + (jitter - 0.5) * self._spacing
        return np.clip(values, -self.bound, self.bound)

    def discretization_report(self) -> dict:
        """Measure the discretization error instead of assuming it.

        Returns:
            A dict with `log_partition`, the same quantity on a grid of twice
            the resolution (`log_partition_fine`) and on a grid of twice the
            extent (`log_partition_wide`), their differences, and
            `tail_density`, the site marginal's value at the boundary relative
            to its maximum. Small differences and a tiny tail density mean the
            grid is adequate.
        """
        base = self.log_partition()
        fine = self._rebuilt(n_grid=2 * self.n_grid - 1, bound=self.bound).log_partition()
        wide = self._rebuilt(n_grid=4 * (self.n_grid - 1) + 1,
                             bound=2 * self.bound).log_partition()
        marginal = self.site_marginal()
        return {
            "log_partition": base,
            "log_partition_fine": fine,
            "refinement_change": fine - base,
            "log_partition_wide": wide,
            "extent_change": wide - base,
            "tail_density": float(marginal[0] / marginal.max()),
        }

    # Internals. The sampler's conditionals are exposed to tests, which check
    # that they multiply out to the chain's exact discretized law.

    def _rebuilt(self, **changes) -> "PhiFourChainOracle":
        settings = dict(u=self.u, a=self.a, kappa=self.kappa, h=self.h, n_sites=self.n_sites,
                        periodic=self.periodic, n_grid=self.n_grid, bound=self.bound)
        settings.update(changes)
        return PhiFourChainOracle(**settings)

    def _check_site(self, site):
        if not 0 <= site < self.n_sites:
            raise ValueError(f"site must be in [0, {self.n_sites}), got {site}")

    def _eigendecomposition(self):
        if self._eigen is None:
            values, vectors = np.linalg.eigh(self._ring)
            self._eigen = (np.clip(values, 0.0, None), vectors)
        return self._eigen

    def _forward_messages(self):
        """f_i(j): weight of sites 0..i with site i at grid point j (Dirichlet).

        Each message is divided by its maximum, and the logarithms of those
        divisors accumulate in `scales`, so a long chain cannot overflow.
        """
        if self._messages is None or "forward" not in self._messages:
            message = self._site * self._end
            scale = float(np.log(message.max()))
            message = message / message.max()
            messages, scales = [message], [scale]
            for _ in range(1, self.n_sites):
                message = self._site * (self._bond @ message)
                top = message.max()
                message, scale = message / top, scale + float(np.log(top))
                messages.append(message)
                scales.append(scale)
            self._messages = dict(self._messages or {},
                                  forward=(np.array(messages), np.array(scales)))
        return self._messages["forward"]

    def _backward_messages(self):
        """b_i(j): weight of sites i+1..N-1 given site i at grid point j (Dirichlet)."""
        if self._messages is None or "backward" not in self._messages:
            messages = [None] * self.n_sites
            messages[-1] = self._end.copy()
            scales = [0.0] * self.n_sites
            for site in range(self.n_sites - 2, -1, -1):
                message = self._bond @ (self._site * messages[site + 1])
                top = message.max()
                messages[site] = message / top
                scales[site] = scales[site + 1] + float(np.log(top))
            self._messages = dict(self._messages or {},
                                  backward=(np.array(messages), np.array(scales)))
        return self._messages["backward"]

    def _site_mass(self, site=0):
        """Unnormalized probability of each grid cell at one site."""
        self._check_site(site)
        if self.periodic:
            values, vectors = self._eigendecomposition()
            ratio = values / values.max()
            return (vectors**2) @ ratio**self.n_sites
        forward, _ = self._forward_messages()
        backward, _ = self._backward_messages()
        return forward[site] * backward[site]

    def _transfer_powers(self):
        """A^m for m = 1 .. n_sites - 1, which a ring's conditionals need."""
        if self._powers is None:
            needed = (self.n_sites - 1) * self.n_grid**2 * 8
            if needed > _MAX_POWER_BYTES:
                raise MemoryError(
                    f"sampling a ring of {self.n_sites} sites would cache "
                    f"{needed / 1e9:.1f} GB of transfer powers; lower n_grid "
                    f"(currently {self.n_grid})"
                )
            powers = [None, self._ring / self._ring.max()]
            for _ in range(2, self.n_sites):
                nxt = powers[-1] @ powers[1]
                powers.append(nxt / nxt.max())
            self._powers = powers
        return self._powers

    def _first_site_probabilities(self):
        """The law of the first site drawn, over grid cells."""
        mass = self._site_mass(0)
        return mass / mass.sum()

    def _conditional(self, site, first, previous):
        """p(x_site | x_{site-1}, x_0) over grid cells, shape (batch, n_grid).

        On a ring the conditional depends on the first site as well, because
        the chain has to close on it; with Dirichlet ends it does not.
        """
        if self.periodic:
            powers = self._transfer_powers()
            closing = powers[self.n_sites - site][:, first].T
            table = self._ring[previous] * closing
        else:
            backward, _ = self._backward_messages()
            table = self._bond[previous] * (self._site * backward[site])[None, :]
        return table / table.sum(axis=1, keepdims=True)
