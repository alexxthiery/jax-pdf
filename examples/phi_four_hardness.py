"""Map where a phi-four chain is genuinely two-moded, and how hard it is.

In one dimension a chain with nearest-neighbour coupling has no symmetry
breaking in the long-chain limit: a domain wall costs a finite amount, so a
chain much longer than the correlation length xi breaks into domains and its
magnetization concentrates near zero. Whether a chain is a two-moded target is
therefore decided by n_sites against xi, not by the barrier alone, and a study
that picks parameters by the barrier can end up with a unimodal target.

This sweep reports, for each (u, kappa, n_sites) at h = 0 and a = 1:

  xi           the correlation length, exactly, from the transfer spectrum
  n/xi         how many correlation lengths the chain spans
  barrier      n u a^4, the uniform-path barrier, exactly
  sd(m)        the spread of the magnetization, from exact draws
  P(|m| > 1/2) how much of the mass sits away from the symmetric point
  binder       1 - <m^4> / (3 <m^2>^2): 2/3 for two sharp phases, 0 for a
               Gaussian around zero

Everything but the last three is exact. Those three come from exact draws of
the discretized chain, so they carry Monte Carlo error, reported alongside.

Run: python examples/phi_four_hardness.py
"""

import numpy as np

from jax_pdf.phi_four_oracle import PhiFourChainOracle

N_SAMPLES = 4000
N_GRID, BOUND = 301, 3.0
SETTINGS = [(u, kappa, n) for u in (0.5, 1.0, 2.0)
            for kappa in (1.0, 2.0, 4.0, 8.0)
            for n in (16, 32, 64)]


def survey(u, kappa, n_sites, rng):
    oracle = PhiFourChainOracle(u=u, a=1.0, kappa=kappa, h=0.0, n_sites=n_sites,
                                periodic=True, n_grid=N_GRID, bound=BOUND)
    xi = oracle.correlation_length()
    m = oracle.sample(rng, N_SAMPLES).mean(axis=1)

    second, fourth = np.mean(m**2), np.mean(m**4)
    binder = 1 - fourth / (3 * second**2)
    ordered = np.mean(np.abs(m) > 0.5)
    return {
        "xi": xi,
        "spanned": n_sites / xi,
        "barrier": n_sites * u,
        "sd_m": float(np.sqrt(second)),
        "ordered": ordered,
        "ordered_se": float(np.sqrt(ordered * (1 - ordered) / N_SAMPLES)),
        "binder": float(binder),
    }


def figure(rows, rng):
    """Magnetization histograms for three regimes, and the Binder collapse."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (u, kappa, n_sites), colour, label in [((0.5, 1.0, 64), "#2a78d6", "many domains"),
                                               ((1.0, 4.0, 64), "#eda100", "marginal"),
                                               ((2.0, 8.0, 64), "#eb6834", "one domain")]:
        oracle = PhiFourChainOracle(u=u, a=1.0, kappa=kappa, h=0.0, n_sites=n_sites,
                                    periodic=True, n_grid=N_GRID, bound=BOUND)
        m = oracle.sample(rng, N_SAMPLES).mean(axis=1)
        axes[0].hist(m, bins=80, density=True, histtype="step", lw=1.6, color=colour,
                     label=f"{label}: u={u}, kappa={kappa}, n/xi={n_sites / oracle.correlation_length():.1f}")
    axes[0].set(xlabel="magnetization m", ylabel="density",
                title=f"Magnetization of a {64}-site chain")
    axes[0].legend(fontsize=8)

    spanned = [row["spanned"] for _, row in rows]
    binder = [row["binder"] for _, row in rows]
    axes[1].scatter(spanned, binder, s=18, color="#2a78d6")
    axes[1].axhline(2 / 3, color="#0b0b0b", lw=0.8, ls="--")
    axes[1].set(xscale="log", xlabel="n / xi", ylabel="Binder cumulant",
                title="Order is set by n / xi, not by the barrier (dashed: 2/3)")
    fig.tight_layout()
    fig.savefig("examples/phi_four_hardness.png", dpi=200)
    print("\nSaved examples/phi_four_hardness.png")


def main():
    rng = np.random.default_rng(0)
    print(f"{'u':>5s}{'kappa':>7s}{'n':>5s}{'xi':>9s}{'n/xi':>8s}{'barrier':>9s}"
          f"{'sd(m)':>8s}{'P(|m|>1/2)':>13s}{'binder':>9s}")
    rows = []
    for u, kappa, n_sites in SETTINGS:
        row = survey(u, kappa, n_sites, rng)
        rows.append(((u, kappa, n_sites), row))
        print(f"{u:5.1f}{kappa:7.1f}{n_sites:5d}{row['xi']:9.2f}{row['spanned']:8.2f}"
              f"{row['barrier']:9.1f}{row['sd_m']:8.3f}"
              f"{row['ordered']:8.3f} +-{row['ordered_se']:.3f}{row['binder']:9.3f}")

    # A usable stress-test target is two-moded: most of the mass away from the
    # symmetric point, and a Binder cumulant near 2/3. Chain length is usually
    # fixed by other considerations, so report the weakest coupling that
    # orders each length rather than one global winner.
    print()
    for n_sites in sorted({n for _, _, n in SETTINGS}):
        ordered = [(key, row) for key, row in rows
                   if key[2] == n_sites and row["ordered"] > 0.9 and row["binder"] > 0.5]
        if not ordered:
            print(f"n = {n_sites:3d}: nothing in this sweep orders it")
            continue
        (u, kappa, _), row = min(ordered, key=lambda item: (item[0][1], item[0][0]))
        print(f"n = {n_sites:3d}: u = {u}, kappa = {kappa} (a = 1, h = 0) gives xi = {row['xi']:.1f}, "
              f"n/xi = {row['spanned']:.2f}, P(|m| > 1/2) = {row['ordered']:.3f}, "
              f"Binder = {row['binder']:.3f}")

    # The barrier is the quantity a reader reaches for, and it is the wrong one.
    by_key = dict(rows)
    deep = by_key[(2.0, 1.0, 64)]
    ordered = by_key[(1.0, 8.0, 64)]
    print(f"\nThe barrier does not decide it. On 64 sites, u = 2, kappa = 1 has a "
          f"uniform-path barrier of {deep['barrier']:.0f} nats and a Binder cumulant of "
          f"{deep['binder']:.3f} (one mode), while u = 1, kappa = 8 has half that barrier, "
          f"{ordered['barrier']:.0f} nats, and a Binder cumulant of {ordered['binder']:.3f} "
          f"(two modes). The chain that spans fewer correlation lengths is the two-moded one.")

    figure(rows, rng)


if __name__ == "__main__":
    main()
