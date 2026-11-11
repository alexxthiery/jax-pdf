"""Shared free-cluster geometry and confinement; callers validate event shapes."""

import jax.numpy as jnp


def pair_distances(positions):
    """Unique i<j distances, shape (..., N*(N-1)//2), with the cluster norm floor.

    The 1e-8 squared-distance offset preserves the DW4/free-LJ convention.
    Periodic and molecular potentials have different regularization contracts.
    """
    difference = positions[..., :, None, :] - positions[..., None, :, :]
    distances = jnp.sqrt(jnp.sum(difference**2, axis=-1) + 1e-8)
    i, j = jnp.triu_indices(positions.shape[-2], k=1)
    return distances[..., i, j]


def harmonic_trap(positions, strength, mode):
    """Individual or COM-only harmonic energy, shape (...); mode is prevalidated."""
    if mode == "individual":
        return 0.5*strength*jnp.sum(positions**2, axis=(-2, -1))
    com = positions.mean(axis=-2)
    return 0.5*strength*positions.shape[-2]*jnp.sum(com**2, axis=-1)
