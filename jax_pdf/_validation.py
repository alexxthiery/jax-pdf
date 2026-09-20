"""Parameter validation that survives JAX tracing (internal)."""

import jax


def is_concrete(value) -> bool:
    """Whether a parameter holds a value that Python can compare.

    Numeric parameters are pytree children, so a distribution that crosses a
    ``jit`` or ``vmap`` boundary is rebuilt by ``tree_unflatten`` with tracers
    in their place, and ``__post_init__`` runs again on them. Comparing a
    tracer cannot produce a Python bool, so a validation guarded by this
    function checks concrete values and is skipped under tracing.

    Structural parameters (sizes, flags, modes) are static instead, so they
    are always concrete and need no guard, and they may drive Python control
    flow inside ``__call__``.

    Args:
        value: A parameter value.

    Returns:
        True for an ordinary Python or NumPy value, False for a JAX tracer.
    """
    return not isinstance(value, jax.core.Tracer)
