"""Internal validation of shape metadata and concrete parameter values."""

import jax


def check_event_shape(x: jax.Array, event_shape: tuple[int, ...]) -> None:
    """Require the trailing axes of ``x`` to match one distribution event.

    Leading batch axes are unrestricted, including axes of length zero.
    Comparing shape metadata uses no array operations: with ordinary JIT,
    validation runs during tracing and adds no work to the compiled function.
    Unlike numeric parameter validation, this check must not be skipped when
    ``x`` is a tracer. Event sizes must be static; symbolic batch sizes are
    allowed, but a symbolic event size must be provably equal to the expected
    size for export to succeed.

    Args:
        x: Input array of shape ``batch_shape + event_shape``.
        event_shape: Nonempty tuple of expected trailing dimensions, e.g.
            ``(dim,)`` or ``(n_particles, spatial_dim)``.

    Raises:
        ValueError: If ``x`` has too few axes or its trailing axes do not match.
    """
    if x.shape[-len(event_shape):] != event_shape:
        raise ValueError(
            f"Expected x.shape[-{len(event_shape)}:] == {event_shape}, "
            f"got shape {tuple(x.shape)}."
        )


def is_concrete(value) -> bool:
    """Whether a parameter holds a value that Python can compare.

    Numeric parameters are pytree children, so a distribution that crosses a
    ``jit`` or ``vmap`` boundary is rebuilt by ``tree_unflatten`` with tracers
    in their place, and ``__post_init__`` runs again on them. Comparing a
    tracer cannot produce a Python bool, so a validation guarded by this
    function checks concrete values and is skipped under tracing. JAX also
    probes pytree structure using plain ``object()`` leaves before mapping;
    these placeholders have no numeric value to validate. Test the exact type,
    since every Python value is an instance of object.

    Structural parameters (sizes, flags, modes) are static instead, so they
    are always concrete and need no guard, and they may drive Python control
    flow inside ``__call__``.

    Args:
        value: A parameter value.

    Returns:
        True for ordinary parameter values, False for JAX tracers and plain
        object placeholders used during pytree reconstruction.
    """
    return type(value) is not object and not isinstance(value, jax.core.Tracer)
