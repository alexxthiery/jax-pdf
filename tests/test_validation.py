"""Shape metadata checks must reject bad events without adding array work."""

import jax
import jax.numpy as jnp
import pytest

from jax_pdf import DoubleWell, _validation


@pytest.mark.parametrize("event_shape", [(1,), (10,), (4, 3)])
@pytest.mark.parametrize("batch_shape", [(), (5,), (2, 3), (0,), (2, 0)])
def test_accepts_exact_event_shape_with_arbitrary_batch_axes(event_shape, batch_shape):
    # Abstract inputs make accessing array values impossible.
    x = jax.ShapeDtypeStruct(batch_shape + event_shape, jnp.float32)
    assert _validation.check_event_shape(x, event_shape) is None


@pytest.mark.parametrize("shape,event_shape", [
    ((), (10,)), ((0,), (10,)), ((1,), (10,)), ((2,), (10,)),
    ((20,), (10,)), ((10, 1), (10,)), ((2, 3, 9), (10,)),
    ((), (4, 3)), ((3,), (4, 3)), ((12,), (4, 3)),
    ((3, 4), (4, 3)), ((5, 3), (4, 3)), ((4, 2), (4, 3)),
])
def test_rejects_wrong_rank_or_event_axes(shape, event_shape):
    x = jax.ShapeDtypeStruct(shape, jnp.float32)
    with pytest.raises(ValueError) as error:
        _validation.check_event_shape(x, event_shape)
    assert str(event_shape) in str(error.value)
    assert f"got shape {shape}" in str(error.value)


def test_shape_check_adds_no_jax_operations():
    """A Python metadata check must not become device work or a callback."""
    def identity(x):
        _validation.check_event_shape(x, (10,))
        return x

    x = jnp.arange(10, dtype=jnp.float32)
    assert jax.make_jaxpr(identity)(x).jaxpr.eqns == []
    assert jnp.array_equal(jax.jit(identity)(x), x)
    with pytest.raises(ValueError, match="10"):
        jax.jit(identity)(jnp.zeros(2))


def test_export_accepts_symbolic_batch_size_with_fixed_event_size():
    """One exported density can accept different batch sizes, with dim fixed."""
    export = pytest.importorskip("jax.export")
    dist = DoubleWell(n_dims=10)
    spec = jax.ShapeDtypeStruct(export.symbolic_shape("b, 10"), jnp.float32)

    exported = export.export(jax.jit(lambda x: dist(x)))(spec)

    for batch_size in (1, 3):
        actual = exported.call(jnp.ones((batch_size, 10)))
        assert actual.shape == (batch_size,)
        assert jnp.all(actual == 25.0)


def test_export_rejects_unconstrained_symbolic_event_size():
    """An event of size 2*k cannot be proven to belong to a fixed 10D target."""
    export = pytest.importorskip("jax.export")
    dist = DoubleWell(n_dims=10)
    spec = jax.ShapeDtypeStruct(export.symbolic_shape("b, 2*k"), jnp.float32)

    with pytest.raises(ValueError, match="10"):
        export.export(jax.jit(lambda x: dist(x)))(spec)
