"""Tests for how distributions behave across a JAX trace boundary.

Numeric parameters are pytree children, so they can be swept with ``vmap`` or
differentiated. Structural parameters (sizes, flags, modes) are static, so they
stay concrete and may drive Python control flow. Validation runs on concrete
values only (``jax_pdf._validation.is_concrete``).

The shared interface test ``test_call_with_the_distribution_as_a_jit_argument``
covers every distribution; these tests pin the semantics that split makes
possible, and the one cost it carries.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_pdf import HarmonicCrystal, LennardJones, PhiFour

DIM = 8
X = jnp.linspace(-1.0, 1.0, DIM)


class TestFieldSplit:

    def test_structural_fields_are_not_pytree_leaves(self):
        """
        Claim: only the numeric parameters are pytree children.
        Bug it catches: a size, flag or mode left as a child, which becomes a
        tracer across a trace boundary and then cannot be compared or branched
        on. Oracle: the leaves written out.
        """
        leaves = jax.tree_util.tree_leaves(PhiFour(a=0.1, b=0.2, dim_grid=DIM, beta=1.5))

        assert leaves == [0.1, 0.2, 1.5]                 # a, b, beta; not dim_grid or periodic

    def test_structural_fields_of_a_particle_distribution(self):
        """Same claim for the particle tier, where sizes set the input shape."""
        leaves = jax.tree_util.tree_leaves(LennardJones(n_particles=4, epsilon=2.0, rm=1.5))

        assert leaves == [2.0, 1.5, 1.0, 1.0]            # epsilon, rm, trap_scale, beta

    def test_a_static_option_still_branches_across_a_trace_boundary(self):
        """
        Claim: an option that drives Python control flow keeps working when the
        distribution is a jit argument.
        Bug it catches: making the option a child, which makes the branch a
        boolean conversion of a tracer. Oracle: the two options disagree, and
        each matches its own value outside jit.
        """
        call = jax.jit(lambda d, x: d(x))
        x = jnp.linspace(0.1, 1.0, 12).reshape(4, 3)
        individual = LennardJones(n_particles=4, trap_mode="individual")
        com = LennardJones(n_particles=4, trap_mode="com")

        assert call(individual, x) == pytest.approx(float(individual(x)), rel=1e-6)
        assert call(com, x) == pytest.approx(float(com(x)), rel=1e-6)
        assert float(individual(x)) != float(com(x)), "the fixture cannot see trap_mode"

    def test_periodic_flag_branches_across_a_trace_boundary(self):
        """The same claim for PhiFour's boundary condition."""
        call = jax.jit(lambda d, x: d(x))
        open_bc, ring = PhiFour(dim_grid=DIM), PhiFour(dim_grid=DIM, periodic=True)

        assert call(open_bc, X) == pytest.approx(float(open_bc(X)), rel=1e-6)
        assert call(ring, X) == pytest.approx(float(ring(X)), rel=1e-6)
        assert float(open_bc(X)) != float(ring(X)), "the fixture cannot see periodic"


class TestNumericParametersAreTraced:

    def test_a_parameter_can_be_swept_with_vmap(self):
        """
        Claim: a numeric parameter is a pytree child, so a ladder of values maps
        in one call.
        Bug it catches: making every field static, which turns the sweep into a
        retrace per value and fails under vmap.
        Oracle: the same values from a Python loop.
        """
        betas = jnp.array([0.5, 1.0, 2.0])

        swept = jax.vmap(lambda b: PhiFour(a=0.1, dim_grid=DIM, beta=b)(X))(betas)
        looped = jnp.array([PhiFour(a=0.1, dim_grid=DIM, beta=float(b))(X) for b in betas])

        np.testing.assert_allclose(np.asarray(swept), np.asarray(looped), rtol=1e-6)

    def test_gradient_with_respect_to_a_parameter(self):
        """
        Claim: a numeric parameter is differentiable.
        Bug it catches: a static parameter, which gives no gradient at all.
        Oracle: log p = -beta U(x) is linear in beta, so d log p / d beta is
        exactly log p / beta.
        """
        beta = 1.7

        grad = jax.grad(lambda b: PhiFour(a=0.1, dim_grid=DIM, beta=b)(X))(beta)
        expected = float(PhiFour(a=0.1, dim_grid=DIM, beta=beta)(X)) / beta

        assert float(grad) == pytest.approx(expected, rel=1e-5)


class TestValidation:

    def test_positions_shape_is_validated_inside_jit(self):
        """Skipping sentinel validation must not skip available tracer shapes."""
        with pytest.raises(ValueError, match="positions must be"):
            jax.jit(lambda positions: HarmonicCrystal(positions=positions).dim)(
                jnp.zeros(3)
            )

    @pytest.mark.parametrize("bad", [-1.0, np.float32(-1.0), jnp.float32(-1.0)])
    def test_validation_fires_on_every_concrete_value(self, bad):
        """
        Claim: validation still rejects a bad parameter, whether it arrives as a
        Python float, a NumPy scalar or a concrete JAX scalar.
        Bug it catches: a guard so broad that it skips real values too.
        Oracle: the documented error message.
        """
        with pytest.raises(ValueError, match="a must be positive"):
            PhiFour(a=bad, dim_grid=DIM)

    def test_validation_is_skipped_under_tracing(self):
        """
        Claim, and the cost of the split: a parameter that is a tracer cannot be
        compared, so a distribution built inside a traced function is not
        validated. This documents the behaviour rather than blessing it.
        Oracle: the call returns the value the formula gives, with no exception.
        """
        negative_beta = jax.jit(lambda b: PhiFour(a=0.1, dim_grid=DIM, beta=b)(X))(-1.0)

        assert jnp.isfinite(negative_beta)
        assert float(negative_beta) == pytest.approx(-float(PhiFour(a=0.1, dim_grid=DIM, beta=1.0)(X)),
                                                     rel=1e-6)

    def test_a_structural_parameter_is_always_validated(self):
        """
        Claim: static fields are concrete everywhere, so their validation needs
        no guard and cannot be bypassed by tracing.
        Oracle: the documented error, raised inside a jitted construction.
        """
        with pytest.raises(ValueError, match="dim_grid must be >= 2"):
            jax.jit(lambda b: PhiFour(a=0.1, dim_grid=1, beta=b)(X))(1.0)
