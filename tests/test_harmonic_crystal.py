"""Tests for the HarmonicCrystal target (analytic / numeric oracles)."""
import math

import jax
import jax.numpy as jnp
import pytest

from jax_pdf import HarmonicCrystal


def _sites(n=8, d=3, key=0):
    return jax.random.uniform(jax.random.PRNGKey(key), (n, d), minval=0.0, maxval=5.0)


class TestHarmonicCrystal:
    def test_minimum_at_sites(self):
        """U = 0 at the well centres, so log p = 0 there."""
        s = _sites()
        hc = HarmonicCrystal(positions=s, spring_constant=2.0, beta=0.5)
        assert jnp.allclose(hc(s), 0.0, atol=1e-6)

    def test_call_value(self):
        """One particle displaced by 1: U = (k/2)*1, log p = -beta*U."""
        s = jnp.zeros((2, 3))
        hc = HarmonicCrystal(positions=s, spring_constant=2.0, beta=1.0)
        x = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        assert jnp.allclose(hc(x), -1.0, atol=1e-6)   # -beta * (0.5*2*1)

    def test_output_shape_batch(self):
        hc = HarmonicCrystal(positions=_sites(4, 3))
        x = jax.random.normal(jax.random.PRNGKey(1), (5, 4, 3))
        assert hc(x).shape == (5,)

    def test_wrong_shape_raises(self):
        hc = HarmonicCrystal(positions=_sites(4, 3))
        with pytest.raises(ValueError, match=r"x.shape\[-2:\]"):
            hc(jnp.zeros((4, 2)))

    def test_log_normalization_matches_numeric_integral(self):
        """One DoF: log Z must equal log(integral exp(-beta*(k/2) u^2) du)."""
        s = jnp.zeros((1, 1))
        for beta, k in [(1.0, 1.0), (2.0, 3.0)]:
            hc = HarmonicCrystal(positions=s, spring_constant=k, beta=beta)
            u = jnp.linspace(-15.0, 15.0, 300_001)
            integral = jnp.trapezoid(jnp.exp(-beta * 0.5 * k * u**2), u)
            assert jnp.allclose(
                hc.log_normalization(), float(jnp.log(integral)), atol=1e-4
            )

    def test_log_normalization_dof_scaling(self):
        """log Z scales linearly with N*d."""
        hc1 = HarmonicCrystal(positions=jnp.zeros((1, 1)), spring_constant=2.0, beta=0.5)
        hc = HarmonicCrystal(positions=jnp.zeros((4, 3)), spring_constant=2.0, beta=0.5)
        assert jnp.allclose(hc.log_normalization(), 12 * hc1.log_normalization(), atol=1e-6)

    def test_pbc_wrap_invariance(self):
        """With box_length, a particle shifted by a box vector has equal energy."""
        s = _sites(3, 3, key=2)
        L = 6.0
        hc = HarmonicCrystal(positions=s, spring_constant=1.0, beta=1.0, box_length=L)
        x = s + 0.1
        x_wrapped = x.at[0].add(jnp.array([L, 0.0, 0.0]))
        assert jnp.allclose(hc(x), hc(x_wrapped), atol=1e-5)

    def test_validation(self):
        with pytest.raises(ValueError):
            HarmonicCrystal(positions=_sites(), spring_constant=-1.0)
        with pytest.raises(ValueError):
            HarmonicCrystal(positions=_sites(), beta=0.0)

    def test_dim_property(self):
        assert HarmonicCrystal(positions=jnp.zeros((4, 3))).dim == 12

    def test_beta_scaling(self):
        """log p scales linearly with beta (the energy is fixed)."""
        s = _sites(3, 3, key=3)
        x = s + 0.2
        hc1 = HarmonicCrystal(positions=s, spring_constant=2.0, beta=1.0)
        hc2 = HarmonicCrystal(positions=s, spring_constant=2.0, beta=2.0)
        assert jnp.allclose(hc2(x), 2.0 * hc1(x), atol=1e-6)

    def test_gradient_finite(self):
        hc = HarmonicCrystal(positions=_sites(8, 3), spring_constant=1.5, beta=0.5)
        x = jax.random.normal(jax.random.PRNGKey(4), (8, 3))
        grad = jax.grad(hc)(x)
        assert grad.shape == (8, 3)
        assert bool(jnp.all(jnp.isfinite(grad)))

    def test_jit(self):
        hc = HarmonicCrystal(positions=_sites(8, 3))
        x = jax.random.normal(jax.random.PRNGKey(5), (8, 3))
        assert jnp.allclose(jax.jit(hc.__call__)(x), hc(x), atol=1e-6)
