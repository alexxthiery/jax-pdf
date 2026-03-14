"""Plot the 2D double-well log-density."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jax_pdf import DoubleWell

dw = DoubleWell(n_dims=2)

x = np.linspace(-4.0, 4.0, 400)
y = np.linspace(-6.0, 6.0, 400)
X, Y = np.meshgrid(x, y)
points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)

log_p = np.array(dw(points)).reshape(X.shape)

vmax = log_p.max()
Z = np.clip(log_p, vmax - 20, vmax)

fig, ax = plt.subplots(figsize=(6, 5))
cf = ax.contourf(X, Y, Z, levels=100, cmap="jet", vmin=vmax - 20, vmax=vmax)
fig.colorbar(cf, ax=ax, label="log p(x, y)")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Double-well log-density (2D)")
ax.set_aspect("equal")

fig.tight_layout()
fig.savefig("examples/double_well_density.png", dpi=300)
print("Saved examples/double_well_density.png")
