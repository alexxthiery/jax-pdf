"""Plot the Muller-Brown potential energy surface."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jax_pdf import MullerBrown

mb = MullerBrown(beta=1.0)

xgrid = np.linspace(-1.5, 1.0, 200)
ygrid = np.linspace(-0.2, 2.0, 200)
X, Y = np.meshgrid(xgrid, ygrid)
points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)

# Energy U = -log_p (since log_p = -beta * U and beta=1)
E = -np.array(mb(points)).reshape(X.shape)
E = np.minimum(E, 200.0)

fig, ax = plt.subplots(figsize=(7, 4))
cf = ax.contourf(X, Y, E, levels=100, cmap="jet", vmin=-150, vmax=-40)
fig.colorbar(cf, ax=ax, label="Energy")

# Mark the three minima
minima = np.array([[-0.558, 1.442], [0.624, 0.028], [-0.050, 0.467]])
ax.scatter(minima[:, 0], minima[:, 1], c="white", edgecolors="black", s=50, zorder=5)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Muller-Brown potential")

fig.tight_layout()
fig.savefig("examples/muller_brown_potential.png", dpi=300)
print("Saved examples/muller_brown_potential.png")
