# Examples

Each example has its own folder containing its script or notebook and saved figures.
The commands below assume you are in the repository root.

Install the package into the Python environment used to run the examples:

```bash
python -m pip install -e .
```

Plotting examples also need Matplotlib. The LGCP notebook needs Jupyter and IPython;
install the package in its kernel's environment as well.

| Example | Purpose | Run or open |
| --- | --- | --- |
| [Alanine](alanine/alanine.py) | Evaluate the optional flexible molecular target in Cartesian and internal coordinates | `python examples/alanine/alanine.py --solvent implicit` |
| [Double well](double_well/double_well_density.py) | Plot the two-dimensional log density | `python examples/double_well/double_well_density.py` |
| [Log Gaussian Cox process](lgcp/log_gaussian_pine.ipynb) | Inspect the MAP estimate, optimization history and Laplace uncertainty | `jupyter lab examples/lgcp/log_gaussian_pine.ipynb` |
| [Muller–Brown](muller_brown/muller_brown_potential.py) | Plot the potential energy surface and its minima | `python examples/muller_brown/muller_brown_potential.py` |
| [Phi-four](phi_four/phi_four_hardness.py) | Compare chain length and correlation length to identify useful sampling regimes | `python examples/phi_four/phi_four_hardness.py` |

For alanine, install `jaxmm >=0.2` and the chemistry extra first; see
[the chemistry instructions](../docs/alanine.md#installation-and-use).
The example does not require OpenMM or external trajectory data.

Plotting scripts save figures beside their source files. Jupyter normally starts
the notebook kernel in the notebook's folder; its figure is saved in the kernel's
working directory. Existing figures and notebook outputs are included for browsing
without rerunning the computations.
