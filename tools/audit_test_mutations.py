"""Challenge selected test contracts with plausible numerical mistakes.

Run from the repository root with PYTHONPATH=., for example:
    PYTHONPATH=. python tools/audit_test_mutations.py --output /tmp/mutations.json

An unmodified full-suite run must pass first. Each case changes one method in
memory, runs its distribution tests plus the shared interface/tracing tests,
then restores the method. JAX caches are cleared
between cases so an old compiled function cannot conceal the mutation. No source
files are edited. These hand-selected faults are a focused audit, not an
exhaustive mutation score; a surviving fault makes the command exit nonzero.
"""
import argparse
import contextlib
import importlib
import inspect
import io
import json
from pathlib import Path
import textwrap

import jax
import pytest

CASES = [
    ('banana_precision', 'banana', 'Banana2D', '__call__', '-0.5 * quad', '-1.0 * quad', ['test_banana.py']),
    ('banana_sampler_scale', 'banana', 'Banana2D', 'sample', 'self.sigma * jr.normal', '1.0 * jr.normal', ['test_banana.py']),
    ('funnel_conditional_scale', 'neal_funnel', 'NealFunnel', '__call__', 'jnp.exp(x0 / 2.0)', 'jnp.exp(x0)', ['test_neal_funnel.py']),
    ('funnel_sampler_scale', 'neal_funnel', 'NealFunnel', 'sample', 'std_cond * jr.normal', '1.0 * jr.normal', ['test_neal_funnel.py']),
    ('lgcp_likelihood_missing', 'log_gauss_pines', 'LGCP', '__call__', 'return prior + likelihood', 'return prior', ['test_lgcp.py']),
    ('doublewell_uniform_sampler', 'double_well', 'DoubleWell', 'sample', 'p=_WEIGHTS', 'p=None', ['test_double_well.py']),
    ('doublewell_missing_gaussian_logz', 'double_well', 'DoubleWell', 'log_normalization', '(_log_z_x() + log_z_y)', '(_log_z_x())', ['test_double_well.py']),
    ('muller_brown_half_energy', 'muller_brown', 'MullerBrown', '__call__', '-self.beta * potential', '-0.5 * self.beta * potential', ['test_muller_brown.py']),
    ('dw4_offset_ignored', 'dw4', 'DW4', '__call__', '+ self.c,', '+ 0.0,', ['test_dw4.py']),
    ('lj_length_scale_ignored', 'lennard_jones', 'LennardJones', '__call__', 'self.rm / r_pairs', '1.0 / r_pairs', ['test_lennard_jones.py']),
    ('phi_four_missing_end_bond', 'phi_four', 'PhiFour', '__call__', '+ ((1, 1),)', '+ ((1, 0),)', ['test_phi_four.py', 'test_lattice_phi_four.py', 'test_phi_four_oracle.py']),
    ('doublewell_shape_check_missing', 'double_well', 'DoubleWell', '__call__', 'check_event_shape(x, (self.dim,))', 'pass', ['test_double_well.py', 'test_validation.py']),
    ('lattice_double_coupling', 'lattice_phi_four', 'LatticePhiFour', '__call__', 'self.kappa / 2 * coupling', 'self.kappa * coupling', ['test_lattice_phi_four.py']),
    ('harmonic_double_energy', 'harmonic_crystal', 'HarmonicCrystal', '__call__', '0.5 * self.spring_constant', 'self.spring_constant', ['test_harmonic_crystal.py']),
    ('periodic_lj_shift_missing', 'periodic_lennard_jones', 'PeriodicLennardJones', '_pair_potential', 'u - shift', 'u', ['test_periodic_lennard_jones.py']),
    ('water_three_body_missing', 'monatomic_water', 'MonatomicWater', '__call__', 'self._two_body(diff) + self._three_body(diff / MW_SIGMA)', 'self._two_body(diff)', ['test_monatomic_water.py']),
]

class Results:
    """Collect evidence without keeping every assertion traceback in memory."""

    def __init__(self):
        self.failed = []
        self.passed = 0
        self.skipped = 0

    def pytest_runtest_logreport(self, report):
        if report.failed:
            self.failed.append(report.nodeid)
        if report.when == "call" and report.passed:
            self.passed += 1
        if report.skipped:
            self.skipped += 1


def run_tests(paths):
    results = Results()
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        code = pytest.main(
            ["-q", "-p", "no:cacheprovider", "--tb=short", "--disable-warnings", *paths],
            plugins=[results],
        )
    if code not in (0, 1):
        raise RuntimeError(f"pytest could not run ({code})\n{capture.getvalue()[-2000:]}")
    return results


def run_mutation(case):
    label, module_name, class_name, method, old, new, files = case
    module = importlib.import_module("jax_pdf." + module_name)
    cls = getattr(module, class_name)
    original = getattr(cls, method)
    source = textwrap.dedent(inspect.getsource(original))
    if source.count(old) != 1:
        raise RuntimeError(f"{label}: mutation must match exactly once; review source changes")
    namespace = dict(vars(module))
    exec(compile(source.replace(old, new), f"<mutation:{label}>", "exec"), namespace)
    try:
        setattr(cls, method, namespace[method])
        jax.clear_caches()
        results = run_tests([
            "tests/test_interface.py", "tests/test_tracing.py",
            *["tests/" + f for f in files],
        ])
    finally:
        setattr(cls, method, original)
        jax.clear_caches()
    return dict(mutation=label, detected=bool(results.failed), failed=results.failed,
                passed=results.passed, skipped=results.skipped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON evidence file")
    parser.add_argument("mutations", nargs="*", help="Optional mutation names; default: all")
    args = parser.parse_args()
    selection = set(args.mutations)
    unknown = selection - {case[0] for case in CASES}
    if unknown:
        parser.error(f"Unknown mutations: {sorted(unknown)}")
    if not Path("tests/test_interface.py").is_file():
        parser.error("Run from the repository root with PYTHONPATH=.")

    baseline = run_tests(["tests/"])
    if baseline.failed:
        raise RuntimeError(f"Fix the unmodified suite before interpreting mutations: {baseline.failed}")
    print(f"Baseline: {baseline.passed} passed, {baseline.skipped} skipped", flush=True)
    evidence = {"baseline": vars(baseline), "mutations": []}
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    for case in CASES:
        if selection and case[0] not in selection:
            continue
        record = run_mutation(case)
        evidence["mutations"].append(record)
        args.output.write_text(json.dumps(evidence, indent=2) + "\n")
        print(record["mutation"], "DETECTED" if record["detected"] else "SURVIVED",
              f"({len(record['failed'])} failing / {record['passed']} passing)", flush=True)
    return 0 if all(record["detected"] for record in evidence["mutations"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
