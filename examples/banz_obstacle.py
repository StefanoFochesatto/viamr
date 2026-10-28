#!/usr/bin/env python3
"""Example demonstrating adaptive mesh refinement for a 2D obstacle problem
using the Banz & Schroeder (2021) / Proximal Galerkin a posteriori error estimator
in the viamr library.

The model problem is:
    -Delta u = f  in Omega = (0, 1)^2
    u >= psi      in Omega
    u = 0         on dOmega

where psi is a circular obstacle centered in the domain.

We use VIAMR.banzmark() to:
  1. Compute the certified primal-dual upper bound eta_up.
  2. Inspect the auxiliary (discretization) vs. constraint (contact) branch contributions.
  3. Generate DG0 element markings via Doerfler/bulk marking (method="total").
  4. Adaptively refine the mesh using skeleton-based refinement (refinesbr2D).
"""

from firedrake import *
from viamr import VIAMR


def solve_obstacle(mesh, f_load, psi_ufl):
    """Solve the obstacle problem on the given mesh using PETSc SNESVI."""
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name="Primal Solution")
    v = TestFunction(V)

    lb = Function(V, name="Lower Obstacle").interpolate(psi_ufl)
    ub = Function(V).interpolate(Constant(PETSc.INFINITY))
    bcs = DirichletBC(V, Constant(0.0), "on_boundary")

    F = (inner(grad(u), grad(v)) - f_load * v) * dx
    problem = NonlinearVariationalProblem(F, u, bcs)
    sp = {
        "snes_type": "vinewtonrsls",
        "snes_rtol": 1.0e-8,
        "snes_atol": 1.0e-12,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp)
    solver.solve(bounds=(lb, ub))
    return u, lb, bcs


def main():
    print("=" * 70)
    print("VIAMR: Adaptive Mesh Refinement with Banz-Schroeder Error Control")
    print("=" * 70)

    # Initial coarse mesh
    mesh = UnitSquareMesh(12, 12)
    amr = VIAMR()

    # Obstacle geometry: circular dome
    x, y = SpatialCoordinate(mesh)
    r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2
    r0 = 0.3
    psi_ufl = conditional(le(r2, r0 ** 2), 0.2 * (1.0 - r2 / (r0 ** 2)), -0.1)
    f_load = Constant(0.0)

    max_levels = 4
    theta = 0.4  # Mark elements carrying top 40% of the indicator

    for level in range(max_levels):
        u, lb, bcs = solve_obstacle(mesh, f_load, psi_ufl)

        # Multiplier approximation kappa:
        # * In Proximal Galerkin (PG / LVPP), kappa is given directly by the latent drift
        #   kappa = (s^{k-1} - s^k) / alpha (Banz-Schroeder Galerkin relation holds for free).
        # * When solving with a discrete VI solver (such as PETSc SNESVI), kappa is
        #   reconstructed from the discrete residual:
        #     res_i = \int_Omega grad(u) . grad(phi_i) dx - \int_Omega f phi_i dx
        #   using mass-lumped scaling kappa_i = max(res_i / \int phi_i dx, 0.0) with
        #   Dirichlet boundary nodes zeroed.
        #
        # Calling amr.banz_discrete_multiplier(u, f_ufl=f_load, bcs=bcs) computes this.
        # (Passing kappa=None to banzmark will also compute it automatically).
        kappa = amr.banz_discrete_multiplier(u, f_ufl=f_load, bcs=bcs)

        # Evaluate Banz-Schroeder a posteriori estimator
        mark, fields, eta_up = amr.banzmark(
            u,
            (lb, None),
            kappa=kappa,
            f_ufl=f_load,
            bcs=bcs,
            method="total",
            theta=theta,
            tau="grad",
        )

        n_verts, n_cells, hmin, hmax = amr.meshsizes(mesh)
        n_marked = amr.countmark(mark)

        info = fields.info
        print(f"\n[Level {level}]")
        print(f"  Mesh:        {n_cells} cells, {n_verts} vertices, h in [{hmin:.4f}, {hmax:.4f}]")
        print(f"  Estimator:   eta_up = {eta_up:.4e}")
        print(f"  Breakdown:   aux_part2 = {info['aux_part2']:.4e}, constraint_part2 = {info['constraint_part2']:.4e}")
        print(f"  Dominance:   {fields.dominant_branch.upper()} error dominates")
        print(f"  Marking:     {n_marked} cells marked for refinement (threshold = {info['ethresh']:.4e})")

        if level < max_levels - 1:
            # Refine mesh based on DG0 indicator
            mesh = amr.refinesbr2D(mesh, mark)

    print("\n" + "=" * 70)
    print("Adaptive refinement complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
