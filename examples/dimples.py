# This example is a stepping stone toward comparing AMR strategies (NSV,
# NSV+VIAMR.safeactiveunmark(), UDO+BR) on a *dimpled* obstacle problem,
# where a state-degenerate operator can make active-set refinement provably
# wasted effort even though the obstacle has fine unresolved detail.  See
# the FIXME stub at the bottom of this file for that eventual comparison,
# not yet implemented.
#
# For now, this file compares the three AMR strategies from the FIXME stub
# below on the plain (non-dimpled) "sphere" obstacle problem from sphere.py:
# obstacle psi is a spherical cap glued to a linear far-field, f=0
# (harmonic), and the exact solution/boundary data is sphere.py's known
# logarithmic far-field, so the active set is confined to a disc r <= afree
# near the origin, exactly as in sphere.py.  The spherical cap is concave
# down (superharmonic) there, so sigma_psi = -div(grad(psi)) - f > 0
# throughout the active set, and VIAMR.safeactiveunmark() is expected to
# certify the whole active set safe at every level.

import numpy as np
from firedrake import *
from firedrake.petsc import PETSc
from viamr import VIAMR

print = PETSc.Sys.Print  # enables correct printing in parallel

m0 = 10  # initial mesh is m0 x m0
targetelements = 1.0e5  # all methods stop refining once this is met
maxlevels = 11  # backstop target element complexity; set to 15 for more data?
thetaBR = 0.5  # controls BR resolution in inactive set
methodBR = "total"


def psiUFL(r):
    """Spherical-cap obstacle (as in sphere.py): a hemisphere for r <= r0,
    glued C^1 to its tangent line for r > r0."""
    r0 = 0.9
    psi0 = np.sqrt(1.0 - r0 * r0)
    dpsi0 = -r0 / psi0
    return conditional(le(r, r0), sqrt(1.0 - r * r), psi0 + dpsi0 * (r - r0))


def uexactUFL(r):
    """Exact solution (and boundary condition) as UFL, from sphere.py:
    equals the obstacle inside the (known) free boundary radius afree, and
    the matching harmonic log far-field outside it."""
    afree = 0.697965148223374
    A, B = 0.680259411891719, 0.471519893402112
    return conditional(le(r, afree), psiUFL(r), -A * ln(r) + B)


def f_ufl():
    """Source term f; f=0 (harmonic) matches sphere.py. FIXME f needed for
    dimpled version."""
    return Constant(0.0)


def F_strong(u, f):
    """Strong form of the classical obstacle problem operator: L(u) - f."""
    return -div(grad(u)) - f


def errornorm_H1semi_inactive_deg(u, uh, iamark, fixeddegree=6):
    """H^1 seminorm (energy norm) error of uh against exact solution u,
    restricted to the *computed inactive set* iamark, a DG0 indicator, as
    returned by VIAMR.eleminactive(uh, lb, strong=True).  Usess fixed
    high-degree quadrature to avoid a TSFC warning.

    Restricting to the inactive set is appropriate when comparing AMR
    methods that correctly stop refining a safe active set: since u=psi
    there regardless of mesh resolution, further refinement does not
    reduce error against the known (continuum) data psi."""
    dus = inner(grad(u - uh), grad(u - uh))
    normsq = assemble(dus * iamark * dx(degree=fixeddegree))
    return np.sqrt(normsq)


sp = {
    "snes_type": "vinewtonrsls",
    "snes_vi_zero_tolerance": 1.0e-12,
    "snes_linesearch_type": "basic",
    "snes_max_it": 200,
    "snes_rtol": 1.0e-8,
    "snes_atol": 1.0e-12,
    "snes_stol": 0.0,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_converged_reason": None,
}

# three AMR strategies: plain NSV (kept to show this method's disadvantage:
# unbounded active-set refinement growth); NSV filtered by
# VIAMR.safeactiveunmark(); and UDO+BR.  The real comparison of interest is
# UDO+BR (heuristic, fast) against NSV+safeactiveunmark() (more rigorous,
# slower, and now itself a bit heuristic because of the safety filter).
# NSV is the meaningful pairing with safeactiveunmark(), not UDO+BR:
# nsvmark()'s term 1 estimator is computed over the *entire* domain (not
# restricted to the inactive set), so it can force refinement deep in the
# active set whenever its X = |f + sigma_h| (or |f| off the active
# neighborhood) term is nonzero there.  UDO+BR marks active-side elements
# only incidentally, via udomark()'s border dilation reaching just past the
# free boundary, so safeactiveunmark() has much less to remove there.
amrtypes = ["nsv", "nsvsafe", "udobr"]

for amrtype in amrtypes:
    print(f"solving sphere obstacle problem (f=0, no dimples yet) using {amrtype.upper()} ...")
    mesh = RectangleMesh(
        m0,
        m0,
        2.0,
        2.0,
        originX=-2.0,
        originY=-2.0,
        diagonal="crossed",
        distribution_parameters=VIAMR.PARALLEL_OVERLAP,
    )
    amr = VIAMR()
    for i in range(maxlevels + 1):
        print(f"mesh {i}:")
        V = FunctionSpace(mesh, "CG", 1)
        amr.meshreport(mesh)
        x, y = SpatialCoordinate(mesh)
        r = sqrt(x * x + y * y)
        psi_ufl = psiUFL(r)
        g_ufl = uexactUFL(r)

        if i == 0:
            uh = Function(V, name="u_h")
        else:  # warm start by cross-mesh interpolation from the previous level
            uUFL = conditional(uh < lb, lb, uh)
            uh = Function(V, name="u_h").interpolate(uUFL)

        v = TestFunction(V)
        F = (inner(grad(uh), grad(v)) - f_ufl() * v) * dx
        bcs = DirichletBC(V, g_ufl, "on_boundary")
        problem = NonlinearVariationalProblem(F, uh, bcs)
        solver = NonlinearVariationalSolver(problem, solver_parameters=sp, options_prefix="s")
        lb = Function(V, name="psi").interpolate(psi_ufl)
        ub = Function(V).interpolate(Constant(PETSc.INFINITY))
        solver.solve(bounds=(lb, ub))

        _, Ne, _, _ = amr.meshsizes(mesh)
        active = amr.elemactive(uh, lb)
        iamark = amr.eleminactive(uh, lb, strong=True)
        errH1 = errornorm_H1semi_inactive_deg(g_ufl, uh, iamark)  # g_ufl = uexactUFL(r)
        Na = amr.countmark(active)
        print(f"  active elements: {Na} ({100.0 * Na / Ne:.2f} %)")
        print(f"  |u_exact - u_h|_H1 (inactive set) = {errH1:.3e}")

        if Ne > targetelements:
            break

        if amrtype in ("nsv", "nsvsafe"):
            g = Function(V).interpolate(g_ufl)
            safe = None
            if amrtype == "nsvsafe":
                # compute safe *before* nsvmark(), so its eta values are
                # excluded from the marking threshold itself, not merely
                # filtered from the mark afterward (see nsvmark() docstring)
                safe = amr.safeactiveunmark(uh, lb, F_strong, psi_ufl, f_ufl())
                print(f"  safeactiveunmark() excludes {amr.countmark(safe)} "
                      "active elements from the nsvmark() threshold")
            mark, _, _, _, _ = amr.nsvmark(uh, lb, g, f_ufl(), g_ufl, safe=safe)
        else:
            fbmark = amr.udomark(uh, lb, n=1)
            residual = -div(grad(uh))
            imark, _, _ = amr.brinactivemark(uh, lb, residual, theta=thetaBR, method=methodBR)
            mark = amr.unionmarks(fbmark, imark)

        mesh = amr.refinemarkedelements(mesh, mark)

    # write output files
    outfile = f"result_dimples_{amrtype}.pvd"
    active.rename("active")
    gap = Function(V, name="gap = uh-lb").interpolate(uh - lb)
    uexact = Function(V, name="u_exact").interpolate(uexactUFL(r))
    uerr = Function(V, name="error = |uexact - uh|").interpolate(
        abs(uexact - uh)
    )
    fields = [uh, lb, gap, uerr, active]
    if amrtype == "udobr":
        fbmark = amr.udomark(uh, lb, n=1)
        residual = -div(grad(uh))
        imark, _, _ = amr.brinactivemark(uh, lb, residual, theta=thetaBR, method=methodBR)
        mark = amr.unionmarks(fbmark, imark)
        fields += [fbmark, imark, mark]
    if amrtype in ["nsv", "nsvsafe"]:
        g = Function(V).interpolate(g_ufl)
        safe = None
        if amrtype == "nsvsafe":
            safe = amr.safeactiveunmark(uh, lb, F_strong, psi_ufl, f_ufl())
            safe.rename("safe active unmarked")
        mark, _, _, _, _ = amr.nsvmark(uh, lb, g, f_ufl(), g_ufl, safe=safe)
        fields.append(mark)
        if amrtype == "nsvsafe":
            fields.append(safe)
    print(f"done ... writing to {outfile} ...")
    VTKFile(outfile).write(*fields)
    print("")


# FIXME (stub, not yet implemented): extend the above to a *dimpled*
# obstacle, and to a second, porous-like operator.  The three AMR strategies
# (NSV; NSV+safeactiveunmark(); UDO+BR) are already implemented above; what's
# missing is:
#
# start from the sphere problem above and add gaussian dimples scattered on
# the upper hemisphere obstacle to make psi.  change f_ufl() to various
# constant negative values, but not too negative (compared to laplacian of
# psi in the dimples)
#
# add a second problem: the porous-like operator
#   -div((u-psi)^{gamma-1} grad(u)) - f
# alongside the classical -div(grad(u)) - f used above
#
# the main ideas:
#   * for the classical problem, you *need* to refine in the coarse active sets
#     to pick up the (potential or actual according to f vs lap(psi)) small
#     inactive sets in the dimples
#   * for the porous-like problem, with a state-degenerate coefficient, it is safe
#     to skip marking and refinement in the active set as long as f<0
