# The "pyramid obstacle" example from subsection 3.2 of NSV05 = 
#
#   Nochetto, R. H., Siebert, K. G., & Veeser, A. (2005). Fully localized
#   a posteriori error estimators and barrier sets for contact problems.
#   SIAM J. Numer. Anal. 42(5), 2118-2135.
#
# The domain is the diamond Omega, the square (-1,1)^2 rotated 45 degrees.
# (Equivalently, the \ell^1 unit ball.)   The obstacle is
# the pyramid chi(x) = dist(x, boundary of Omega) - 1/5, the source is f = -5,
# and the boundary values are g = 0.
#
# This example compares VIAMR methods nsv05mark() to nsv03mark() for AMR.  The former is what
# NSV05 call "full localization".  The NSV05 residual indicator
# is identically zero on the discrete full-contact set Omega_h^0, so the mesh
# stays coarse inside the contact set, and in particular along the diagonals
# where the pyramid has its ridges.  Compare Figure 3.4 (bottom row) of NSV05,
# which reports DOFs = 5, 381, 3073 at adaptive steps 0, 5, 10.
#
# Features of this example worth knowing:
#   * chi is concave down and piecewise affine, with ridges along the coordinate
#     axes.  The initial crossed mesh puts element edges on both axes, and SBR
#     refinement never produces an element straddling a ridge, so I_h chi = chi
#     *exactly* at every level.  The obstacle approximation terms of the
#     estimator therefore vanish for a real reason, not by assumption.
#   * Inside the contact set u_h = chi_h has a concave kink across each ridge
#     edge, so the jump J_h = -[[grad u_h]].n is negative there.  Since also
#     f = -5 <= 0, the ridge nodes satisfy both sign conditions in NSV05's
#     definition of the full-contact nodes C_h, and the ridges are swallowed
#     by Omega_h^0.  Refinement along the diagonals in the output would mean
#     the sign convention on J_h had been inverted.

# Two caveats when comparing DOF counts to Figure 3.4 of NSV05.  They refine by
# bisection (via ALBERT), possibly with red/green closure, while we use SBR.  And
# neither NSV05 nor NSV03 states the threshold in the maximum strategy, so
# theta=0.5 below is a guess.  It is the ALBERTA default.

# major parameters
maxlevs05 = 11  # adaptive steps; NSV05 Figure 3.4 bottom goes to step 10
maxlevs03 = 18  # adaptive steps; NSV05 Figure 3.4 top goes to step 17
targetnodes = 1e5  # backstop on maxlevs
theta = 0.5  # threshold in the marking strategy; a guess, see above

import numpy as np
from firedrake import *
from viamr import VIAMR
from firedrake.petsc import PETSc

print = PETSc.Sys.Print  # enables correct printing in parallel

# initial mesh: We want the diamond as 4 triangles around the origin, thus 5 nodes,
# matching "step = 0, DOFs = 5" in Figure 3.4 of NSV05.  We build the DMPlex from
# an explicit cell list, rather than by moving the coordinates of a crossed 1x1
# RectangleMesh, because VIAMR.refinesbr2D() transforms mesh.topology_dm and so
# takes the *DMPlex* coordinates, not the Firedrake coordinate Function; moving
# only the latter would give a diamond at step 0 and squares thereafter.
cells = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]], dtype=np.int32)
coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
plex = PETSc.DMPlex().createFromCellList(2, cells, coords, comm=COMM_WORLD)
mesh0 = Mesh(plex, distribution_parameters=VIAMR.PARALLEL_OVERLAP)

sp = {
    "snes_type": "vinewtonrsls",
    "snes_converged_reason": None,
    "snes_atol": 1.0e-12,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

for method in ["NSV05", "NSV03"]:
    print(f"solving the pyramid obstacle example using the {method} AMR method ...")
    mesh = mesh0
    maxlevs = maxlevs05 if method == "NSV05" else maxlevs03
    for j in range(maxlevs):
        x = SpatialCoordinate(mesh)
        V = FunctionSpace(mesh, "CG", 1)
        amr = VIAMR(debug=True)

        # UFL expressions for the data; dist(x, boundary) = (1 - |x|_1)/sqrt(2)
        # on the ell^1 ball, so the obstacle is a concave pyramid of height
        # 1/sqrt(2) - 1/5 with ridges on the coordinate axes
        f_ufl = Constant(-5.0)
        g_ufl = Constant(0.0)
        chi_ufl = (1.0 - abs(x[0]) - abs(x[1])) / sqrt(2.0) - 0.2

        # discrete obstacle chi_h = I_h chi; exact here, see header comment
        lb = Function(V, name="chi_h (obstacle)").interpolate(chi_ufl)

        # initialize by cross-mesh interpolation, i.e. do mesh sequencing, then
        # lift onto the obstacle so the initial iterate is admissible.  The corners
        # of the diamond are exactly on the source mesh boundary, so point location
        # can miss them; allow_missing_dofs leaves those dofs at zero, which is the
        # boundary value there anyway.
        if j == 0:
            uh = Function(V, name="u_h (solution)").interpolate(Constant(0.0))
        else:
            uh = Function(V, name="u_h (solution)").interpolate(uh, allow_missing_dofs=True)
        uh.interpolate(max_value(uh, lb))

        # state and solve the problem
        vh = TestFunction(V)
        F = inner(grad(uh), grad(vh)) * dx - f_ufl * vh * dx
        g = Function(V).interpolate(g_ufl)  # = I_h g in NSV05
        bcs = DirichletBC(V, g, "on_boundary")
        problem = NonlinearVariationalProblem(F, uh, bcs)
        INFupper = Function(V).interpolate(Constant(PETSc.INFINITY))
        solver = NonlinearVariationalSolver(
            problem, solver_parameters=sp, options_prefix="s"
        )
        solver.solve(bounds=(lb, INFupper))

        # Eh is the paper's own estimator: Etilde_h of (7.1) in NSV03, E_h of
        # Theorem 2.7 in NSV05.  Both are max-norm estimators, so Eh accumulates
        # each term as a sup over elements, not as an l^2 sum.
        _, nelements, _, _ = amr.meshsizes(mesh)
        if method == "NSV03":
            (mark, etainf, etad, sigmah, Eh) = amr.nsv03mark(
                uh, lb, g, f_ufl, g_ufl, theta=theta, method="max"
            )
            print(
                f"  step {j}: DOFs = {V.dim()}, elements = {nelements}, "
                f"Eh = {Eh:.3e}, marked = {amr.countmark(mark)}"
            )
        elif method == "NSV05":
            # mark using the fully-localized NSV05 estimator
            (mark, eta, sz, fullcontact, Eh) = amr.nsv05mark(
                uh, lb, g, f_ufl, g_ufl, theta=theta, method="max"
            )
            nfull = amr.countmark(fullcontact)
            print(
                f"  step {j}: DOFs = {V.dim()}, elements = {nelements}, "
                f"full-contact elements = {nfull} ({100 * nfull / nelements:.1f}%), "
                f"Eh = {Eh:.3e}, marked = {amr.countmark(mark)}"
            )
        else:
            raise NotImplementedError

        if V.dim() > targetnodes or j == maxlevs - 1:
            break
        mesh = amr.refinesbr2D(mesh, mark)

    # fields on the final mesh
    gap = Function(V, name="gap = u_h - chi_h").interpolate(uh - lb)
    active = amr.elemactive(uh, lb)
    active.rename("active")
    fields = [uh, lb, gap, active, mark]
    if method == "NSV03":
        fields += [etainf, sigmah, etad]
    else:
        fields += [fullcontact, eta, sz]
    if mesh.comm.size > 1:
        rank = Function(FunctionSpace(mesh, "DG", 0), name="rank")
        rank.dat.data[:] = mesh.comm.rank
        fields.append(rank)

    outfile = f"result_pyramid_{method}.pvd"
    print(f"generating output file {outfile} ...")
    VTKFile(outfile).write(*fields)
