# Solve a porous-media nonlinear obstacle problem, with strong form
#   - div(u^{gamma-1} grad(u)) = f,  u >= 0
# The domain is a square, and u=g on boundary.  The source term f(r)
# is a negative constant in a center disc, and linear outside of that.
# Defaults to gamma=2.  The exact solution is known for any gamma.
#
# The solver must handle the degeneracy of the coefficient at
# the free boundary.  Thus the coefficient is regularized:
#   F(u)[v] = (u + eps)^{gamma-1} grad(u) . grad (v) - f * v
#
# For a given mesh, an eps-continuation loop is used: eps is stepped down
# geometrically from a mild eps_start to a small eps_final, re-solving
# and warm-starting from the previous eps at each step.  This is needed
# for robustness at larger gamma, where the coefficient (u+eps)^{gamma-1}
# is more degenerate near the free boundary than at gamma=2.
#
# The Newton initial iterate is raised eps_start above the obstacle,
# and simple backtracking line search is used for the Newton solver.
#
# The AMR method is UDO+BV00.  That is, we use the (heuristic) weighted-norm
# Bernardi & Verfurth (2000) version of the BR78 estimator.  We report the
# weighted quasi-norm error.  Without the weighting, large BR inactive
# set estimator (=eta) values occur near the free boundary.
#
# We generate a .png convergence figure of the norms vs DOFs.  Because
# the porous-media operator is degenerate, not uniformly elliptic like
# the Laplacian in nsv.py and sphere.py, we fit and
# report the empirical rate from the data itself.  But we also show
# the classical a priori rate DOFs^(-2/d) as a reference.

import numpy as np
from firedrake import *
from firedrake.petsc import PETSc

print = PETSc.Sys.Print  # enables correct printing in parallel

from viamr import VIAMR
from pyop2.mpi import MPI  # for MPI reduce in parallel

# major parameters
m0 = 6  # initial mesh is m0 x m0
levels = 7  # number of AMR levels; converges to 9, at least, for gamma=2
gamma = 2.0  # solves porous-media as VI:  - div(u^{gamma-1} grad(u)) = f
             # note that exact solution has infinite H^1 norm if gamma >= 4
a, b = 1.2, 1.0  # parameters in building exact solution
useweightedBR = True  # apply brinactivemark() using BV00 weighting

# eps-continuation: on each mesh, step eps down geometrically, by _shrink, from
# _start to _final, re-solving and warm-starting at each step
eps_start = 0.1
eps_final = 0.0005
eps_shrink = 0.3

# solver parameters for VI
sp = {
    "snes_type": "vinewtonrsls",
    "snes_vi_zero_tolerance": 1.0e-12,
    "snes_linesearch_type": "bt",
    "snes_linesearch_order": 1,  # force simple backtracking
    "snes_max_it": 200,
    "snes_rtol": 1.0e-8,
    "snes_atol": 1.0e-12,
    "snes_stol": 0.0,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_converged_reason": None,
    # "snes_vi_monitor": None,
    # "snes_linesearch_monitor": None,
}


def get_uexact_ufl(x, y):
    """Exact radial solution to  - Div(grad(gamma^-1 u^gamma)) = f.
    Note that this exact solution has infinite H^1 norm at gamma>=4."""
    r = sqrt(x * x + y * y)
    tmp = ((a + b) / 2.0) * (0.5 * (r ** 2 - 1.0) - ln(r))
    tmp -= (a / 3.0) * ((r ** 3 - 1.0) / 3 - ln(r))
    return conditional(r <= 1.0, 0.0, (gamma * tmp) ** (1.0 / gamma))


def maxeta(mesh, eta):  # maximum of BR estimator, even in parallel
    mine = max(eta.dat.data_ro)
    return float(eta.function_space().mesh().comm.allreduce(mine, op=MPI.MAX))


print(f"solving gamma={gamma} porous-media obstacle problem using UDO+BV00 ...")
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
dofs, errL2s, errH1s, errqns, hausdorffs = [], [], [], [], []  # for convergence figure
for i in range(levels + 1):
    # initialize on this mesh
    V = FunctionSpace(mesh, "CG", 1)
    print(f"mesh {i}:")
    amr.meshreport(mesh)

    # initial iterate *above* solution, especially on active set
    if i == 0:
        uh = Function(V, name="u_h").interpolate(Constant(eps_start))
    else:  # cross-mesh interpolation to new mesh
        uUFL = conditional(uh < lb, lb, uh)  # uses old data uh, lb
        uh = Function(V, name="u_h").interpolate(uUFL + Constant(eps_start))

    # radial source term which is constant then linear
    x, y = SpatialCoordinate(mesh)
    r = sqrt(x * x + y * y)
    fUFL = conditional(r <= 1.0, -b, -b + a * (r - 1.0))
    fsource = Function(V, name="f_source").interpolate(fUFL)

    # weak form for porous media, with epsilon regularization; epsC is
    # updated in-place by the continuation loop below
    epsC = Constant(eps_start)
    v = TestFunction(V)
    Z = abs(uh + epsC) ** (gamma - 1.0)
    F = Z * inner(grad(uh), grad(v)) * dx - fsource * v * dx

    # boundary condition u=g needs exact solution
    uUFL = get_uexact_ufl(x, y)
    g = Function(V, name="g_bdry").interpolate(uUFL)
    bcs = DirichletBC(V, g, "on_boundary")

    # solve with u >= 0, via eps-continuation: step eps down geometrically
    # from eps_start to eps_final, warm-starting uh from the previous step
    prob = NonlinearVariationalProblem(F, uh, bcs)
    solver = NonlinearVariationalSolver(prob, solver_parameters=sp, options_prefix="s")
    lb = Function(V, name="psi").interpolate(Constant(0.0))
    ub = Function(V).interpolate(Constant(PETSc.INFINITY))
    epsval = eps_start
    while True:
        epsC.assign(epsval)
        solver.solve(bounds=(lb, ub))
        if epsval <= eps_final:
            break
        epsval = max(eps_final, epsval * eps_shrink)

    # errors relative to exact; we re-implement errornorm() with fixed
    # quadrature degree; also get weighted quasi-norm
    errL2 = assemble(inner(uUFL - uh, uUFL - uh) * dx(degree=6)) ** (1 / 2)
    dus = inner(grad(uUFL - uh), grad(uUFL - uh))
    errH1 = assemble(dus * dx(degree=6)) ** (1 / 2)
    hT = CellDiameter(mesh)
    CQN = 1.0
    Zqn = abs(uh + CQN * hT) ** (gamma - 1.0)  # well-behaved mesh-native version of Z
    errqn = assemble(Zqn * dus * dx(degree=6)) ** (1 / 2)
    dofs.append(V.dim())
    errL2s.append(float(errL2))
    errH1s.append(float(errH1))
    errqns.append(float(errqn))

    # active set agreement by jaccard
    neweactive = amr.elemactive(uh, lb)
    if i > 0:
        jac = amr.jaccard(neweactive, eactive, submesh=True)
    eactive = neweactive

    # free-boundary Hausdorff distance vs exact; hausdorff() returns None
    # for an empty free-boundary edge set (e.g. a coarse initial mesh)
    uexact = Function(V, name="u_exact").interpolate(uUFL)
    _, fbexact = amr.freeboundarygraph2D(uexact, lb)
    _, fb = amr.freeboundarygraph2D(uh, lb)
    haus = amr.hausdorff(fbexact, fb)
    hausstr = f"{haus:.5f}" if haus is not None else "n/a"
    hausdorffs.append(haus if haus is not None else np.nan)  # nan plots as a gap

    # mark and refine by UDO+BR; tot_eta from this is used in reported effectivity index
    # note: regularize by eps_final, not bare abs(uh), because div()
    # differentiates symbolically, producing gamma-2 as power; for gamma<2 that is
    # negative and uh==0 throughout the active set, giving 0**(negative) = inf/nan
    Zunreg = abs(uh + eps_final) ** (gamma - 1.0)
    res = - div(Zunreg * grad(uh)) - fsource
    if not useweightedBR:
        Zqn = None
    imark, eta, tot_eta = amr.brinactivemark(uh, Constant(0.0), res, alpha=Zqn, theta=0.5, method="total")
    eff = tot_eta / (errqn if useweightedBR else errH1)
    fbmark = amr.udomark(uh, lb, n=1)
    mark = amr.unionmarks(fbmark, imark)

    # report errors, effectivity index, Jaccard agreement
    print(f"  ||u-u_h||_L2={errL2:.3e};  |u-u_h|_H1={errH1:.3e};  |u-u_h|_qn={errqn:.3e}")
    print(f"  eff={eff:.2f}" + (f";  Jaccard({i-1},{i})={100*jac:.2f}%" if i > 0 else ""))
    print(f"  hausdorff(Gamma_u, Gamma_uh) = {hausstr}")

    # actually refine if we will solve on next mesh
    if i < levels:
        mesh = amr.refinemarkedelements(mesh, mark)

# generate Paraview-readable file
outfile = "result_porous.pvd"
print(f"done ... writing solution to {outfile} ...")
err = Function(V, name="uerr = u-u_exact").interpolate(uh - uUFL)
imark.rename("inactive mark")
fbmark.rename("free-boundary mark")
mark.rename("UDO+BR mark")
eta.rename("eta " + "(BV00)" if useweightedBR else "(BR78)")
fields = [uh, fsource, err, imark, fbmark, mark, eta]
if mesh.comm.size > 1:
    # in parallel, write integer-valued element-wise process rank
    DG0 = FunctionSpace(mesh, "DG", 0)
    rank = Function(DG0, name="rank")
    rank.dat.data[:] = mesh.comm.rank
    fields.append(rank)
VTKFile(outfile).write(*fields)

# convergence figures; measured rates are empirically fit, since the
# porous-media operator is degenerate (not uniformly elliptic) and so
# the classical rates are not rigorously justified here; we plot them
# anyway, as references, alongside the measured rates
if mesh.comm.rank == 0:
    import matplotlib.pyplot as plt

    d = 2  # spatial dimension
    dofs_a = np.array(dofs)

    figfile = "porous_convergence.png"
    print(f"generating convergence figure {figfile} ...")
    series = [
        ("$L^2$", errL2s, "ko"),
        ("$H^1$", errH1s, "bs"),
        ("quasi-norm", errqns, "g^"),
    ]
    plt.figure()
    for label, vals, style in series:
        vals_a = np.array(vals)
        rate = np.polyfit(np.log(dofs_a), np.log(vals_a), 1)[0]
        plt.loglog(dofs_a, vals_a, style, label=f"{label} (rate={rate:.2f})")
    y = dofs_a.astype(float) ** (-2.0 / d)
    y *= errL2s[0] / y[0]  # anchor to first L^2 data point
    plt.loglog(dofs_a, y, "k:", label="DOFs^(-2/d)")
    plt.legend()
    plt.grid(True)
    plt.xlabel("DOFs")
    plt.ylabel("error norm")
    plt.title(f"porous-media problem (gamma={gamma}): UDO+BV00 convergence")
    plt.savefig(figfile)

    # Hausdorff distance is a geometric (not squared-error) quantity, so the
    # natural reference is mesh-resolution scaling h ~ DOFs^(-1/d), not the
    # L^2/H^1 rates above; as in sphere.py
    hausfile = "porous_convergence_hausdorff.png"
    print(f"generating convergence figure {hausfile} ...")
    haus_a = np.array(hausdorffs)
    valid = np.isfinite(haus_a) & (haus_a > 0)
    plt.figure()
    if np.any(valid):
        rate = np.polyfit(np.log(dofs_a[valid]), np.log(haus_a[valid]), 1)[0]
        plt.loglog(dofs_a, haus_a, "ko", label=f"Hausdorff (rate={rate:.2f})")
        anchor = np.argmax(valid)  # index of first valid entry
        y = dofs_a.astype(float) ** (-1.0 / d)
        y = y * haus_a[anchor] / y[anchor]
        plt.loglog(dofs_a, y, "k:", label="DOFs^(-1/d)")
        plt.legend()
    plt.grid(True)
    plt.xlabel("DOFs")
    plt.ylabel("hausdorff(Gamma_u, Gamma_uh)")
    plt.title(f"porous-media problem (gamma={gamma}): free-boundary Hausdorff distance")
    plt.savefig(hausfile)
