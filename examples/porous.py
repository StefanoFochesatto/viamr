# Solve a porous-media nonlinear obstacle problem, with strong form
#   - div(u^{gamma-1} grad(u)) = f,  u >= 0
# The domain is a square, and u=g on its boundary.  The source term
# f = f(r) is a negative constant in a center disc, and linear outside
# of that.  Defaults to gamma=2.  The exact solution is known for any gamma.
#
# The solver must handle the degeneracy of the coefficient at
# the free boundary.  Thus the coefficient is regularized:
#   F(u)[v] = \int_\Omega (u + eps)^{gamma-1} grad(u) . grad (v) - f * v dx
#
# For a given mesh, an eps-continuation loop is used: eps is stepped down
# geometrically from a mild eps_start to a small eps_final, re-solving
# and warm-starting from the previous eps at each step.  This is needed
# for robustness at larger gamma, where the coefficient (u+eps)^{gamma-1}
# is more degenerate near the free boundary than at gamma=2.  The eps_final
# floor is raised to 0.1*hmin^(2/gamma) on fine meshes, since below that
# scale the coefficient varies by orders of magnitude within a single
# element (an unresolved layer near the free boundary), which otherwise makes Newton
# diverge; the hmin^(2/gamma) scaling comes from matched asymptotics of
# the degenerate free-boundary profile (gap ~ xi^(2/gamma)).
#
# The Newton initial iterate is raised eps_start above the obstacle,
# and simple backtracking line search is used for the Newton solver.
#
# The AMR method is UDO+BV00.  That is, we use the (heuristic) weighted-norm
# Bernardi & Verfurth (2000; "BV00") version of the BR78 estimator.
# We report the weighted quasi-norm error.  Without the weighting,
# large BR inactive set estimator (=eta) values occur near the free boundary.
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
levels = 7  # number of AMR levels
gamma = 2.0  # note that exact solution has infinite H^1 norm if gamma >= 4
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


def get_exact_ufl(x, y):
    """Exact radial solution to  - Div(grad(gamma^-1 u^gamma)) = f.
    Returns u, f.  This solution has infinite H^1 norm at gamma>=4."""
    a, b = 1.2, 1.0  # parameters in building exact solution
    r = sqrt(x * x + y * y)
    f = conditional(r <= 1.0, -b, -b + a * (r - 1.0))
    tmp = ((a + b) / 2.0) * (0.5 * (r ** 2 - 1.0) - ln(r))
    tmp -= (a / 3.0) * ((r ** 3 - 1.0) / 3 - ln(r))
    u = conditional(r <= 1.0, 0.0, (gamma * tmp) ** (1.0 / gamma))
    return u, f


print(f"solving gamma={gamma} porous-media obstacle problem using continuation and UDO+BV00 ...")
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
dof, errL2, errH1semi, errqn, hausdorff, eff = [], [], [], [], [], []  # for convergence figures
for i in range(levels + 1):
    # initialize on this mesh
    V = FunctionSpace(mesh, "CG", 1)
    dof.append(V.dim())
    print(f"mesh {i}:")
    amr.meshreport(mesh)
    _, _, hmin, _ = amr.meshsizes(mesh)

    # initial iterate *above* solution, especially on active set
    if i == 0:
        uh = Function(V, name="u_h").interpolate(Constant(eps_start))
    else:  # cross-mesh interpolation to new mesh
        uUFL = conditional(uh < lb, lb, uh)  # uses old data uh, lb
        uh = Function(V, name="u_h").interpolate(uUFL + Constant(eps_start))

    # radial source term which is constant then linear
    x, y = SpatialCoordinate(mesh)
    uUFL, fUFL = get_exact_ufl(x, y)
    fsource = Function(V, name="f_source").interpolate(fUFL)

    # weak form for porous media, with epsilon regularization; epsC is
    # updated in-place by the continuation loop below
    epsC = Constant(eps_start)
    v = TestFunction(V)
    Z = abs(uh + epsC) ** (gamma - 1.0)
    F = Z * inner(grad(uh), grad(v)) * dx - fsource * v * dx

    # boundary condition u=g, from exact solution
    g = Function(V, name="g_bdry").interpolate(uUFL)
    bcs = DirichletBC(V, g, "on_boundary")

    # set up solver with u >= 0
    prob = NonlinearVariationalProblem(F, uh, bcs)
    solver = NonlinearVariationalSolver(prob, solver_parameters=sp, options_prefix="s")
    lb = Function(V).interpolate(Constant(0.0))
    ub = Function(V).interpolate(Constant(PETSc.INFINITY))

    # solve using eps-continuation: step eps down geometrically
    # from eps_start to eps_final, warm-starting uh from the previous step.
    # The eps floor is raised when the mesh is finer than eps_final: below that
    # scale Z=(u+eps)^(gamma-1) varies by orders of magnitude *within a
    # single element* near the free boundary (an unresolved boundary layer),
    # which causes Newton to diverge on fine meshes.  Matched asymptotics of
    # the degenerate profile near a generic free-boundary point give
    # gap(xi) ~ kappa*xi^(2/gamma) (xi = distance into the active set), so
    # the layer has physical width xi* ~ (eps/kappa)^(gamma/2); resolving
    # xi* with the mesh requires eps >~ kappa*hmin^(2/gamma).  kappa is
    # problem-dependent and not known in closed form, so it's replaced here
    # by a tuned constant.
    epsfinal_eff = max(eps_final, 0.1 * hmin ** (2.0 / gamma))
    if epsfinal_eff > eps_final:
        print(f"  eps-continuation floor raised to {epsfinal_eff:.2e} > eps_final={eps_final:.2e}")
    epsval = eps_start
    while True:
        epsC.assign(epsval)
        solver.solve(bounds=(lb, ub))
        if epsval <= epsfinal_eff:
            break
        epsval = max(epsfinal_eff, epsval * eps_shrink)

    # errors relative to exact; re-implement errornorm() with fixed
    # quadrature degree
    tmp = assemble(inner(uUFL - uh, uUFL - uh) * dx(degree=6)) ** (1 / 2)
    errL2.append(float(tmp))
    dus = inner(grad(uUFL - uh), grad(uUFL - uh))
    tmp = assemble(dus * dx(degree=6)) ** (1 / 2)
    errH1semi.append(float(tmp))

    # quasi-norm error, with coefficient from numerical solution
    hT = CellDiameter(mesh)
    CQN = 1.0
    Zqn = abs(uh + CQN * hT) ** (gamma - 1.0)  # mesh-native version of Z in solver
    tmp = assemble(Zqn * dus * dx(degree=6)) ** (1 / 2)
    errqn.append(float(tmp))

    # free-boundary Hausdorff distance vs exact; hausdorff2D() returns None
    # for an empty free-boundary edge set (e.g. a coarse initial mesh)
    uexact = Function(V, name="u_exact").interpolate(uUFL)
    _, fbexact = amr.freeboundarygraph2D(uexact, lb)
    _, fb = amr.freeboundarygraph2D(uh, lb)
    tmp = amr.hausdorff2D(fbexact, fb)
    hausstr = f"{tmp:.3e}" if tmp is not None else "n/a"
    hausdorff.append(tmp if tmp is not None else np.nan)  # nan plots as a gap

    # mark and refine by UDO+BV00; tot_eta from this is used in reported effectivity index
    # note: regularize by eps_final because div()
    # differentiates symbolically, producing gamma-2 as power; for gamma<2 that is
    # negative and uh==0 throughout the active set, giving 0**(negative) = inf/nan
    Zunreg = abs(uh + eps_final) ** (gamma - 1.0)
    res = - div(Zunreg * grad(uh)) - fsource
    if not useweightedBR:
        Zqn = None
    imark, eta, tot_eta = amr.brinactivemark(uh, Constant(0.0), res, alpha=Zqn, theta=0.5, method="total")
    fbmark = amr.udomark(uh, lb, n=1)
    mark = amr.unionmarks(fbmark, imark)

    # report errors, effectivity index, Jaccard agreement
    eff.append(tot_eta / errqn[i])  # effectivity index vs quasi-norm error
    print(f"  ||u-u_h||_L2={errL2[i]:.3e};  |u-u_h|_H1={errH1semi[i]:.3e};  |u-u_h|_qn={errqn[i]:.3e}")
    print(f"  eff_qn={eff[i]:.2f};  hausdorff2D(Gamma_u, Gamma_uh) = {hausstr}")

    # actually refine if we will solve on next mesh
    if i < levels:
        mesh = amr.refinesbr2D(mesh, mark)

# generate Paraview-readable file
outfile = "result_porous.pvd"
print(f"done ... writing solution to {outfile} ...")
err = Function(V, name="uerr = u-u_exact").interpolate(uh - uUFL)
imark.rename("inactive mark")
fbmark.rename("free-boundary mark")
mark.rename("mark")
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
    dof_a = np.array(dof)

    figfile = "porous_norms.png"
    print(f"generating convergence figure {figfile} ...")
    series = [
        ("$L^2$ norm", errL2, "ko"),
        ("$H^1$ semi-norm", errH1semi, "bs"),
        ("quasi-norm", errqn, "g^"),
    ]
    plt.figure()
    for label, vals, style in series:
        vals_a = np.array(vals)
        rate = np.polyfit(np.log(dof_a), np.log(vals_a), 1)[0]
        plt.loglog(dof_a, vals_a, style, label=f"{label} (rate={rate:.2f})")
    y = dof_a.astype(float) ** (-2.0 / d)
    y *= errL2[0] / y[0]  # anchor to first L^2 data point
    plt.loglog(dof_a, y, "k:", label="DOFs^(-2/d)")
    plt.legend()
    plt.grid(True)
    plt.xlabel("DOFs")
    plt.ylabel("error norm")
    plt.title(f"error norms $|u-u_h|_*$")
    plt.savefig(figfile)

    # Hausdorff distance is a geometric (not squared-error) quantity, so the
    # natural reference is mesh-resolution scaling h ~ DOFs^(-1/d), not the
    # L^2/H^1 rates above; as in sphere.py
    hausfile = "porous_hausdorff.png"
    print(f"generating convergence figure {hausfile} ...")
    haus_a = np.array(hausdorff)
    valid = np.isfinite(haus_a) & (haus_a > 0)
    plt.figure()
    if np.any(valid):
        plt.loglog(dof_a, haus_a, "ko", label=f"Hausdorff distance")
        anchor = np.argmax(valid)  # index of first valid entry
        y = dof_a.astype(float) ** (-1.0 / d)
        y = y * haus_a[anchor] / y[anchor]
        plt.loglog(dof_a, y, "k:", label="DOFs^(-1/d)")
        plt.legend()
    plt.grid(True)
    plt.xlabel("DOFs")
    plt.ylabel("hausdorff2D(Gamma_u, Gamma_uh)")
    plt.title(f"free-boundary Hausdorff distance")
    plt.savefig(hausfile)

    # effectivity index = estimator / true error, in the quasi-norm the
    # weighted BV00 estimator targets; as in nsv.py, use a linear
    # y-axis with a semilogx x-axis and a horizontal reference at eff=1
    efffile = "porous_effectivity.png"
    print(f"generating effectivity figure {efffile} ...")
    eff_a = np.array(eff)
    plt.figure()
    plt.semilogx(dof_a, eff_a, "g^", label="BV00 in quasi-norm")
    plt.axhline(1.0, color="k", linestyle=":", label="ideal eff=1")
    plt.legend()
    plt.grid(True)
    plt.xlabel("DOFs")
    plt.ylabel("effectivity ratio (estimator / true-error)")
    plt.title("effectivity index vs DOFs")
    plt.savefig(efffile)
