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
# floor is raised to 0.2*hmin^(2/gamma) on fine meshes, since below that
# scale the coefficient varies by orders of magnitude within a single
# element (an unresolved layer near the free boundary), which otherwise makes Newton
# diverge; the hmin^(2/gamma) scaling comes from matched asymptotics of
# the degenerate free-boundary profile (gap ~ xi^(2/gamma)).
#
# The Newton initial iterate is raised eps_start above the obstacle,
# and simple backtracking line search is used for the Newton solver.
#
# We compare three AMR methods:
#   1. UDOBV = unstructured dilation operator plus the (heuristic)
#              weighted-norm BV00 estimator in the inactive set
#   2. UNI = uniform refinement
#   3. AVM = averaged-metric mesh adaptation (only if "import animate" succeeds)
#
# NSV is not included: that estimator assumes the Laplacian internally,
# so it is not meaningful for this degenerate operator.
#
# We generate .pvd files: result_porous_{udobv,uni,avm}.pvd
#
# We generate five .png convergence figures comparing all methods:
#   porous_L2.png           ||u-u_h||_2 vs DOFs
#   porous_H1.png           |u-u_h|_{H^1 seminorm} vs DOFs
#   porous_qn.png           weighted quasi-norm error vs DOFs
#   porous_hausdorff.png    hausdorff2D(Gamma_u, Gamma_uh) vs DOFs
#   porous_effectivity.png  UDOBV-only: estimator/true-error effectivity index vs DOFs
# Because the porous-media operator is degenerate, not uniformly elliptic
# like the Laplacian in nsv.py and sphere.py, the classical a priori rates
# (DOFs^(-2/d) for L2/quasi-norm, DOFs^(-1/d) for H1/Hausdorff) are shown
# only as references, not as rigorously justified rates.

import numpy as np
from firedrake import *
from firedrake.petsc import PETSc

print = PETSc.Sys.Print  # enables correct printing in parallel

from viamr import VIAMR
from pyop2.mpi import MPI  # for MPI reduce in parallel

try:
    import netgen
except ImportError:
    raise ImportError("Unable to import NetGen.  Exiting.")
from netgen.geom2d import SplineGeometry

# what methods of adaptive refinement do we test
refinetypes = ["udobv", "uni"]
try:
    import animate
except ImportError:
    print(
        "WARNING: animate import failed, proceeding without it using Firedrake meshes only"
    )
else:
    refinetypes.append("avm")  # if import succeeded

# major parameters
m0 = 6  # initial mesh is m0 x m0, except AVM
levels = 7  # number of AMR levels
uniformlevels = 5  # levels of uniform refinement for UNI; kept smaller than
# `levels` because uniform refinement multiplies element count by ~4 per
# level, so `levels` uniform steps from m0 would be far more expensive than
# the adaptive methods reach in the same number of steps
gamma = 2.0  # note that exact solution has infinite H^1 norm if gamma >= 4
useweightedBR = True  # apply brinactivemark() using BV00 weighting
targetelements = 1.0e5  # UDOBV/AVM: stop refining once this many elements is met

# AVM parameters; attempts to do apples-to-apples vs UDOBV
initialhAVM = 4.0 / m0
targetsAVM = [100, 300, 900, 3000, 7000, 18000, 50000, 100000]

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


print(f"solving gamma={gamma} porous-media obstacle problem using continuation ...")

dp = VIAMR.PARALLEL_OVERLAP  # freeboundarygraph2D() needs this in parallel

results = {}  # per-method: (dof, errL2, errH1semi, errqn, hausdorff, eff)
for amrtype in refinetypes:
    print(f"solving by VIAMR using {amrtype.upper()} method ...")
    amr = VIAMR()
    dof, errL2, errH1semi, errqn, hausdorff, eff = [], [], [], [], [], []  # for convergence figures

    # create initial mesh
    if amrtype == "avm":
        geo = SplineGeometry()
        geo.AddRectangle(p1=(-2, -2), p2=(2, 2), bc="rectangle")
        ngmsh = geo.GenerateMesh(maxh=initialhAVM)
        mesh = Mesh(ngmsh, distribution_parameters=dp)
    else:
        mesh = RectangleMesh(
            m0,
            m0,
            2.0,
            2.0,
            originX=-2.0,
            originY=-2.0,
            diagonal="crossed",
            distribution_parameters=dp,
        )
    meshHist = [mesh]
    if amrtype == "uni":
        unimh = MeshHierarchy(mesh, uniformlevels)
    # UNI stops after uniformlevels distinct meshes; re-solving on the same
    # finest uniform mesh past that point would be redundant
    looplevels = uniformlevels if amrtype == "uni" else levels

    for i in range(looplevels + 1):
        mesh = meshHist[i]
        V = FunctionSpace(mesh, "CG", 1)
        dof.append(V.dim())
        print(f"mesh {i}:")
        amr.meshreport(mesh)
        _, Ne, hmin, _ = amr.meshsizes(mesh)

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
        epsfinal_eff = max(eps_final, 0.2 * hmin ** (2.0 / gamma))
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

        print(f"  ||u-u_h||_L2={errL2[i]:.3e};  |u-u_h|_H1={errH1semi[i]:.3e};  |u-u_h|_qn={errqn[i]:.3e}")

        if amrtype == "udobv":
            # mark by UDO+BV00; tot_eta from this is used in reported effectivity index
            # note: regularize by eps_final because div()
            # differentiates symbolically, producing gamma-2 as power; for gamma<2 that is
            # negative and uh==0 throughout the active set, giving 0**(negative) = inf/nan
            Zunreg = abs(uh + eps_final) ** (gamma - 1.0)
            res = - div(Zunreg * grad(uh)) - fsource
            imark, eta, tot_eta = amr.brinactivemark(
                uh, Constant(0.0), res, alpha=(Zqn if useweightedBR else None), theta=0.5, method="total"
            )
            fbmark = amr.udomark(uh, lb, n=1)
            mark = amr.unionmarks(fbmark, imark)
            eff.append(tot_eta / errqn[i])  # effectivity index vs quasi-norm error
            print(f"  eff_qn={eff[i]:.2f};  hausdorff2D(Gamma_u, Gamma_uh) = {hausstr}")
        else:
            print(f"  hausdorff2D(Gamma_u, Gamma_uh) = {hausstr}")

        # UDOBV/AVM: stop once target complexity is met, rather than
        # continuing to refine past it (UNI is already bounded by looplevels)
        if amrtype in ("udobv", "avm") and Ne > targetelements:
            break

        # actually refine if we will solve on next mesh
        if i < looplevels:
            if amrtype == "uni":
                mesh = unimh[i + 1]
            elif amrtype == "avm":
                amr.setmetricparameters(
                    target_complexity=targetsAVM[min(i + 1, len(targetsAVM) - 1)],
                    h_min=1.0e-4,
                    h_max=1.0,
                )
                metric = amr.buildaveragedmetric(mesh, uh, lb)
                mesh = animate.adapt(mesh, metric)
            else:
                mesh = amr.refinesbr2D(mesh, mark)
            meshHist.append(mesh)

    results[amrtype] = (dof, errL2, errH1semi, errqn, hausdorff, eff)

    # generate Paraview-readable file
    outfile = f"result_porous_{amrtype}.pvd"
    print(f"done ... writing solution to {outfile} ...")
    err = Function(V, name="uerr = u-u_exact").interpolate(uh - uUFL)
    fields = [uh, fsource, err]
    if amrtype == "udobv":
        imark.rename("inactive mark")
        fbmark.rename("free-boundary mark")
        mark.rename("mark")
        eta.rename("eta " + "(BV00)" if useweightedBR else "(BR78)")
        fields += [imark, fbmark, mark, eta]
    if mesh.comm.size > 1:
        # in parallel, write integer-valued element-wise process rank
        DG0 = FunctionSpace(mesh, "DG", 0)
        rank = Function(DG0, name="rank")
        rank.dat.data[:] = mesh.comm.rank
        fields.append(rank)
    VTKFile(outfile).write(*fields)
    print("")

# convergence figures comparing all methods; measured rates are empirically
# fit, since the porous-media operator is degenerate (not uniformly
# elliptic) and so the classical rates are not rigorously justified here;
# we plot them anyway, as references, alongside the measured rates
print("generating .png plots to compare algorithms ...")
if mesh.comm.rank == 0:
    import matplotlib.pyplot as plt

    d = 2  # spatial dimension
    stylemap = {"udobv": "ko", "uni": "rs", "avm": "g^"}

    def _convergence_plot(index, ylabel, title, outfile, rateexp, ratelabel):
        plt.figure()
        for amrtype in refinetypes:
            rdofs, rvals = np.array(results[amrtype][0]), np.array(results[amrtype][index])
            rate = np.polyfit(np.log(rdofs), np.log(rvals), 1)[0]
            plt.loglog(rdofs, rvals, stylemap[amrtype], label=f"{amrtype.upper()} (rate={rate:.2f})")
        # anchor the reference rate to UNI's first data point
        rdofs, rvals = np.array(results["uni"][0]), np.array(results["uni"][index])
        y = rdofs.astype(float) ** rateexp
        y = y * rvals[0] / y[0]
        plt.loglog(rdofs, y, "k:", label=ratelabel)
        plt.legend()
        plt.grid(True)
        plt.xlabel("DOFs")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(outfile)

    _convergence_plot(1, "||u-u_h||_2", "solution norm vs DOFs", "porous_L2.png", -2.0 / d, "DOFs^(-2/d)")
    _convergence_plot(2, "|u-u_h|_{H^1}", "H^1 seminorm error vs DOFs", "porous_H1.png", -1.0 / d, "DOFs^(-1/d)")
    _convergence_plot(3, "quasi-norm error", "weighted quasi-norm error vs DOFs", "porous_qn.png", -2.0 / d, "DOFs^(-2/d)")

    # Hausdorff distance is a geometric (not squared-error) quantity, so the
    # natural reference is mesh-resolution scaling h ~ DOFs^(-1/d), not the
    # L^2/H^1 rates above; as in sphere.py
    hausfile = "porous_hausdorff.png"
    plt.figure()
    for amrtype in refinetypes:
        rdofs = np.array(results[amrtype][0])
        haus_a = np.array(results[amrtype][4])
        valid = np.isfinite(haus_a) & (haus_a > 0)
        if np.any(valid):
            plt.loglog(rdofs, haus_a, stylemap[amrtype], label=amrtype.upper())
    rdofs = np.array(results["uni"][0])
    haus_a = np.array(results["uni"][4])
    valid = np.isfinite(haus_a) & (haus_a > 0)
    if np.any(valid):
        anchor = np.argmax(valid)  # index of first valid entry
        y = rdofs.astype(float) ** (-1.0 / d)
        y = y * haus_a[anchor] / y[anchor]
        plt.loglog(rdofs, y, "k:", label="DOFs^(-1/d)")
    plt.legend()
    plt.grid(True)
    plt.xlabel("DOFs")
    plt.ylabel("hausdorff2D(Gamma_u, Gamma_uh)")
    plt.title("free-boundary Hausdorff distance")
    plt.savefig(hausfile)

    # effectivity index = estimator / true error, in the quasi-norm the
    # weighted BV00 estimator targets; UDOBV only, since UNI/AVM don't
    # compute an a posteriori estimator; linear y-axis, semilogx x-axis,
    # horizontal reference at eff=1
    efffile = "porous_effectivity.png"
    edofs, eff_a = np.array(results["udobv"][0]), np.array(results["udobv"][5])
    plt.figure()
    plt.semilogx(edofs, eff_a, "g^", label="BV00 in quasi-norm")
    plt.axhline(1.0, color="k", linestyle=":", label="ideal eff=1")
    plt.legend()
    plt.grid(True)
    plt.xlabel("DOFs")
    plt.ylabel("effectivity ratio (estimator / true-error)")
    plt.title("effectivity index vs DOFs")
    plt.savefig(efffile)
