# Compare n=0 UDO+BR to the method from
#
#   Nochetto, R. H., Siebert, K. G., & Veeser, A. (2003). Pointwise
#   a posteriori error control for elliptic obstacle problems.
#   Numerische Mathematik, 95(1), 163-195.
#
# on "7.2 Example: Constant Obstacle".  Generates convergence and effectivity
# .png figures.

# major parameters
d = 2  # spatial dimension
targetnodes = 5e4 if d == 2 else 2e4  # target number of nodes; less in 3D because matrices
maxlevs = 12  # backstop targetnodes
nUDO = 0  # observe that {sigma_h * u_h > 0} is same as UDO mark with nUDO=0
dualtol = 1.0e-8  # used for admissibility (sigma_h >= -dualtol) *and* in estimator

from firedrake import *
from viamr import VIAMR
from firedrake.petsc import PETSc

print = PETSc.Sys.Print  # enables correct printing in parallel

# initial mesh
assert d in [2, 3]
if d == 2:
    m = 3  # initial mesh resolution
    mesh0 = RectangleMesh(
        m,
        m,
        1.0,
        1.0,
        originX=-1.0,
        originY=-1.0,
        diagonal="crossed",
        distribution_parameters=VIAMR.PARALLEL_OVERLAP,
    )
else:
    # 3D SBR refinement needs Netgen mesh and Netgen refinement (and produces bad meshes)
    from netgen.occ import *

    box = Box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    h0 = 0.5  # initial mesh resolution
    ngmesh = OCCGeometry(box, dim=3).GenerateMesh(maxh=h0)
    mesh0 = Mesh(ngmesh, distribution_parameters=VIAMR.PARALLEL_OVERLAP)

# all methods use same VI solver
sp = {
    "snes_type": "vinewtonrsls",
    "snes_converged_reason": None,
    # "snes_monitor": None,
    # "snes_view": None,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def errornorm_Linf(amr, u, uh):
    """Approximate sup-norm (L^infty) error, via interpolation of the
    (generally non-polynomial) exact-minus-computed difference into a
    higher-degree CG space, then VIAMR.scalarrange() for a parallel-safe
    max; same technique nsvmark() itself uses internally for non-polynomial
    data (e.g. its bdryerr term).  This is the norm NSV03's "pointwise a
    posteriori error control" theory targets, in contrast to BR78/BV00's
    energy (H^1 seminorm) norm."""
    W = FunctionSpace(uh.function_space().mesh(), "CG", 4)
    err = Function(W).interpolate(abs(u - uh))
    return amr.scalarrange(err)[1]


print(f"solving {d}D example from Nochetto, Siebert, & Veeser (2003) ...")
r = 0.7  # parameter in defining problem
results = {}
effs = {}  # effs[method] = (eff_dofs, eff_vals), for the effectivity-index figure
methods = ["UDOBR", "NSV"]
for method in methods:
    mesh = mesh0
    dofs, errs, errsinf = [], [], []
    eff_dofs, eff_vals = [], []
    for j in range(maxlevs):
        print(f"using AMR by {method} ...")
        x = SpatialCoordinate(mesh)
        V = FunctionSpace(mesh, "CG", 1)
        amr = VIAMR(debug=True)

        # UFL expressions for source function and boundary values
        if d == 2:
            x2 = x[0] ** 2 + x[1] ** 2
        else:
            x2 = x[0] ** 2 + x[1] ** 2 + x[2] ** 2
        circle = x2 - r ** 2
        f_ufl = conditional(
            x2 <= r ** 2, -8.0 * r ** 2 * (1.0 - circle), -4.0 * (2.0 * x2 + d * circle)
        )
        g_ufl = circle ** 2  # note this is quartic, so P4 interpolation should be exact

        # initialize by cross-mesh interpolation, i.e. do mesh sequencing
        uh = Function(V, name="u_h (solution)").interpolate(uh if j > 0 else 0.0)

        # state the problem
        vh = TestFunction(V)
        F = inner(grad(uh), grad(vh)) * dx - f_ufl * vh * dx
        g = Function(V).interpolate(g_ufl)  # = I_h g in NSV03
        bcs = DirichletBC(V, g, "on_boundary")
        problem = NonlinearVariationalProblem(F, uh, bcs)
        psih = Function(V).interpolate(0.0)
        INFupper = Function(V).interpolate(Constant(PETSc.INFINITY))

        # solve the problem
        solver = NonlinearVariationalSolver(
            problem, solver_parameters=sp, options_prefix="s"
        )
        solver.solve(bounds=(psih, INFupper))

        # error relative to exact (UFL) solution
        u_ufl = conditional(x2 <= r ** 2, 0.0, circle ** 2)
        dofs.append(V.dim())
        errs.append(float(errornorm(u_ufl, uh)))  # default norm is L^2
        errsinf.append(float(errornorm_Linf(amr, u_ufl, uh)))  # L^inf too
        print(f"  level {j}: nodes = {dofs[-1]}, |u-u_h|_2 = {errs[-1]:.3e},  |u-u_h|_inf = {errsinf[-1]:.3e}, ")

        # compute marking; note fmark is written to file for comparison
        if method == "UDOBR":
            fmark = amr.udomark(uh, psih, n=nUDO)
            residual = -div(grad(uh)) - f_ufl
            (imark, _, tot_eta) = amr.brinactivemark(uh, psih, residual, theta=0.5, method="total")
            mark = amr.unionmarks(fmark, imark)
            # effectivity index vs the SAME inactive set brinactivemark()
            # restricts its estimator to; matches the H^1 seminorm BR78's
            # unweighted estimator targets
            iamark = amr.eleminactive(uh, psih, strong=True)
            dus = inner(grad(u_ufl - uh), grad(u_ufl - uh))
            errH1_inactive = assemble(dus * iamark * dx(degree=6)) ** 0.5
            eff_br = tot_eta / errH1_inactive if errH1_inactive > 0 else float("nan")
            print(f"  eff_BR (inactive-set, energy norm) = {eff_br:.3f}")
            eff_dofs.append(dofs[-1])
            eff_vals.append(eff_br)
        else:
            (mark, etainf, sigmah, Eh, etad) = amr.nsvmark(
                uh, psih, g, f_ufl, g_ufl, theta=0.5, dualtol=dualtol, method="total"
            )
            # effectivity index vs NSV03's own target norm: Eh is the estimator
            # Etilde_h of (7.1), the whole-domain (not inactive-set-restricted)
            # quantity its pointwise theory bounds ||u-u_h||_infty by, in contrast
            # to BR78/BV00's energy-norm estimator above
            errLinf = errornorm_Linf(amr, u_ufl, uh)
            eff_nsv = Eh / errLinf if errLinf > 0 else float("nan")
            print(f"  eff_NSV (sup norm) = {eff_nsv:.3f}")
            eff_dofs.append(dofs[-1])
            eff_vals.append(eff_nsv)

        # done with this method if we reach target complexity
        if dofs[-1] > targetnodes:
            break

        # get next mesh by refinement
        if d == 2:
            mesh = amr.refinesbr2D(mesh, mark)  # PETSc DM refinement
        else:
            mesh = mesh.refine_marked_elements(mark)  # Netgen refinement

    # for figures below
    results[method] = (dofs, errs, errsinf)
    effs[method] = (eff_dofs, eff_vals)

    # compute fields on final mesh (independent of method)
    uerr = Function(V, name="u_err = u_h - u_exact").interpolate(uh - u_ufl)
    active = amr.elemactive(uh, psih)
    active.rename("active")
    tactive = amr.thinelemactive(uh, psih)
    tactive.rename("thin active")
    if mesh.comm.size > 1:
        rank = Function(FunctionSpace(mesh, "DG", 0))
        rank.dat.data[:] = mesh.comm.rank
        rank.rename("rank")

    # write Paraview-readable output
    outfile = f"result_{method}.pvd"
    print(f"generating output file {outfile} ...")
    fields = [uh, uerr, active, tactive, mark]
    if method == "UDOBR":
        imark.rename("UDO inactive mark")
        fmark.rename("UDO FB mark")
        fields += [imark, fmark]
    else:
        fields += [sigmah, etainf, etad]
    if mesh.comm.size > 1:
        fields.append(rank)
    VTKFile(outfile).write(*fields)

# convergence figure
if mesh.comm.rank == 0:
    import matplotlib.pyplot as plt
    import numpy as np

    print(f"generating convergence and effectivity figures nsv_*.png ...")

    figfile = "nsv_convergence_l2.png"
    markers = ["ko", "bs"]
    plt.figure()
    for j in range(len(methods)):
        meth = methods[j]
        dofs, errs = np.array(results[meth][0]), np.array(results[meth][1])
        #print(np.polyfit(np.log(dofs), np.log(errs), 1))
        plt.loglog(dofs, errs, markers[j], label=meth)
        if j == len(methods) - 1:
            y = dofs ** (-2.0 / d)
            y = y * errs[0] / y[0]  # fix constant so that it aligns
            plt.loglog(dofs, y, "k:", label=f"DOFs^(-2/d) for d={d}")
    plt.legend()
    plt.grid(True)
    plt.xlabel("DOFs")
    plt.ylabel("norm error |u-u_h|_2")
    plt.title("compare Figure 7.1 in Nochetto, Siebert, & Veeser (2003)")
    plt.savefig(figfile)

    figfile = "nsv_convergence_linf.png"
    markers = ["ko", "bs"]
    plt.figure()
    for j in range(len(methods)):
        meth = methods[j]
        dofs, errs = np.array(results[meth][0]), np.array(results[meth][2])
        #print(np.polyfit(np.log(dofs), np.log(errs), 1))
        plt.loglog(dofs, errs, markers[j], label=meth)
        if j == len(methods) - 1:
            y = dofs ** (-2.0 / d)
            y = y * errs[0] / y[0]  # fix constant so that it aligns
            plt.loglog(dofs, y, "k:", label=f"DOFs^(-2/d) for d={d}")
    plt.legend()
    plt.grid(True)
    plt.xlabel("DOFs")
    plt.ylabel("norm error |u-u_h|_inf")
    plt.savefig(figfile)

    # effectivity index = estimator / true error, in whichever norm the
    # estimator targets (energy norm for BR78, sup norm for NSV03); unlike
    # the norm-convergence figure above, there's no power-law rate to plot
    # against, since a reliable+efficient estimator should stay O(1) (ideally
    # near 1) as DOFs grow, so this uses a linear y-axis with a semilogx
    # x-axis, and a horizontal reference line at the ideal value of 1
    efffile = "nsv_effectivity.png"
    effstylemap = {"UDOBR": ("ko", "BR78 (inactive-set, energy norm)"),
                    "NSV": ("bs", "NSV03 (sup norm)")}
    plt.figure()
    for meth in methods:
        edofs, evals = np.array(effs[meth][0]), np.array(effs[meth][1])
        style, label = effstylemap[meth]
        plt.semilogx(edofs, evals, style, label=label)
    plt.axhline(1.0, color="k", linestyle=":", label="ideal eff=1")
    plt.legend()
    plt.grid(True)
    plt.xlabel("DOFs")
    plt.ylabel("effectivity ratio (estimator / true-error)")
    plt.title("effectivity index vs DOFs")
    plt.savefig(efffile)
