# Compare n=0 UDO+BR to the method from
#
#   Nochetto, R. H., Siebert, K. G., & Veeser, A. (2003). Pointwise
#   a posteriori error control for elliptic obstacle problems.
#   Numerische Mathematik, 95(1), 163-195.
#
# on "7.2 Example: Constant Obstacle".  Generates convergence and effectivity
# .png figures.

# TODO
#   add figure comparing NSV05 estimate of hausdorff to computed hausdorf (2D only)

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
    max; same technique nsv03mark() itself uses internally for non-polynomial
    data (e.g. its bdryerr term).  This is the norm NSV03's "pointwise a
    posteriori error control" theory targets, in contrast to BR78/BV00's
    energy (H^1 seminorm) norm."""
    W = FunctionSpace(uh.function_space().mesh(), "CG", 4)
    err = Function(W).interpolate(abs(u - uh))
    return amr.scalarrange(err)[1]


print(f"solving example 7.2 ({d}D case) from Nochetto, Siebert, & Veeser (2003) ...")
r = 0.7  # parameter in defining problem
results = {}
methods = ["UDOBR", "NSV03", "NSV05"]
for method in methods:
    print("")
    mesh = mesh0
    dofs, errsl2, errsinf, errsH1ia, ests, eff_vals = [], [], [], [], [], []
    for j in range(maxlevs):
        print(f"using AMR by {method} method ...")
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

        # compute error norms, disregarding relevance to method
        u_ufl = conditional(x2 <= r ** 2, 0.0, circle ** 2)
        dofs.append(V.dim())
        errsl2.append(float(errornorm(u_ufl, uh)))  # default norm is L^2
        errsinf.append(float(errornorm_Linf(amr, u_ufl, uh)))
        iamark = amr.eleminactive(uh, psih, strong=True)
        dus = inner(grad(u_ufl - uh), grad(u_ufl - uh))
        errsH1ia.append(assemble(dus * iamark * dx(degree=6)) ** 0.5)
        print(f"  level {j}: nodes = {dofs[-1]}, |u-u_h|_2 = {errsl2[-1]:.3e}, |u-u_h|_inf = {errsinf[-1]:.3e}, ")

        # marking and effectivity
        if method == "UDOBR":
            fmark = amr.udomark(uh, psih, n=nUDO)
            residual = -div(grad(uh)) - f_ufl
            (imark, _, tot_eta) = amr.brinactivemark(uh, psih, residual, theta=0.5, method="total")
            mark = amr.unionmarks(fmark, imark)
            ests.append(tot_eta)
            # effectivity index vs the inactive set H^1 seminorm BR78 estimator targets
            eff = tot_eta / errsH1ia[-1] if errsH1ia[-1] > 0 else float("nan")
            print(f"  eff_BR (inactive-set, energy norm) = {eff:.3f}")
        elif method == "NSV03":
            (mark, etainf, etad, sigmah, Eh) = amr.nsv03mark(
                uh, psih, g, f_ufl, g_ufl, theta=0.5, dualtol=dualtol, method="total"
            )
            ests.append(Eh)
            # effectivity index vs NSV03's ||u-u_h||_infty target norm: Eh is the
            # whole-domain estimator Etilde_h of (7.1)
            eff = Eh / errsinf[-1] if errsinf[-1] > 0 else float("nan")
            print(f"  eff_NSV03 (sup norm) = {eff:.3f}")
        elif method == "NSV05":
            (mark, eta, sz, fullcontact, Eh) = amr.nsv05mark(
                uh, psih, g, f_ufl, g_ufl, theta=0.5, dualtol=dualtol, method="total"
            )
            ests.append(Eh)
            # effectivity index vs NSV05's ||u-u_h||_infty target norm
            eff = Eh / errsinf[-1] if errsinf[-1] > 0 else float("nan")
            print(f"  eff_NSV05 (sup norm) = {eff:.3f}")
        else:
            raise NotImplementedError
        eff_vals.append(eff)

        # done with this method if we reach target complexity
        if dofs[-1] > targetnodes:
            break

        # get next mesh by refinement
        if d == 2:
            mesh = amr.refinesbr2D(mesh, mark)  # PETSc DM refinement
        else:
            mesh = mesh.refine_marked_elements(mark)  # Netgen refinement

    # for figures below
    results[method] = (dofs, errsl2, errsinf, errsH1ia, ests, eff_vals)

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
    elif method == "NSV03":
        fields += [sigmah, etainf, etad]
    elif method == "NSV05":
        fields += [eta, sz, fullcontact]
    if mesh.comm.size > 1:
        fields.append(rank)
    VTKFile(outfile).write(*fields)

# convergence figure
if mesh.comm.rank == 0:
    import matplotlib.pyplot as plt
    import numpy as np

    print(f"generating convergence and effectivity figures nsv_*.png ...")
    markers = ["ko", "bs", "md"]

    figfile = "nsv_convergence_l2.png"
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
    plt.title("L^2 errors")
    plt.savefig(figfile)

    figfile = "nsv_convergence_h1ia.png"
    plt.figure()
    for j in range(len(methods)):
        meth = methods[j]
        dofs, errs, ests = np.array(results[meth][0]), np.array(results[meth][3]), np.array(results[meth][4])
        #print(np.polyfit(np.log(dofs), np.log(errs), 1))
        plt.loglog(dofs, errs, markers[j], label=meth+' error')
        if meth == "UDOBR":
            plt.loglog(dofs, ests, markers[j], markerfacecolor='white', label=meth+' estimator')
        if j == len(methods) - 1:
            y = dofs ** (-1.0 / d)
            y = y * errs[0] / y[0]  # fix constant so that it aligns
            plt.loglog(dofs, y, "k:", label=f"DOFs^(-1/d) for d={d}")
    plt.legend()
    plt.grid(True)
    plt.xlabel("DOFs")
    plt.ylabel("|u-u_h|_H1 inactive")
    plt.title("H^1 errors restricted to inactive set")
    plt.savefig(figfile)

    figfile = "nsv_convergence_linf.png"
    plt.figure()
    for j in range(len(methods)):
        meth = methods[j]
        dofs, errs, ests = np.array(results[meth][0]), np.array(results[meth][2]), np.array(results[meth][4])
        #print(np.polyfit(np.log(dofs), np.log(errs), 1))
        plt.loglog(dofs, errs, markers[j], label=meth+' error')
        if meth in ("NSV03", "NSV05"):
            plt.loglog(dofs, ests, markers[j], markerfacecolor='white', label=meth+' estimator')
        if j == len(methods) - 1:
            y = dofs ** (-2.0 / d)
            y = y * errs[0] / y[0]  # fix constant so that it aligns
            plt.loglog(dofs, y, "k:", label=f"DOFs^(-2/d) for d={d}")
    plt.legend()
    plt.grid(True)
    plt.xlabel("DOFs")
    plt.ylabel("norm error |u-u_h|_inf")
    plt.title("L^inf errors (compare Figure 7.1 top in NSV03)")
    plt.savefig(figfile)

    # effectivity index = estimator / true error, in whichever norm the
    # estimator targets (energy norm for BR78, sup norm for NSV03); unlike
    # the norm-convergence figure above, there's no power-law rate to plot
    # against, since a reliable+efficient estimator should stay O(1) (ideally
    # near 1) as DOFs grow, so this uses a linear y-axis with a semilogx
    # x-axis, and a horizontal reference line at the ideal value of 1
    efffile = "nsv_effectivity.png"
    effstylemap = {"UDOBR": ("ko", "BR78 (inactive-set, energy norm)"),
                   "NSV03": ("bs", "NSV03 (sup norm)"),
                   "NSV05": ("md", "NSV05 (sup norm)")}
    plt.figure()
    for meth in methods:
        edofs, evals = np.array(results[meth][0]), np.array(results[meth][5])
        style, label = effstylemap[meth]
        plt.semilogx(edofs, evals, style, label=label)
    plt.axhline(1.0, color="k", linestyle=":", label="ideal eff=1")
    plt.legend()
    plt.grid(True)
    plt.xlabel("DOFs")
    plt.ylabel("effectivity ratio (estimator / true-error)")
    plt.title("effectivity index vs DOFs")
    plt.savefig(efffile)
