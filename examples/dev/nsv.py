# from NSV03:
#
#   Nochetto, R. H., Siebert, K. G., & Veeser, A. (2003). Pointwise
#   a posteriori error control for elliptic obstacle problems.
#   Numerische Mathematik, 95(1), 163-195.
#
# first runs UDO+BR refinement for a few levels; optionally one can
#   generate a convergence plot from UDO+BR to compare to the NSV03 results
# then does the following on the final mesh:
#   1. computes sigma_h from section 2.1 in NSV03
#   2. computes the "practical estimator" \eta_\infty and \eta_d in formula (7.1) of NSV03
#   3. solves "7.2 Example: Constant Obstacle"
# TODO implement VIAMR.nsvmark(); use it here in a permanent example; add it to sphere.py

from firedrake import *
from viamr import VIAMR
from firedrake.petsc import PETSc

# major parameters
d = 2  # spatial dimension
m = 3  # initial mesh resolution
levs = 7 if d == 2 else 4  # number of refinements
nUDO = 0  # observe that {sigma_h * u_h > 0} is same as UDO mark with nUDO=0
# primal admissibility requires u_h >= psi_h - primaltol, but note that psi_h=0 here
primaltol = 0.0
dualtol = 1.0e-8  # used for admissibility (sigma_h >= -dualtol) *and* in estimator

# initial mesh
assert d in [2, 3]
if d == 2:
    mesh0 = RectangleMesh(m, m, 1.0, 1.0, originX=-1.0, originY=-1.0, diagonal="crossed")
else:
    # 3D SBR refinement needs Netgen mesh and Netgen refinement (and produces bad meshes)
    from netgen.occ import *

    box = Box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    mesh0 = Mesh(OCCGeometry(box, dim=3).GenerateMesh(maxh=0.8))

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

print = PETSc.Sys.Print  # enables correct printing in parallel
print(f"solving {d}D example from Nochetto, Siebert, & Veeser (2003) ...")
r = 0.7  # parameter in defining problem
results = {}
methods = ["UDOBR", "NSV", "NSVfb"]
for method in methods:
    mesh = mesh0
    dofs, errs = [], []
    for j in range(levs):
        print(f"using AMR by {method} ...")
        x = SpatialCoordinate(mesh)
        V = FunctionSpace(mesh, "CG", 1)

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
        # following admissibility check removes a term from the estimator
        assert min(uh.dat.data_ro) >= 0.0

        # error relative to exact (UFL) solution
        u_ufl = conditional(x2 <= r ** 2, 0.0, circle ** 2)
        dofs.append(V.dim())
        errs.append(float(errornorm(u_ufl, uh)))
        print(f"  level {j}: nodes = {dofs[-1]}, |u-u_h|_2 = {errs[-1]:.3e}")

        # compute marking; note fmark is written to file for comparison
        amr = VIAMR()
        if method == "UDOBR":
            fmark = amr.udomark(uh, psih, n=nUDO)
            residual = -div(grad(uh))
            (imark, _, _) = amr.brinactivemark(uh, psih, residual, theta=0.5)
            mark = amr.unionmarks(fmark, imark)
        else:
            Cfb = 10.0 if method == "NSVfb" else 1.0
            (mark, etainf, sigmah, _) = amr.nsvmark(uh, psih, g, f_ufl, g_ufl, theta=0.5, Cfb=Cfb, dualtol=dualtol)

        # get next mesh by refinement
        if j == levs - 1:
            break
        if d == 2:
            mesh = amr.refinemarkedelements(mesh, mark)  # PETSc DM refinement
        else:
            mesh = mesh.refine_marked_elements(mark)  # Netgen refinement

    # for figure below
    results[method] = (dofs, errs)

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

    outfile = f"result_{method}.pvd"
    print(f"generating output file {outfile} ...")
    if method == "UDOBR":
        fmark.rename("UDO FB mark")
        print(f"writing to {outfile} ...")
        if mesh.comm.size > 1:
            VTKFile(outfile).write(uh, uerr, fmark, active, tactive, rank)
        else:
            VTKFile(outfile).write(uh, uerr, fmark, active, tactive)
    else:
        # FIXME etad could go into VIAMR.nsvmark()?
        # compute *for each closed triangle T* within the thin active set, for formula (7.1):
        #   \eta_d = C1 |h^2 grad(sigmah)|_d
        C1 = 0.01
        sigslope = inner(grad(sigmah), grad(sigmah)) ** (d / 2)  # = |\grad\sigma_h|^d
        _, DG0 = amr.spaces(mesh)
        hT = project(CellSize(mesh), DG0)
        v0 = TestFunction(DG0)
        tmp = assemble(hT ** (2 * d) * sigslope * tactive * v0 * dx).riesz_representation()
        etad = Function(DG0, name="eta_d").interpolate(C1 * tmp ** (1.0 / d))

        if mesh.comm.size > 1:
            VTKFile(outfile).write(uh, uerr, sigmah, etainf, etad, active, tactive, rank)
        else:
            VTKFile(outfile).write(uh, uerr, sigmah, etainf, etad, active, tactive)

# convergence figure
if mesh.comm.rank == 0:
    import matplotlib.pyplot as plt
    import numpy as np

    markers = ["ko", "bs", "rs"]
    for j in range(3):
        meth = methods[j]
        dofs, errs = np.array(results[meth][0]), np.array(results[meth][1])
        #print(np.polyfit(np.log(dofs), np.log(errs), 1))
        plt.loglog(dofs, errs, markers[j], label=meth)
        if meth == "UDOBR":
            y = dofs ** (-2.0 / d)
            y = y * errs[0] / y[0]  # fix constant so that it aligns
            plt.loglog(dofs, y, "k:", label=f"DOFs^(-2/d) for d={d}")
    plt.legend()
    plt.grid(True)
    plt.xlabel("DOFs")
    plt.ylabel("error")
    #plt.title("compare Figure 7.1 in Nochetto, Siebert, & Veeser (2003)")
    plt.show()
