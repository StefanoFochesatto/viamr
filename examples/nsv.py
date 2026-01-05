# example "7.2 Example: Constant Obstacle" from NSV03:
#
#   Nochetto, R. H., Siebert, K. G., & Veeser, A. (2003). Pointwise
#   a posteriori error control for elliptic obstacle problems.
#   Numerische Mathematik, 95(1), 163-195.

from firedrake import *
from viamr import VIAMR
from firedrake.petsc import PETSc

print = PETSc.Sys.Print  # enables correct printing in parallel

# major parameters
d = 2  # spatial dimension
m = 3  # initial mesh resolution
levs = 9 if d == 2 else 6  # number of refinements
figure = False  # generate figure to compare to NSV2003

assert d in [2, 3]
if d == 2:
    mesh = RectangleMesh(m, m, 1.0, 1.0, originX=-1.0, originY=-1.0, diagonal="crossed")
else:
    # 3D SBR refinement needs Netgen mesh and Netgen refinement
    from netgen.occ import *

    box = Box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    mesh = Mesh(OCCGeometry(box, dim=3).GenerateMesh(maxh=0.8))

sp = {
    "snes_type": "vinewtonrsls",
    "snes_converged_reason": None,
    # "snes_monitor": None,
    # "snes_view": None,
    "ksp_type": "preonly",
    # "ksp_converged_reason": None,
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

r = 0.7  # parameter in defining problem
dofs = [None for j in range(levs)]
errs = [None for j in range(levs)]
print(f"solving {d}D example from Nochetto, Siebert, & Veeser (2003) ...")
for j in range(levs):
    x = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)

    if d == 2:
        x2 = x[0] ** 2 + x[1] ** 2
    else:
        x2 = x[0] ** 2 + x[1] ** 2 + x[2] ** 2
    circle = x2 - r ** 2
    g_ufl = circle ** 2
    f_ufl = conditional(
        x2 <= r ** 2, -8.0 * r ** 2 * (1.0 - circle), -4.0 * (2.0 * x2 + d * circle)
    )

    if j == 0:
        uh = Function(V, name="u_h (solution)")
    else:
        # initialize by cross-mesh interpolation; mesh sequencing
        uh = Function(V, name="u_h (solution)").interpolate(uh)

    vh = TestFunction(V)
    F = inner(grad(uh), grad(vh)) * dx - f_ufl * vh * dx
    bcs = DirichletBC(V, Function(V).interpolate(g_ufl), "on_boundary")

    problem = NonlinearVariationalProblem(F, uh, bcs)
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=sp, options_prefix="s"
    )

    psih = Function(V).interpolate(0.0)
    INF = Function(V).interpolate(Constant(PETSc.INFINITY))
    solver.solve(bounds=(psih, INF))

    u_ufl = conditional(x2 <= r ** 2, 0.0, circle ** 2)
    dofs[j] = V.dim()
    errs[j] = float(errornorm(u_ufl, uh))
    print(f"  level {j}: nodes = {dofs[j]}, |u-u_h|_2 = {errs[j]:.3e}")

    # compute marking; note fmark is written to file
    amr = VIAMR()
    fmark = amr.udomark(uh, psih, n=nUDO)
    residual = -div(grad(uh))
    (imark, _, _) = amr.brinactivemark(uh, psih, residual, theta=0.5)
    mark = amr.unionmarks(fmark, imark)

    if j == levs - 1:
        break

    if d == 2:
        mesh = amr.refinemarkedelements(mesh, mark)  # PETSc DM refinement
    else:
        mesh = mesh.refine_marked_elements(mark)  # Netgen refinement

if figure and mesh.comm.rank == 0:
    import matplotlib.pyplot as plt
    import numpy as np

    dofs = np.array(dofs)
    errs = np.array(errs)
    # print(np.polyfit(np.log(dofs), np.log(errs), 1))
    plt.loglog(dofs, errs, "ko", label=r"$\|u - u_h\|_0$")
    y = dofs ** (-2.0 / d)
    y = y * errs[0] / y[0]  # fix constant so that it aligns
    plt.loglog(dofs, y, "k:", label=f"DOFs^(-2/d) for d={d}")
    plt.legend()
    plt.grid(True)
    plt.xlabel("DOFs")
    plt.ylabel("error")
    plt.title("compare Figure 7.1 in Nochetto, Siebert, & Veeser (2003)")
    plt.show()

P2 = FunctionSpace(mesh, "CG", 2)
udiff = Function(P2, name="u_h - u_exact").interpolate(uh - u_ufl)
f = Function(P2, name="f").interpolate(f_ufl)

outfile = "result_nsv.pvd"
print(f"writing to {outfile} ...")
if mesh.comm.size > 1:
    rank = Function(FunctionSpace(mesh, "DG", 0))
    rank.dat.data[:] = mesh.comm.rank
    rank.rename("rank")
    VTKFile(outfile).write(uh, udiff, f, rank)
else:
    VTKFile(outfile).write(uh, udiff, f)
