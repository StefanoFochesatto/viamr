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
levs = 4 if d == 2 else 3  # number of refinements
nUDO = 1  # for nUDO = 0: observe that sigma_h * u_h > 0 is same as UDO mark
figure = False  # generate figure to compare to NSV03

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

# compute some quantities for output file
fmark.rename("UDO FB mark")
uerr = Function(V, name="u_err = u_h - u_exact").interpolate(uh - u_ufl)
P3 = FunctionSpace(mesh, "CG", 3)
f = Function(P3, name="f").interpolate(f_ufl)

# following section 2.1 of NSV03, compute residual sigmah in V=P1
#   observe this measures how close to complementarity we are
#   noting psih=0, complementarity would be
#     uh >= 0,  sigmah >= 0,  uh sigmah = 0
#   here using opposite sign from NSV03
# create cofunction with values int_Omega phi_i dx for *all* nodes i
phi = TestFunction(V)
scaleh = assemble(phi * dx)  # cofunction
# compute sigma_h
sigmah = Function(V, name="sigma_h (residual)")
res = assemble((inner(grad(uh), grad(phi)) - f_ufl * phi) * dx)  # cofunction
sigmah.dat.data[:] = res.dat.data_ro / scaleh.dat.data_ro  # divide numpy arrays
# correct it on boundary; note all boundary nodes are inactive in this example
# FIXME section 2.1 of NSV03 addresses cases where boundary nodes are active
#    perhaps use:  n = FacetNormal(mesh); ?? inner(grad(uh), n) * omegah * dS
DirichletBC(V, Constant(0.0), "on_boundary").apply(sigmah)

# FIXME: following is playing around, but in the right way
# compute DG0 field where  sigmah_k * uh_k > 0  at DG0 degree of freedom k
DG0 = FunctionSpace(mesh, "DG", 0)
noncomph = Function(DG0, name="sigma_h u_h > 0}")
activetol = 1.0e-10
noncomph.interpolate(conditional(sigmah * uh > activetol, 1.0, 0.0))

outfile = "result_nsv.pvd"
print(f"writing to {outfile} ...")
if mesh.comm.size > 1:
    rank = Function(FunctionSpace(mesh, "DG", 0))
    rank.dat.data[:] = mesh.comm.rank
    rank.rename("rank")
    VTKFile(outfile).write(uh, uerr, f, sigmah, noncomph, fmark, rank)
else:
    VTKFile(outfile).write(uh, uerr, f, sigmah, noncomph, fmark)
