# example "7.2 Example: Constant Obstacle" from
#   Nochetto, R. H., Siebert, K. G., & Veeser, A. (2003). Pointwise
#   a posteriori error control for elliptic obstacle problems.
#   Numerische Mathematik, 95(1), 163-195.

from firedrake import *
from viamr import VIAMR

d = 2
m = 100   # FIXME hi-res for debug

assert d == 2, "implement 3D later"

mesh = RectangleMesh(m, m, 1.0, 1.0, originX=-1.0, originY=-1.0)
x, y = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "CG", 1)

r = 0.7
x2 = x ** 2 + y ** 2
circle = x2 - r ** 2
g_ufl = circle ** 2

u_ufl = (conditional(x2 <= r ** 2, 0.0, circle)) ** 2
uexact = Function(V, name="uexact").interpolate(u_ufl)

f_ufl = conditional(x2 <= r ** 2,
                    - 8.0 * r ** 2 * (1.0 - circle),
                    - 4.0 * (2.0 * x2 + d * circle))
f = Function(V, name="f").interpolate(f_ufl)

# uh, vh = Function(V), TestFunction(V)
# F = inner(grad(uh), grad(vh)) * dx - Constant(-1) * vh * dx
# bcs = DirichletBC(V, Function(V).interpolate(uexact), "on_boundary")
# problem = NonlinearVariationalProblem(F, uh, bcs)

# sp = {"snes_type": "vinewtonrsls"}
# solver = NonlinearVariationalSolver(problem, solver_parameters=sp)
# psih = Function(V).interpolate(0.0)
# INF = Function(V).interpolate(Constant(PETSc.INFINITY))
# solver.solve(bounds=(psih, INF))

# amr = VIAMR()
# mark = amr.vcdmark(uh, psih)
# VTKFile("mesh.pvd").write(uh, mark)

# refinedmesh = amr.refinemarkedelements(mesh, mark)

VTKFile("result.pvd").write(f, uexact)
