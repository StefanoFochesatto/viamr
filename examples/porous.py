# Solve a porous-medium nonlinear obstacle problem, with
# strong form
#   - div(u grad(u)) = f,  u >= 0
# The domain is a square, and u=g on boundary.  The source term
# f(r) is a negative constant, in the center and linear outside
# of that.  The exact solution is known.
#
# The AMR method is UDO+BR.
#
# The solver must handle the degeneracy of the coefficient at
# the free boundary.  Thus the coefficient is regularized with
# decreasing epsilon > 0 values on mesh i,
#   F(u)[v] = (u + eps_i) grad(u) . grad (v) - f * v,
# Also, the initial iterate is raised by eps_i above the obstacle.
# Finally, simple backtracking line search is used.
#
# In this example, even though all data is quite smooth, large
# BR inactive set estimator (=eta) values occur near the free
# boundary.  By comparison, with a ordinary Laplacian that would
# not be true; the solution would be smoother across the
# free boundary.

from firedrake import *
from firedrake.petsc import PETSc

print = PETSc.Sys.Print  # enables correct printing in parallel

from viamr import VIAMR
from pyop2.mpi import MPI  # for MPI reduce in parallel

# major parameters
m0 = 6  # initial mesh is m0 x m0
levels = 7  # number of AMR levels, by
gamma = 2.0  # solves:  - div(u^{gamma-1} grad(u)) = f  porous-media equation, as VI
a, b = 1.2, 1.0  # parameters in building exact solution

# initial mesh
mesh = RectangleMesh(m0, m0, 2.0, 2.0, originX=-2.0, originY=-2.0, diagonal="crossed")

# schedule of eps_i for mesh i; must have length >= levels + 1
epssched = [0.02, 0.01, 0.005, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002]

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
    """Exact radial solution to
    - Div(grad(gamma^-1 u^gamma)) = f."""
    r = sqrt(x * x + y * y)
    tmp = ((a + b) / 2.0) * (0.5 * (r ** 2 - 1.0) - ln(r))
    tmp -= (a / 3.0) * ((r ** 3 - 1.0) / 3 - ln(r))
    return conditional(r <= 1.0, 0.0, (gamma * tmp) ** (1.0 / gamma))


def maxeta(eta):  # maximum of BR estimator, even in parallel
    mine = max(eta.dat.data_ro)
    return float(eta.function_space().mesh().comm.allreduce(mine, op=MPI.MAX))


amr = VIAMR()
for i in range(levels + 1):
    print(f"solving porous-type problem with UDO+BR on mesh {i} ...")
    amr.meshreport(mesh)
    V = FunctionSpace(mesh, "CG", 1)

    if i == 0:
        uh = Function(V, name="u_h")
    else:
        # initialize by cross-mesh interpolation to fine mesh
        uUFL = conditional(uh < lb, lb, uh)  # use old data
        uh = Function(V, name="u_h").interpolate(uUFL)

    # source term which is -2.1 at (0,0) and +3.9 at (1,1)
    x, y = SpatialCoordinate(mesh)
    r = sqrt(x * x + y * y)
    fUFL = conditional(r <= 1.0, -b, -b + a * (r - 1.0))
    fsource = Function(V, name="f_source").interpolate(fUFL)

    # regularization for the problem on this mesh
    eps = epssched[i]

    # initial iterate *above* solution, especially on active set
    uh = Function(V, name="u_h").interpolate(uh + eps)

    # weak form for porous media, with epsilon regularization
    v = TestFunction(V)
    Z = (uh + eps) ** (gamma - 1.0)
    F = Z * inner(grad(uh), grad(v)) * dx - fsource * v * dx

    # solve with u >= 0 and u=g on boundary
    uUFL = get_uexact_ufl(x, y)
    g = Function(V, name="g_bdry").interpolate(uUFL)
    bcs = DirichletBC(V, g, "on_boundary")
    prob = NonlinearVariationalProblem(F, uh, bcs)
    solver = NonlinearVariationalSolver(prob, solver_parameters=sp, options_prefix="s")
    lb = Function(V, name="psi").interpolate(Constant(0.0))
    ub = Function(V).interpolate(Constant(PETSc.INFINITY))
    solver.solve(bounds=(lb, ub))

    # error relative to exact (UFL) solution
    #   (we re-implement errornorm() for L^2, with control on quadrature degree)
    expr = inner(uUFL - uh, uUFL - uh)
    err = assemble(expr * dx(degree=6)) ** (1 / 2)
    print(f"  level {j}: nodes = {V.dim()}, |u-u_h|_2 = {err:.3e}")

    # active set convergence measure by jaccard
    neweactive = amr.elemactive(uh, lb)
    if i > 0:
        jac = amr.jaccard(neweactive, eactive, submesh=True)
        print(f"  Jaccard agreement {100*jac:.2f}% [levels {i-1}, {i}]")
    eactive = neweactive

    # get BR estimator on this mesh, for refinement and reporting
    residual_ufl = -div(uh ** (gamma - 1.0) * grad(uh)) - fsource
    imark, eta, _ = amr.brinactivemark(uh, Constant(0.0), residual_ufl)

    # AMR with UDO+BR
    mark = amr.udomark(uh, lb, n=1)
    mark = amr.unionmarks(mark, imark)
    if i == levels:
        break

    # actually refine mesh
    mesh = amr.refinemarkedelements(mesh, mark)

# generate Paraview-readable file
outfile = "result_porous.pvd"
print(f"done ... writing solution to {outfile} ...")
err = Function(V, name="uerr = u - u_exact").interpolate(uh - uUFL)
imark.rename("inactive set mark")
mark.rename("UDOBR mark")
if mesh.comm.size > 1:
    # in parallel, write integer-valued element-wise process rank
    DG0 = FunctionSpace(mesh, "DG", 0)
    rank = Function(DG0, name="rank")
    rank.dat.data[:] = mesh.comm.rank
    VTKFile(outfile).write(uh, fsource, err, imark, mark, eta, rank)
else:
    VTKFile(outfile).write(uh, fsource, err, imark, mark, eta)
