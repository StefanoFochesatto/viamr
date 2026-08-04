# Solve a porous-medium nonlinear obstacle problem, with strong form
#   - div(u^{gamma-1} grad(u)) = f,  u >= 0
# The domain is a square, and u=g on boundary.  The source term f(r)
# is a negative constant in a center disc, and linear outside of that.
# Defaults to gamma=2.  The exact solution is known for any gamma.
#
# The solver must handle the degeneracy of the coefficient at
# the free boundary.  Thus the coefficient is regularized with
# decreasing epsilon > 0 values on mesh i,
#   F(u)[v] = (u + eps_i)^{gamma-1} grad(u) . grad (v) - f * v,
# Also, the Newton initial iterate is raised eps_i above the obstacle.
# Simple backtracking line search is used for the Newton solver.
#
# The AMR method is UDO+BR.  By default we use the (heuristic)
# weighted-norm Bernardi & Verfurth (2000) modification of the
# BR78 estimator.  We report the weighted quasi-norm error.
# Without the BV00 weighting, very large BR inactive set estimator
# (=eta) values occur near the free boundary.

from firedrake import *
from firedrake.petsc import PETSc

print = PETSc.Sys.Print  # enables correct printing in parallel

from viamr import VIAMR
from pyop2.mpi import MPI  # for MPI reduce in parallel

# major parameters
m0 = 6  # initial mesh is m0 x m0
levels = 7  # number of AMR levels; converges to 9, at least, for gamma=2
gamma = 2.0  # solves porous-media as VI:  - div(u^{gamma-1} grad(u)) = f
a, b = 1.2, 1.0  # parameters in building exact solution
useweightedBR = True  # when computing eta from brinactivemark(), use weighting

# schedule of eps_i for mesh i; must have length >= levels + 1
eps = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005]

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
    """Exact radial solution to  - Div(grad(gamma^-1 u^gamma)) = f."""
    r = sqrt(x * x + y * y)
    tmp = ((a + b) / 2.0) * (0.5 * (r ** 2 - 1.0) - ln(r))
    tmp -= (a / 3.0) * ((r ** 3 - 1.0) / 3 - ln(r))
    return conditional(r <= 1.0, 0.0, (gamma * tmp) ** (1.0 / gamma))


def maxeta(mesh, eta):  # maximum of BR estimator, even in parallel
    mine = max(eta.dat.data_ro)
    return float(eta.function_space().mesh().comm.allreduce(mine, op=MPI.MAX))


print(f"solving porous-medium obstacle problem using UDO+BR ...")
mesh = RectangleMesh(m0, m0, 2.0, 2.0, originX=-2.0, originY=-2.0, diagonal="crossed")
amr = VIAMR()
for i in range(levels + 1):
    # initialize on this mesh
    V = FunctionSpace(mesh, "CG", 1)
    print(f"mesh {i}:")
    amr.meshreport(mesh)

    # initial iterate *above* solution, especially on active set
    if i == 0:
        uh = Function(V, name="u_h").interpolate(Constant(eps[i]))
    else:  # cross-mesh interpolation to new mesh
        uUFL = conditional(uh < lb, lb, uh)  # uses old data uh, lb
        uh = Function(V, name="u_h").interpolate(uUFL + Constant(eps[i]))

    # radial source term which is constant then linear
    x, y = SpatialCoordinate(mesh)
    r = sqrt(x * x + y * y)
    fUFL = conditional(r <= 1.0, -b, -b + a * (r - 1.0))
    fsource = Function(V, name="f_source").interpolate(fUFL)

    # weak form for porous media, with epsilon regularization
    v = TestFunction(V)
    Z = (uh + Constant(eps[i])) ** (gamma - 1.0)
    F = Z * inner(grad(uh), grad(v)) * dx - fsource * v * dx

    # boundary condition u=g needs exact solution
    uUFL = get_uexact_ufl(x, y)
    g = Function(V, name="g_bdry").interpolate(uUFL)
    bcs = DirichletBC(V, g, "on_boundary")

    # solve with u >= 0
    prob = NonlinearVariationalProblem(F, uh, bcs)
    solver = NonlinearVariationalSolver(prob, solver_parameters=sp, options_prefix="s")
    lb = Function(V, name="psi").interpolate(Constant(0.0))
    ub = Function(V).interpolate(Constant(PETSc.INFINITY))
    solver.solve(bounds=(lb, ub))

    # errors relative to exact; we re-implement errornorm() with fixed
    # quadrature degree; also get weighted quasi-norm
    errL2 = assemble(inner(uUFL - uh, uUFL - uh) * dx(degree=6)) ** (1 / 2)
    dus = inner(grad(uUFL - uh), grad(uUFL - uh))
    errH1 = assemble(dus * dx(degree=6)) ** (1 / 2)
    hT = CellDiameter(mesh)
    CQN = 1.0
    Zqn = (uh + CQN * hT) ** (gamma - 1.0)  # well-behaved mesh-native version of Z
    errqn = assemble(Zqn * dus * dx(degree=6)) ** (1 / 2)

    # active set agreement by jaccard
    neweactive = amr.elemactive(uh, lb)
    if i > 0:
        jac = amr.jaccard(neweactive, eactive, submesh=True)
    eactive = neweactive

    # mark and refine by UDO+BR; tot_eta from this is used in reported effectivity index
    Zexact = uh ** (gamma - 1.0)
    res = -div(Zexact * grad(uh)) - fsource
    if not useweightedBR:
        Zqn = None
    imark, eta, tot_eta = amr.brinactivemark(uh, Constant(0.0), res, kappa=Zqn)
    eff = tot_eta / (errqn if useweightedBR else errH1)
    fbmark = amr.udomark(uh, lb, n=1)
    mark = amr.unionmarks(fbmark, imark)

    # report errors, effectivity index, Jaccard agreement
    print(f"  ||u-u_h||_L2={errL2:.3e};  |u-u_h|_H1={errH1:.3e};  |u-u_h|_qn={errqn:.3e}")
    print(f"  eff={eff:.2f}" + (f";  Jaccard({i-1},{i})={100*jac:.2f}%" if i > 0 else ""))

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
