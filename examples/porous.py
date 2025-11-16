# Solve a porous-medium nonlinear obstacle problem, with
# strong form
#   - div(u grad(u)) = f,  u >= 0
# The domain is the unit square, and u=g on boundary.  The
# source term f(x,y) is negative in lower left and positive in
# upper right.
#
# The solver must handle the degeneracy of the coefficient at
# the free boundary.  Thus the coefficient is regularized with
# decreasing epsilon > 0 values on mesh i,
#   (u + eps_i) grad(u) . grad (v) - f * v,
# and the initial iterate (for mesh i) is raised by eps_i
# above the obstacle.  Additionally, simple backtracking line
# search is appropriate.
#
# In this example, even though all data is smooth, the large
# BR inactive set estimator (=eta) values are near the free
# boundary.  By comparison, with a ordinary Laplacian that would
# not be true; the solution would be pretty smooth across the
# free boundary.

from firedrake import *
from firedrake.petsc import PETSc

print = PETSc.Sys.Print  # enables correct printing in parallel
from viamr import VIAMR
from pyop2.mpi import MPI  # for MPI reduce in parallel

m0 = 10
levels = 4
laplacian = False   # set to True to compare ordinary Poisson equation

# schedule of eps_i for mesh i; must have length >= levels + 1
epssched = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]

# setting distribution parameters should not be necessary ... but bug in netgen
dp = {
    "partition": True,
    "overlap_type": (DistributedMeshOverlapType.VERTEX, 1),
}
mesh = UnitSquareMesh(m0, m0, distribution_parameters=dp)

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

def maxeta(eta):  # maximum of BR estimator, even in parallel
    mine = max(eta.dat.data_ro)
    return float(eta.function_space().mesh().comm.allreduce(mine, op=MPI.MAX))

for i in range(levels + 1):
    print(f"solving porous-type problem with UDO on mesh {i} ...")
    V = FunctionSpace(mesh, "CG", 1)

    if i == 0:
        uh = Function(V, name="u_h")
    else:
        # initialize by cross-mesh interpolation to fine mesh
        uUFL = conditional(uh < lb, lb, uh)  # use old data
        uh = Function(V, name="u_h").interpolate(uUFL)

    amr = VIAMR()
    amr.meshreport(mesh)

    # source term which is -2.1 at (0,0) and +3.9 at (1,1)
    x, y = SpatialCoordinate(mesh)
    fsource = Function(V).interpolate(3.0 * (x * x + y * y - 0.7))

    # regularization for the problem on this mesh
    eps = epssched[i]

    # initial iterate *above* solution, especially on active set
    uh = Function(V, name="u_h").interpolate(uh + eps)

    # weak form for gamma = 1 porous medium, with epsilon regularization
    v = TestFunction(V)
    if laplacian:
        F = inner(grad(uh), grad(v)) * dx - fsource * v * dx
    else:
        F = (uh + eps) * inner(grad(uh), grad(v)) * dx - fsource * v * dx

    # solve with u >= 0 and u=g on boundary
    # note 0 <= g <= 0.2 with only g(0,0)=0
    g = 0.1 * (x + y)
    bcs = DirichletBC(V, g, "on_boundary")
    problem = NonlinearVariationalProblem(F, uh, bcs)
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=sp, options_prefix="s"
    )
    lb = Function(V, name="psi").interpolate(Constant(0.0))
    ub = Function(V).interpolate(Constant(PETSc.INFINITY))
    solver.solve(bounds=(lb, ub))

    # active set convergence measure by jaccard
    neweactive = amr.elemactive(uh, lb)
    if i > 0:
        jac = amr.jaccard(neweactive, eactive, submesh=True)
        print(f"  Jaccard agreement {100*jac:.2f}% [levels {i-1}, {i}]")
    eactive = neweactive

    # get BR estimator on this mesh, for refinement and reporting
    # for uh in CG1, residual_ufl = -fsource
    if laplacian:
        residual_ufl = -div(grad(uh)) - fsource
    else:
        residual_ufl = -div(uh * grad(uh)) - fsource
    imark, eta, _ = amr.brinactivemark(uh, Constant(0.0), residual_ufl)
    print(f"  eta <= {maxeta(eta):.3e}")

    if i == levels:
        break

    # AMR with UDO+BR
    # optional: uniform-in-inactive refinement, on every third mesh
    # if i % 3 == 1:
    #     imark = amr.eleminactive(uh, lb)
    mark = amr.udomark(uh, lb, n=1)
    mark = amr.unionmarks(mark, imark)
    mesh = amr.refinemarkedelements(mesh, mark)

# generate Paraview-readable file
outfile = "result_porous.pvd"
print(f"done ... writing u_h to {outfile} ...")
if mesh.comm.size > 1:
    # in parallel, write integer-valued element-wise process rank
    DG0 = FunctionSpace(mesh, "DG", 0)
    rank = Function(DG0, name="rank")
    rank.dat.data[:] = mesh.comm.rank
    VTKFile(outfile).write(uh, eta, rank)
else:
    VTKFile(outfile).write(uh, eta)
