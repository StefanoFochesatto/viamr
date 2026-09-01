# Solve 3D advection-diffusion problem with upper and lower bounds:
#   0 <= u <= 1.
# This problem is the d=3 version of Example 8.3 in
#    E. Bueler and P. Farrell (2024). A full approximation scheme multilevel
#    method for nonlinear variational inequalities, SIAM J. Sci. Comput. 46,
#    A2421–A2444, https://doi.org/10.1137/23M1594200

from firedrake.petsc import PETSc

print = PETSc.Sys.Print  # enables correct printing in parallel
from firedrake import *
from viamr import VIAMR
from netgen.occ import *

refinements = 2

# initial mesh
box = Box((-1, -1, -1), (1, 1, 1))
ngmesh = OCCGeometry(box, dim=3).GenerateMesh(maxh=0.4)
mesh = Mesh(ngmesh, distribution_parameters=VIAMR.PARALLEL_OVERLAP)

# parameters for constructing the wind and source term
alpha = 30.0  # strength of positive source
beta = 4.0  # strength of negative removal
x0, y0, z0, r0 = -0.5, 0.5, 0.5, 1.0 / 3.0
x1, y1, z1, r1 = -0.5, -0.5, -0.2, 1.0 / 5.0

# do AMR on the VI
amr = VIAMR()
for i in range(refinements + 1):
    print(f"solving on mesh {i} ...")
    amr.meshreport(mesh)

    # obstacle and solution are in P1
    V = FunctionSpace(mesh, "CG", 1)
    Z = VectorFunctionSpace(mesh, "CG", 1)  # velocity in this space

    # generate wind and source term from formulas found in
    #   https://bitbucket.org/pefarrell/fascd/src/master/examples/pollutant.py
    (x, y, z) = SpatialCoordinate(mesh)
    w = Function(Z, name="w (velocity)").interpolate(
        as_vector([7.0 + 5.0 * y, -5.0 * x, 2.0 * z])
    )
    d0 = (x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0)
    d1 = (x - x1) * (x - x1) + (y - y1) * (y - y1) + (z - z1) * (z - z1)
    a_ufl = conditional(Or(d0 < r0 ** 2, d1 < r1 ** 2), alpha, 0.0)
    b_ufl = conditional(x > 0, -beta * (1.0 - cos(6.0 * pi * x)), 0.0)
    f = Function(V, name="f (source)").interpolate(a_ufl + b_ufl)

    # initialize strictly admissible
    u = Function(V, name="u (FE soln)").interpolate(Constant(0.5))

    # weak form
    v = TestFunction(V)
    C = Constant(0.1)
    F = C * inner(grad(u), grad(v)) * dx + inner(dot(w, grad(u)) - f, v) * dx
    bdry_ids = (1,)  # Dirichlet only on x = -1 side
    bcs = DirichletBC(V, Constant(0.0), bdry_ids)
    prob = NonlinearVariationalProblem(F, u, bcs)

    # solve
    sp = {
        "snes_type": "vinewtonrsls",
        "snes_rtol": 1.0e-8,
        "snes_atol": 1.0e-12,
        "snes_stol": 1.0e-12,
        "snes_vi_zero_tolerance": 1.0e-12,
        "snes_converged_reason": None,
        "snes_vi_monitor": None,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    lb = Function(V).interpolate(Constant(0.0))
    ub = Function(V).interpolate(Constant(1.0))
    solver = NonlinearVariationalSolver(prob, solver_parameters=sp, options_prefix="")
    solver.solve(bounds=(lb, ub))

    if i == refinements:
        break

    # ub is an upper obstacle, so its call needs boxside="upper"; the default
    # boxside="lower" would treat ub as a floor, which is harmless with
    # debug=False but trips VIAMR._checkuhbound()'s admissibility assertion
    # (uh >= ub) with debug=True
    marklower = amr.udomark(u, lb, boxside="lower", n=1)
    markupper = amr.udomark(u, ub, boxside="upper", n=1)
    mark = amr.unionmarks(marklower, markupper)
    mesh = mesh.refine_marked_elements(mark)  # uses Netgen refinement
    # VTKFile(f'test{i}.pvd').write(u, f, w, lb, ub, marklower, markupper, mark)

outfile = "result_pollutant.pvd"
print(f"done ... writing to {outfile} ...")
if mesh.comm.size > 1:  # integer-valued element-wise process rank
    rank = Function(FunctionSpace(mesh, "DG", 0))
    rank.dat.data[:] = mesh.comm.rank
    rank.rename("rank")
    VTKFile(outfile).write(u, f, w, rank)
else:
    VTKFile(outfile).write(u, f, w)
