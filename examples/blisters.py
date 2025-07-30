import numpy as np
from firedrake import *
from firedrake.petsc import PETSc
from viamr import VIAMR

print = PETSc.Sys.Print  # enables correct printing in parallel

levels = 4
m_initial = 30
m_data = 500
outfile = "result_blisters.pvd"

def normal2d(mesh, x0, y0, sigma):
    # return UFL expression for one gaussian hump
    x, y = SpatialCoordinate(mesh)
    C = 1.0 / (2.0 * pi * sigma**2)
    dsqr = (x - x0) ** 2 + (y - y0) ** 2
    return C * exp(-dsqr / (2.0 * sigma**2))


def eval_fsource(mesh):
    xysw = [
        (0.3, 0.8, 0.04, 0.6),
        (0.8, 0.35, 0.02, 0.5),
        (0.8, 0.25, 0.03, 0.4),
        (0.1, 0.3, 0.02, 0.4),
        (0.3, 0.32, 0.02, 0.3),
        (0.4, 0.2, 0.02, 0.4),
        (0.8, 0.66, 0.01, 0.2),
        (0.78, 0.75, 0.01, 0.1),
        (0.7, 0.82, 0.01, 0.2),
    ]
    f_ufl = -17.0
    for x, y, sigma, weight in xysw:
        f_ufl += weight * normal2d(mesh, x, y, sigma)
    return f_ufl


params = {
    "snes_type": "vinewtonrsls",
    "snes_vi_zero_tolerance": 1.0e-12,
    "snes_linesearch_type": "basic",
    # "snes_monitor": None,
    "snes_vi_monitor": None,
    "snes_converged_reason": None,
    "snes_rtol": 1.0e-8,
    "snes_atol": 1.0e-12,
    "snes_stol": 1.0e-12,
    "snes_max_it": 200,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

print(f"evaluating source data f(x,y) on fine ({m_data} x {m_data} CG2) data mesh ...")
datamesh = UnitSquareMesh(m_data, m_data)
dataV = FunctionSpace(datamesh, "CG", 2)
fdata = Function(dataV, name="f_data(x,y)")
fdata.interpolate(eval_fsource(datamesh))

datafile = "result_data.pvd"
print(f"writing source f(x,y) to {datafile} ...")
VTKFile(datafile).write(fdata)

initial_mesh = UnitSquareMesh(m_initial, m_initial)

amr = VIAMR()
meshhierarchy = [
    initial_mesh,
]
for i in range(levels + 1):
    mesh = meshhierarchy[i]
    print(f"solving on mesh {i} ...")
    amr.meshreport(mesh)

    V = FunctionSpace(mesh, "CG", 1)
    # cross-mesh interpolation from data mesh:
    fsource = Function(V, name="f_source(x,y)").interpolate(fdata)
    # cross-mesh interpolation from coarser mesh:
    u = Function(V, name="u_h(x,y)").interpolate(Constant(0.0) if i == 0 else u)

    v = TestFunction(V)
    F = inner(grad(u), grad(v)) * dx - fsource * v * dx
    bcs = DirichletBC(V, Constant(0.0), (1, 2, 3, 4))
    problem = NonlinearVariationalProblem(F, u, bcs)

    # solve the VI
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=params, options_prefix="s"
    )
    lb = Function(V).interpolate(Constant(0.0))
    ub = Function(V).interpolate(Constant(PETSc.INFINITY))
    solver.solve(bounds=(lb, ub))

    # evaluate inactive fraction
    if i > 0:
        newei = amr.eleminactive(u, lb)
        jac = amr.jaccard(newei, ei, submesh=True)
        print(f"  Jaccard agreement {100*jac:.2f}% [levels {i-1}, {i}]")
        ei = newei
    else:
        ei = amr.eleminactive(u, lb)
    ifrac = assemble(ei * dx)
    print(f"  inactive fraction {ifrac:.6f}")

    if i == levels:
        break

    # apply VCD+BR AMR
    mark = amr.vcdmark(u, lb, bracket=[0.05, 0.85])
    residual = -div(grad(u)) - fsource
    imark, _, _ = amr.brinactivemark(u, lb, residual)
    mark = amr.unionmarks(mark, imark)
    mesh = amr.refinemarkedelements(mesh, mark)
    meshhierarchy.append(mesh)

print(f"done ... writing solution u(x,y) and f(x,y) to {outfile} ...")
VTKFile(outfile).write(u, fsource)
