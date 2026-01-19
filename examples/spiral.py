# This example generates two .pvd files, result_spiral_{udo,vcd}.pvd, suitable for
# a figure in the paper comparing n=1 UDO to [0.1,0.9] VCD on the spiral problem.
# Notes:
#   1) For simplicity we just use an initial Firedrake mesh, so AVM is not applied.
#   2) Because of thin active set, Jaccard similarity is zero until level i=4.

from firedrake import *
from firedrake.petsc import PETSc

print = PETSc.Sys.Print  # enables correct printing in parallel
from viamr import VIAMR

m0 = 10  # initial mesh is m0 x m0
targetnodes = 20000  # stop on first mesh to reach this many nodes
maxlevels = 12  # backstop
useVCD = False  # add VCD+BR results if this is True
dualtol = 1.0e-6  # used in NSV

# setting distribution parameters should not be necessary ... but bug in netgen
dp = {
    "partition": True,
    "overlap_type": (DistributedMeshOverlapType.VERTEX, 1),
}
mesh0 = RectangleMesh(m0, m0, Lx=1.0, Ly=1.0, originX=-1.0, originY=-1.0,
                      distribution_parameters=dp)

def psi(x, y):
    '''obstacle psi(x,y) from Graeser & Kornhuber (2009), subsection 7.1.1'''
    r = sqrt(x * x + y * y)
    theta = atan2(y, x)
    tmp = sin(2.0*pi/r + pi/2.0 - theta) + r * (r+1) / (r - 2.0) - 3.0 * r + 3.6
    return conditional(le(r, 1.0e-8), 3.6, tmp)

# solver parameters for VI
sp = {
    "snes_type": "vinewtonrsls",
    "snes_vi_zero_tolerance": 1.0e-12,
    "snes_linesearch_type": "basic",
    "snes_max_it": 200,
    "snes_rtol": 1.0e-8,
    "snes_atol": 1.0e-12,
    "snes_stol": 0.0,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    # "snes_vi_monitor": None,
    "snes_converged_reason": None,
}

if useVCD:
    typelist = ["udobr", "vcdbr", "nsv"]
else:
    typelist = ["udobr", "nsv"]

for amrtype in typelist:
    meshHist = [mesh0]

    for i in range(maxlevels + 1):
        print(f"solving spiral problem using {amrtype.upper()} on mesh {i} ...")
        mesh = meshHist[i]
        V = FunctionSpace(mesh, "CG", 1)
        gbdry = Constant(0.0)

        if i == 0:
            uh = Function(V, name="u_h").interpolate(gbdry)
        else:
            # initialize by cross-mesh interpolation to fine mesh
            uUFL = conditional(uh < lb, lb, uh)  # use old data
            uh = Function(V, name="u_h").interpolate(uUFL)

        amr = VIAMR()
        amr.meshreport(mesh)

        v = TestFunction(V)
        F = inner(grad(uh), grad(v)) * dx
        bcs = DirichletBC(V, gbdry, "on_boundary")
        problem = NonlinearVariationalProblem(F, uh, bcs)
        solver = NonlinearVariationalSolver(
            problem, solver_parameters=sp, options_prefix="s"
        )

        x, y = SpatialCoordinate(mesh)
        lb = Function(V, name="psi").interpolate(psi(x,y))
        ub = Function(V).interpolate(Constant(PETSc.INFINITY))
        solver.solve(bounds=(lb, ub))

        neweactive = amr.elemactive(uh, lb)
        if i > 0:
            jac = amr.jaccard(neweactive, eactive, submesh=True)
            print(f"  Jaccard agreement {100*jac:.2f}% [levels {i-1}, {i}]")
        eactive = neweactive

        if V.dim() >= targetnodes:
            break

        if amrtype in ["udobr", "vcdbr"]:
            residual = -div(grad(uh))
            imark, _, _ = amr.brinactivemark(uh, lb, residual, method="total")
            if amrtype == "udobr":
                mark = amr.udomark(uh, lb, n=1)
            elif amrtype == "vcdbr":
                mark = amr.vcdmark(uh, lb, bracket=[0.1, 0.9])
            mark = amr.unionmarks(mark, imark)
        else:
            (mark, _, _, _) = amr.nsvmark(uh, lb, Constant(0.0), Constant(0.0), Constant(0.0), method="total", dualtol=dualtol)
            
        mesh = amr.refinemarkedelements(mesh, mark)
        meshHist.append(mesh)

    outfile = "result_spiral_" + amrtype + ".pvd"
    print(f"done ... writing to {outfile} ...")
    gap = Function(V, name="gap = uh-lb").interpolate(uh - lb)
    if amrtype in ["udobr", "vcdbr"]:
        # for output file, compute imark, mark on final mesh
        residual = -div(grad(uh))
        imark, _, _ = amr.brinactivemark(uh, lb, residual, method="total")
        if amrtype == "udobr":
            mark = amr.udomark(uh, lb, n=1)
        elif amrtype == "vcdbr":
            mark = amr.vcdmark(uh, lb, bracket=[0.1, 0.9])
        mark = amr.unionmarks(mark, imark)
        imark.rename("imark (BR)")
        mark.rename("mark")
        VTKFile(outfile).write(uh, lb, gap, mark, imark)
    else:
        # for output file, compute mark, etainf, sigmah on final mesh
        (mark, etainf, sigmah, _) = amr.nsvmark(uh, lb, Constant(0.0), Constant(0.0), Constant(0.0), method="total", dualtol=dualtol)
        mark.rename("mark")
        lnsigmah = Function(V, name="ln(sigma_h)").interpolate(ln(sigmah + dualtol))
        lnetainf = Function(V, name="ln(eta_inf)").interpolate(ln(etainf))
        VTKFile(outfile).write(uh, lb, gap, mark, sigmah, lnsigmah, etainf, lnetainf)
    print("")
