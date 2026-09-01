# Solve the classical obstacle problem from Graeser & Kornhuber (2009) which
# generates a spiral-shaped coincidence (active) set.
# This example generates 3 .pvd files, result_spiral_{udobr,vcdbr,nsv}.pvd.
# Notes:
#   1 For simplicity we just use a Firedrake mesh.  AVM is not applied.
#   2 n=0 UDO and [0.1,0.9]-bracket VCD are compared.
#   3 Because of thin and highly-nontrivial active set, Jaccard similarity
#     is zero until a few levels in.
#   4 The less expensive NSV03 method is used.

m0 = 10  # initial mesh is m0 x m0
targetnodes = 1e4  # stop on first mesh to reach this many nodes
maxlevels = 12  # backstop the targetnodes

from firedrake import *
from firedrake.petsc import PETSc

print = PETSc.Sys.Print  # enables correct printing in parallel
from viamr import VIAMR

# udomark() needs this overlap to give correct results in parallel
mesh0 = RectangleMesh(
    m0,
    m0,
    Lx=1.0,
    Ly=1.0,
    originX=-1.0,
    originY=-1.0,
    distribution_parameters=VIAMR.PARALLEL_OVERLAP,
)


def psi(x, y):
    """obstacle psi(x,y) from Graeser & Kornhuber (2009), subsection 7.1.1"""
    r = sqrt(x * x + y * y)
    theta = atan2(y, x)
    tmp = sin(2.0 * pi / r + pi / 2.0 - theta) + r * (r + 1) / (r - 2.0) - 3.0 * r + 3.6
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

typelist = ["udobr", "vcdbr", "nsv"]

for amrtype in typelist:
    print(f"solving spiral problem using {amrtype.upper()} ...")
    meshHist = [mesh0]

    for i in range(maxlevels + 1):
        print(f"mesh {i}:")
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
        lb = Function(V, name="psi").interpolate(psi(x, y))
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
                mark = amr.udomark(uh, lb, n=0)
            elif amrtype == "vcdbr":
                mark = amr.vcdmark(uh, lb, bracket=[0.1, 0.9])
            mark = amr.unionmarks(mark, imark)
        else:
            z = Constant(0.0)
            mark, _, _, _, _ = amr.nsv03mark(uh, (lb, None), z, z, z, method="total", dualtol=1.0e-6)

        mesh = amr.refinesbr2D(mesh, mark)
        meshHist.append(mesh)

    outfile = "result_spiral_" + amrtype + ".pvd"
    print(f"done ... writing to {outfile} ...")
    gap = Function(V, name="gap = uh-lb").interpolate(uh - lb)
    fields = [uh, lb, gap]
    if amrtype in ["udobr", "vcdbr"]:
        # for output file, compute imark, mark on final mesh
        residual = -div(grad(uh))
        imark, _, _ = amr.brinactivemark(uh, lb, residual, method="total")
        if amrtype == "udobr":
            fbmark = amr.udomark(uh, lb, n=1)
        elif amrtype == "vcdbr":
            fbmark = amr.vcdmark(uh, lb, bracket=[0.1, 0.9])
        mark = amr.unionmarks(fbmark, imark)
        imark.rename("imark (BR)")
        fbmark.rename(f"fbmark ({amrtype})")
        mark.rename("mark")
        fields += [mark, fbmark, imark]
    else:
        # for output file, compute fields on final mesh
        z = Constant(0.0)
        mark, etainf, etad, sigmah, _ = amr.nsv03mark(uh, (lb, None), z, z, z, method="total", dualtol=1.0e-6)
        mark.rename("mark")
        fields += [mark, sigmah, etainf, etad]
    VTKFile(outfile).write(*fields)
    print("")
