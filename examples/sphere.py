# This example attempts to do apples-to-apples comparisons of 4 algorithms
# on the "ball" problem:
#   1. UDOBR = unstructured dilation operator plus Babuska & Rheinboldt (1989) in the inactive set
#   2. NSV = Nochetto, Siebert, and Veeser (2003)
#   3. UNI = uniform refinement
#   4. AVM = averaged-metric mesh adaptation  <-- only runs if avm import succeeds
#
# The exact solution is known and so we can compute norm convergence rates.
# We generate .pvd files: result_sphere_{udobr,nsv,uni,avm}.pvd
# We measure Hausdorff convergence rates in serial.
# Optionally we generate .csv files for norm and Jaccard convergence rates.


# note: use higher targetelements and/or maxlevels for serious convergence
# e.g. targetelements=1.0e6 & maxlevels=11 <--> uniformlevels=7
m0 = 10  # for UDOBR,NSV,UNI initial mesh is m0 x m0; see below for AVM
targetelements = 1.0e5  # all methods stop refining once this is met
maxlevels = 11  # backstop target element complexity; set to 15 for more data?
uniformlevels = 5  # generally uniform can't reach high levels ... which is the point
writecsvs = False
dohausdorff = True


import time
import numpy as np
from firedrake import *
from firedrake.petsc import PETSc
from viamr import VIAMR

print = PETSc.Sys.Print  # enables correct printing in parallel

try:
    import netgen
except ImportError:
    raise ImportError("Unable to import NetGen.  Exiting.")
from netgen.geom2d import SplineGeometry

# what methods of adaptive refinement do we test
refinetypes = ["udobr", "nsv", "uni"]
try:
    import animate
except:
    print(
        "WARNING: animate import failed, proceeding without it using Firedrake meshes only"
    )
    pass
else:
    refinetypes.append("avm")  # if import succeeded

# method parameters
thetaBR = 0.9  # controls BR resolution in inactive set, and convergence rate

# AVM parameters; attempts to do apples-to-apples vs UDOBR
initialhAVM = 4.0 / m0
targetsAVM = [
    100,
    300,
    900,
    3000,
    7000,
    18000,
    50000,
    100000,
    250000,
    600000,
    1400000,
    3000000,
]


def psiUFL(r):
    """obstacle as UFL, from UFL expression for r"""
    r0 = 0.9
    psi0 = np.sqrt(1.0 - r0 * r0)
    dpsi0 = -r0 / psi0
    return conditional(le(r, r0), sqrt(1.0 - r * r), psi0 + dpsi0 * (r - r0))


def uexactUFL(r):
    """exact solution (and boundary conditions) as UFL, from UFL expression for r"""
    afree = 0.697965148223374
    A, B = 0.680259411891719, 0.471519893402112
    return conditional(le(r, afree), psiUFL(r), -A * ln(r) + B)


def activeexactUFL(r):
    """exact active set as UFL, from UFL expression for r"""
    afree = 0.697965148223374
    return conditional(le(r, afree), 1.0, 0.0)


def errornorm_deg20(u, uh):
    """L^2 error norm, but avoiding TSFC warning"""
    # set high degree quadrature to avoid TSFC warning
    normsq = assemble((u - uh) ** 2 * dx(degree=20))
    return np.sqrt(normsq)


def errornorm_preferred_deg20(r, uh, activeh):
    """L^2 error norm against "preferred" form of numerical solution"""
    tildeuh = conditional(eq(activeh, 1.0), psiUFL(r), uh)  # preferred
    # high degree quadrature important in next line; setting it avoids TSFC warning
    normsq = assemble((uexactUFL(r) - tildeuh) ** 2 * dx(degree=20))
    return np.sqrt(normsq)


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

for amrtype in refinetypes:
    print(f"solving by VIAMR using {amrtype.upper()} method ...")

    amr = VIAMR()

    # setting distribution parameters should not be necessary ... but bug in netgen
    dp = {
        "partition": True,
        "overlap_type": (DistributedMeshOverlapType.VERTEX, 1),
    }
    if amrtype == "avm":
        geo = SplineGeometry()
        geo.AddRectangle(p1=(-2, -2), p2=(2, 2), bc="rectangle")
        ngmsh = geo.GenerateMesh(maxh=initialhAVM)
        mesh0 = Mesh(ngmsh, distribution_parameters=dp)
    else:
        mesh0 = RectangleMesh(
            m0,
            m0,
            Lx=2.0,
            Ly=2.0,
            originX=-2.0,
            originY=-2.0,
            distribution_parameters=dp,
        )
    meshHist = [mesh0]

    if amrtype == "uni":
        unimh = MeshHierarchy(mesh0, uniformlevels)

    if writecsvs:
        csvfile = open(f"sphere_{amrtype}.csv", "w")
        csvfile.write(
            "I,NV,NE,HMIN,HMAX,ENORM,ENORMPREF,JACCARD,HAUSDORFF,REFINETIME\n"
        )

    for i in range(maxlevels + 1):
        print(f"solving on mesh {i} ...")
        mesh = meshHist[i]
        amr.meshreport(mesh)
        V = FunctionSpace(mesh, "CG", 1)
        if i == 0:
            uh = Function(V, name="u_h")
            refinetime = 0.0
        else:
            # initialize by cross-mesh interpolation to fine mesh
            uUFL = conditional(uh < lb, lb, uh)  # use old data
            uh = Function(V, name="u_h").interpolate(uUFL)

        # set up and solve problem
        v = TestFunction(V)
        F = inner(grad(uh), grad(v)) * dx
        x, y = SpatialCoordinate(mesh)
        r = sqrt(x * x + y * y)
        g_ufl = uexactUFL(r)
        bcs = DirichletBC(V, g_ufl, "on_boundary")
        problem = NonlinearVariationalProblem(F, uh, bcs)
        solver = NonlinearVariationalSolver(
            problem, solver_parameters=sp, options_prefix="s"
        )
        lb = Function(V, name="psi").interpolate(psiUFL(r))
        ub = Function(V).interpolate(Constant(PETSc.INFINITY))
        solver.solve(bounds=(lb, ub))

        # compute norms
        en_no = errornorm_deg20(uexactUFL(r), uh)
        activeh = amr.elemactive(uh, lb)
        en_pre = errornorm_preferred_deg20(r, uh, activeh)
        print(f"  ||u_exact - u_h||_2 = {en_no:.3e}")
        print(f"  ||u_exact - tilde u_h||_2 = {en_pre:.3e}")
        jaccard = amr.jaccardUFL(activeexactUFL(r), activeh)
        print(f"  jaccard(A_u, A_uh) = {jaccard:.5f}")
        haus = PETSc.INFINITY
        if dohausdorff:
            if mesh.comm.size == 1:
                uexact = Function(V, name="u_exact").interpolate(uexactUFL(r))
                _, fbexact = amr.freeboundarygraph2D(uexact, lb)
                _, fb = amr.freeboundarygraph2D(uh, lb)
                haus = amr.hausdorff(fbexact, fb)
                print(f"  hausdorff(Gamma_u, Gamma_uh) = {haus:.5f}")
            else:
                print(
                    "WARNING: measuring Hausdorff distance between exact and approximate free boundary not possible in parallel ... turning it off"
                )
                dohausdorff = False

        # report, and break if targer complexity met
        Nv, Ne, hmin, hmax = amr.meshsizes(mesh)
        if writecsvs:
            csvfile.write(
                f"{i},{Nv},{Ne},{hmin:.5f},{hmax:.5f},{en_no:.3e},{en_pre:.3e},{jaccard:.5f},{haus:.5f},{refinetime:.3e}\n"
            )
        if Ne > targetelements:
            break

        # do an AMR level
        start_time = time.time()
        if amrtype == "uni":
            mesh = unimh[i + 1]
        elif amrtype == "avm":
            amr.setmetricparameters(
                target_complexity=targetsAVM[i + 1], h_min=1.0e-4, h_max=1.0
            )
            mesh = amr.adaptaveragedmetric(mesh, uh, lb)
        elif amrtype == "nsv":
            g = Function(V).interpolate(g_ufl)
            (mark, _, _, _) = amr.nsvmark(uh, lb, g, Constant(0.0), g_ufl)
            mesh = amr.refinemarkedelements(mesh, mark)
        else:
            mark = amr.udomark(uh, lb, n=1)
            residual = -div(grad(uh))
            (imark, _, _) = amr.brinactivemark(uh, lb, residual, theta=thetaBR)
            mark = amr.unionmarks(mark, imark)
            mesh = amr.refinemarkedelements(mesh, mark)
        if amrtype != "uni":
            refinetime = time.time() - start_time
        meshHist.append(mesh)

    if writecsvs:
        csvfile.close()

    outfile = "result_sphere_" + amrtype + ".pvd"
    print(f"done ... writing to {outfile} ...")
    gap = Function(V, name="gap = uh-lb").interpolate(uh - lb)
    uexact = Function(V, name="u_exact").interpolate(uexactUFL(r))
    error = Function(V, name="error = |pi_h(uexact) - uh|").interpolate(
        abs(uexact - uh)
    )
    if amrtype == "udobr":
        # for output file, compute imark, mark on final mesh
        residual = -div(grad(uh))
        imark, _, _ = amr.brinactivemark(uh, lb, residual, theta=thetaBR)
        mark = amr.udomark(uh, lb, n=1)
        mark = amr.unionmarks(mark, imark)
        imark.rename("imark (BR)")
        mark.rename("mark")
        VTKFile(outfile).write(uh, lb, gap, uexact, error, mark, imark)
    elif amrtype == "nsv":
        # for output file, compute mark, etainf, sigmah on final mesh FIXME
        g = Function(V).interpolate(g_ufl)
        (mark, etainf, sigmah, _) = amr.nsvmark(uh, lb, g, Constant(0.0), g_ufl)
        mark.rename("mark")
        dualtol = 1.0e-10
        lnsigmah = Function(V, name="ln(sigma_h)").interpolate(ln(sigmah + dualtol))
        lnetainf = Function(V, name="ln(eta_inf)").interpolate(ln(etainf))
        VTKFile(outfile).write(
            uh, lb, gap, uexact, error, mark, sigmah, lnsigmah, etainf, lnetainf
        )
    else:
        VTKFile(outfile).write(uh, lb, gap, uexact, error)
    print("")
