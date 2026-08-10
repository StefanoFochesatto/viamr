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
# We generate four .png convergence figures comparing all methods:
#   sphere_convergence_unorm.png       ||u_exact - u_h||_2 vs DOFs
#   sphere_convergence_preferred.png   ||u_exact - tilde u_h||_2 vs DOFs
#   sphere_convergence_h1.png          |u_exact - u_h|_{H^1 seminorm} vs DOFs
#   sphere_convergence_hausdorff.png   hausdorff(Gamma_u, Gamma_uh) vs DOFs


# note: use higher targetelements and/or maxlevels for serious convergence
# e.g. targetelements=1.0e6 & maxlevels=11 <--> uniformlevels=7
m0 = 10  # for UDOBR,NSV,UNI initial mesh is m0 x m0; see below for AVM
targetelements = 1.0e5  # all methods stop refining once this is met
maxlevels = 11  # backstop target element complexity; set to 15 for more data?
uniformlevels = 5  # generally uniform can't reach high levels ... which is the point
writecsvs = False


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


def errornorm_H1semi_deg20(u, uh):
    """H^1 seminorm (i.e. Dirichlet-energy / grad-L^2) error norm of the plain
    (not "preferred") numerical solution, avoiding the TSFC warning.  This is
    the norm brinactivemark()'s unweighted BR78 estimator directly targets
    (see its docstring), so it's the natural check on whether that estimator
    is doing what it's designed to do, independent of L^2 behavior."""
    normsq = assemble(inner(grad(u - uh), grad(u - uh)) * dx(degree=20))
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

results = {}  # results[amrtype] = (dofs, errnorm, errnorm_pres, hausdorffs, errnorm_h1s),
# for the four convergence figures generated at the end

for amrtype in refinetypes:
    print(f"solving by VIAMR using {amrtype.upper()} method ...")

    amr = VIAMR()
    dofs, errnorms, errnorm_pres, hausdorffs, errnorm_h1s = [], [], [], [], []

    # udomark() needs this overlap to give correct results in parallel
    dp = VIAMR.PARALLEL_OVERLAP
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
            diagonal="crossed",
            distribution_parameters=dp,
        )
    meshHist = [mesh0]

    if amrtype == "uni":
        unimh = MeshHierarchy(mesh0, uniformlevels)

    if writecsvs:
        csvfile = open(f"sphere_{amrtype}.csv", "w")
        csvfile.write(
            "I,NV,NE,HMIN,HMAX,ENORM,ENORMPREF,ENORMH1,JACCARD,HAUSDORFF,REFINETIME\n"
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
        errnorm = errornorm_deg20(uexactUFL(r), uh)
        activeh = amr.elemactive(uh, lb)
        errnorm_pre = errornorm_preferred_deg20(r, uh, activeh)
        errnorm_h1 = errornorm_H1semi_deg20(uexactUFL(r), uh)
        print(f"  ||u_exact - u_h||_2 = {errnorm:.3e}")
        print(f"  ||u_exact - tilde u_h||_2 = {errnorm_pre:.3e}")
        print(f"  |u_exact - u_h|_{{H^1}} = {errnorm_h1:.3e}")
        jaccard = amr.jaccardUFL(activeexactUFL(r), activeh)
        print(f"  jaccard(A_u, A_uh) = {jaccard:.5f}")
        uexact = Function(V, name="u_exact").interpolate(uexactUFL(r))
        _, fbexact = amr.freeboundarygraph2D(uexact, lb)
        _, fb = amr.freeboundarygraph2D(uh, lb)
        # hausdorff() returns None for an empty free boundary (e.g. a coarse
        # initial mesh); format accordingly
        haus = amr.hausdorff(fbexact, fb)
        hausstr = f"{haus:.5f}" if haus is not None else "n/a"
        print(f"  hausdorff(Gamma_u, Gamma_uh) = {hausstr}")

        # report, and break if target complexity met
        Nv, Ne, hmin, hmax = amr.meshsizes(mesh)
        dofs.append(Nv)
        errnorms.append(errnorm)
        errnorm_pres.append(errnorm_pre)
        hausdorffs.append(haus if haus is not None else np.nan)  # nan plots as a gap
        errnorm_h1s.append(errnorm_h1)
        if writecsvs:
            csvfile.write(
                f"{i},{Nv},{Ne},{hmin:.5f},{hmax:.5f},{errnorm:.3e},{errnorm_pre:.3e},{errnorm_h1:.3e},{jaccard:.5f},{hausstr},{refinetime:.3e}\n"
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
            (mark, _, _, _, _) = amr.nsvmark(uh, lb, g, Constant(0.0), g_ufl)
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

    results[amrtype] = (dofs, errnorms, errnorm_pres, hausdorffs, errnorm_h1s)

    outfile = "result_sphere_" + amrtype + ".pvd"
    print(f"done ... writing to {outfile} ...")
    gap = Function(V, name="gap = uh-lb").interpolate(uh - lb)
    uexact = Function(V, name="u_exact").interpolate(uexactUFL(r))
    error = Function(V, name="error = |pi_h(uexact) - uh|").interpolate(
        abs(uexact - uh)
    )
    fields = [uh, lb, gap, uexact, error]
    if amrtype == "udobr":
        # for output file, compute imark, mark on final mesh
        residual = -div(grad(uh))
        imark, _, _ = amr.brinactivemark(uh, lb, residual, theta=thetaBR)
        mark = amr.udomark(uh, lb, n=1)
        mark = amr.unionmarks(mark, imark)
        imark.rename("imark (BR)")
        mark.rename("mark")
        fields += [mark, imark]
    elif amrtype == "nsv":
        # for output file, compute mark, etainf, sigmah on final mesh FIXME
        g = Function(V).interpolate(g_ufl)
        (mark, etainf, sigmah, _, etad) = amr.nsvmark(uh, lb, g, Constant(0.0), g_ufl)
        mark.rename("mark")
        dualtol = 1.0e-10
        lnsigmah = Function(V, name="ln(sigma_h)").interpolate(ln(sigmah + dualtol))
        lnetainf = Function(V, name="ln(eta_inf)").interpolate(ln(etainf))
        fields += [mark, sigmah, lnsigmah, etainf, lnetainf, etad]
    VTKFile(outfile).write(*fields)
    print("")

# convergence figures comparing all methods; one figure per quantity, all
# methods overlaid, following the convergence-figure convention in nsv.py
if mesh.comm.rank == 0:
    import matplotlib.pyplot as plt

    d = 2  # spatial dimension
    stylemap = {"udobr": "ko", "nsv": "bs", "uni": "rs", "avm": "g^"}

    def _convergence_plot(index, ylabel, title, outfile, rateexp, ratelabel):
        plt.figure()
        for amrtype in refinetypes:
            rdofs, rvals = np.array(results[amrtype][0]), np.array(results[amrtype][index])
            plt.loglog(rdofs, rvals, stylemap[amrtype], label=amrtype.upper())
        rdofs, rvals = np.array(results["uni"][0]), np.array(results["uni"][index])
        # anchor to the first finite, positive UNI value, not just index 0;
        #    relevant to coarse initial mesh
        valid = np.isfinite(rvals) & (rvals > 0)
        if np.any(valid):
            anchor = np.argmax(valid)  # index of first valid entry
            y = rdofs.astype(float) ** rateexp
            y = y * rvals[anchor] / y[anchor]
            plt.loglog(rdofs, y, "k:", label=ratelabel)
        plt.legend()
        plt.grid(True)
        plt.xlabel("DOFs")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(outfile)

    # L^2 norms: DOFs^(-2/d) is the optimal rate for quasi-optimal 2D meshes,
    # as in nsv.py.  The "preferred" reconstruction (see errornorm_preferred_deg20)
    # is kept in the same L^2 norm as the plain error, rather than H^1, because
    # its patched-together tilde u_h is not H^1-conforming.
    _convergence_plot(
        1,
        "||u_exact - u_h||_2",
        "sphere problem: solution norm vs DOFs",
        "sphere_convergence_unorm.png",
        -2.0 / d,
        "DOFs^(-2/d)",
    )
    _convergence_plot(
        2,
        "||u_exact - tilde u_h||_2",
        "sphere problem: preferred-form error norm vs DOFs",
        "sphere_convergence_preferred.png",
        -2.0 / d,
        "DOFs^(-2/d)",
    )
    # Hausdorff distance is a geometric (not squared-error) quantity, so the
    # natural reference is mesh-resolution scaling h ~ DOFs^(-1/d), not the
    # L^2 rate above.
    _convergence_plot(
        3,
        "hausdorff(Gamma_u, Gamma_uh)",
        "sphere problem: free-boundary Hausdorff distance vs DOFs",
        "sphere_convergence_hausdorff.png",
        -1.0 / d,
        "DOFs^(-1/d)",
    )
    # H^1 seminorm: classical (Falk-type) a priori theory for the obstacle
    # problem gives first-order convergence, i.e. DOFs^(-1/d), same as
    # Hausdorff's mesh-resolution rate, not the L^2 rate above.  This is the
    # norm brinactivemark()'s BR78 estimator directly targets.
    _convergence_plot(
        4,
        "|u_exact - u_h|_{H^1}",
        "sphere problem: H^1 seminorm error vs DOFs",
        "sphere_convergence_h1.png",
        -1.0 / d,
        "DOFs^(-1/d)",
    )
