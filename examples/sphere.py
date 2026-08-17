# This example attempts to do apples-to-apples comparisons of 5 algorithms
# on the "ball" problem:
#   1. UDOBR = unstructured dilation operator plus Babuska & Rheinboldt (1989) in the inactive set
#   2. NSV = Nochetto, Siebert, and Veeser (2003)
#   3. NSVSAFE = NSV plus VIAMR.safeactiveunmark(), excluding
#      certified-safe active elements from nsvmark()'s marking threshold
#   4. UNI = uniform refinement
#   5. AVM = averaged-metric mesh adaptation  <-- only runs if avm import succeeds
#
# We generate .pvd files: result_sphere_{udobr,nsv,nsvsafe,uni,avm}.pvd
#
# For the default mode the exact solution is known and so we can compute
# norm convergence rates.  We generate ten .png convergence figures comparing
# all methods:
#   sphere_unorm.png                  ||u_exact - u_h||_2 vs DOFs
#   sphere_unorm_reconstructed.png    ||u_exact - tilde u_h||_2 vs DOFs
#   sphere_h1.png                     |u_exact - u_h|_{H^1 seminorm} vs DOFs
#   sphere_supnorm.png                ||u_exact - u_h||_infty vs DOFs, for all methods
#   sphere_supnorm_reconstructed.png  ||u_exact - tilde u_h||_infty vs DOFs, for all methods
#   sphere_hausdorff.png              hausdorff2D(Gamma_u, Gamma_uh) vs DOFs
#   sphere_amrtime.png                cumulative AMR wall time vs norm
#   sphere_marktime.png               same, but only marking / metric-building cost
#   sphere_meshbuildtime.png          same, but mesh-construction cost (PETSc or Pragmatic)
#   sphere_effectivity.png            estimator/true-error effectivity index vs DOFs
#
# Optionally we generate .csv files for norm and Jaccard convergence rates.


# note: use higher targetelements and/or maxlevels for serious convergence
# e.g. targetelements=1.0e6 & maxlevels=11 <--> uniformlevels=7
m0 = 10  # for UDOBR,NSV,UNI initial mesh is m0 x m0; see below for AVM
targetelements = 1.0e5  # all methods stop refining once this is met
maxlevels = 15  # backstop target element complexity
uniformlevels = 5  # generally uniform can't reach high levels ... which is the point
writecsvs = False

# method parameters
thetaBR = 0.5  # controls BR resolution in inactive set, and convergence rate
methodBR = "total"  # VIAMR._fixedrate() uses a method; vs "max"; affects tradeoffs


import time
import numpy as np
from mpi4py import MPI
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
refinetypes = ["udobr", "nsv", "nsvsafe", "uni"]
try:
    import animate
except:
    print(
        "WARNING: animate import failed, proceeding without it using Firedrake meshes only"
    )
    pass
else:
    refinetypes.append("avm")  # if import succeeded

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


def F_strong(u, f):
    """Strong form of the classical obstacle problem operator: L(u) - f.
    Used by VIAMR.safeactiveunmark() for the NSVSAFE method."""
    return -div(grad(u)) - f


def activeexactUFL(r):
    """exact active set as UFL, from UFL expression for r"""
    afree = 0.697965148223374
    return conditional(le(r, afree), 1.0, 0.0)


def errornorm_deg(u, uh, fixeddegree=6):
    """L^2 error norm, but avoiding TSFC warning"""
    # set high degree quadrature to avoid TSFC warning
    normsq = assemble((u - uh) ** 2 * dx(degree=fixeddegree))
    return np.sqrt(normsq)


def errornorm_reconstructed_deg(r, uh, activeh, fixeddegree=6):
    """L^2 error norm against reconstructed-uh form of numerical solution"""
    tildeuh = conditional(eq(activeh, 1.0), psiUFL(r), uh)  # reconstructed-uh
    # high degree quadrature important in next line; setting it avoids TSFC warning
    normsq = assemble((uexactUFL(r) - tildeuh) ** 2 * dx(degree=fixeddegree))
    return np.sqrt(normsq)


def errornorm_H1semi_deg(u, uh, fixeddegree=6):
    """H^1 seminorm (i.e. Dirichlet-energy / grad-L^2) error norm of the plain
    (not reconstructed-uh) numerical solution, avoiding the TSFC warning.  This is
    the norm brinactivemark()'s unweighted BR78 estimator directly targets
    (see its docstring), so it's the natural check on whether that estimator
    is doing what it's designed to do, independent of L^2 behavior."""
    normsq = assemble(inner(grad(u - uh), grad(u - uh)) * dx(degree=fixeddegree))
    return np.sqrt(normsq)


def errornorm_Linf(amr, u, uh, pdegree=4):
    """Approximate sup-norm (L^infty) error, via interpolation of the
    (generally non-polynomial) exact-minus-computed difference into a
    higher-degree CG^p space, then VIAMR.scalarrange() for a parallel-safe
    max; same technique nsvmark() itself uses internally for non-polynomial
    data (e.g. its bdryerr term).  This is the norm NSV03's "pointwise a
    posteriori error control" theory targets, in contrast to BR78/BV00's
    energy (H^1 seminorm) norm."""
    W = FunctionSpace(uh.function_space().mesh(), "CG", pdegree)
    err = Function(W).interpolate(abs(u - uh))
    return amr.scalarrange(err)[1]


def errornorm_Linf_reconstructed(amr, r, uh, activeh, pdegree=4):
    """Sup-norm (L^infty) error against the reconstructed-uh form of the
    numerical solution (see errornorm_reconstructed_deg), rather than against
    the plain uh.  This is the right comparison for methods (e.g. UDOBR,
    NSVSAFE) that deliberately leave the active-set interior coarse, since
    u=psi there regardless of resolution: the plain sup norm is then
    dominated by the (known, not-actually-erroneous) gap between the coarse
    uh and the true psi, rather than by genuine solution error.  Uses the
    same CG^p-interpolation + VIAMR.scalarrange() technique as
    errornorm_Linf(); see its docstring re. non-polynomial data."""
    tildeuh = conditional(eq(activeh, 1.0), psiUFL(r), uh)  # reconstructed-uh
    W = FunctionSpace(uh.function_space().mesh(), "CG", pdegree)
    err = Function(W).interpolate(abs(uexactUFL(r) - tildeuh))
    return amr.scalarrange(err)[1]


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

results = {}  # results[amrtype] = (dofs, errnorm, errnorm_recons, hausdorffs, errnorm_h1s,
# errnorm_Linfs, errnorm_Linf_recons, amrtimes, marktimes, meshbuildtimes), for the
# nine convergence/timing figures generated at the end
effs = {}  # effs[amrtype] = (eff_dofs, eff_vals), for udobr/nsv/nsvsafe only; see
# the effectivity-index figure generated at the end

for amrtype in refinetypes:
    print(f"solving by VIAMR using {amrtype.upper()} method ...")

    amr = VIAMR()
    dofs, errnorms, errnorm_recons, hausdorffs, errnorm_h1s, errnorm_Linfs, errnorm_Linf_recons = (
        [], [], [], [], [], [], []
    )
    amrtimes = []  # cumulative wall time spent in marking + mesh refinement
    cumamrtime = 0.0
    marktimes = []  # cumulative wall time spent in VIAMR's own marking/metric-building
    cummarktime = 0.0
    meshbuildtimes = []  # cumulative wall time spent in the external mesh-construction backend
    cummeshbuildtime = 0.0
    eff_dofs, eff_vals = [], []  # effectivity index vs DOFs; udobr/nsv/nsvsafe only

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
            "I,NV,NE,HMIN,HMAX,ENORM,ENORMRECON,ENORMH1,JACCARD,HAUSDORFF,"
            "REFINETIME,MARKTIME,MESHBUILDTIME\n"
        )

    for i in range(maxlevels + 1):
        print(f"solving on mesh {i} ...")
        mesh = meshHist[i]
        comm = mesh.comm  # fixed across this level, even once `mesh` is reassigned below
        amr.meshreport(mesh)
        V = FunctionSpace(mesh, "CG", 1)
        if i == 0:
            uh = Function(V, name="u_h")
            refinetime = 0.0
            marktime = 0.0
            meshbuildtime = 0.0
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
        errnorm = errornorm_deg(uexactUFL(r), uh)
        activeh = amr.elemactive(uh, lb)
        errnorm_recon = errornorm_reconstructed_deg(r, uh, activeh)
        errnorm_h1 = errornorm_H1semi_deg(uexactUFL(r), uh)
        errnorm_Linf = errornorm_Linf(amr, uexactUFL(r), uh)
        errnorm_Linf_recon = errornorm_Linf_reconstructed(amr, r, uh, activeh)
        print(f"  ||u_exact - u_h||_2 = {errnorm:.3e}")
        print(f"  ||u_exact - tilde u_h||_2 = {errnorm_recon:.3e}")
        print(f"  |u_exact - u_h|_{{H^1}} = {errnorm_h1:.3e}")
        print(f"  ||u_exact - u_h||_infty = {errnorm_Linf:.3e}")
        print(f"  ||u_exact - tilde u_h||_infty = {errnorm_Linf_recon:.3e}")
        jaccard = amr.jaccardUFL(activeexactUFL(r), activeh)
        print(f"  jaccard(A_u, A_uh) = {jaccard:.5f}")
        uexact = Function(V, name="u_exact").interpolate(uexactUFL(r))
        _, fbexact = amr.freeboundarygraph2D(uexact, lb)
        _, fb = amr.freeboundarygraph2D(uh, lb)
        # hausdorff2D() returns None for an empty free boundary (e.g. a coarse
        # initial mesh); format accordingly
        haus = amr.hausdorff2D(fbexact, fb)
        hausstr = f"{haus:.5f}" if haus is not None else "n/a"
        print(f"  hausdorff2D(Gamma_u, Gamma_uh) = {hausstr}")

        # report, and break if target complexity met
        Nv, Ne, hmin, hmax = amr.meshsizes(mesh)
        dofs.append(Nv)
        errnorms.append(errnorm)
        errnorm_recons.append(errnorm_recon)
        hausdorffs.append(haus if haus is not None else np.nan)  # nan plots as a gap
        errnorm_h1s.append(errnorm_h1)
        errnorm_Linfs.append(errnorm_Linf)
        errnorm_Linf_recons.append(errnorm_Linf_recon)
        # refinetime is the AMR cost (marking + refinement) that produced THIS
        # mesh from the previous one (0.0 at i=0); accumulating it here.
        # marktime/meshbuildtime split it into VIAMR-controlled cost (marking,
        # or AVM's metric-building) vs the external backend's mesh-construction
        # cost (PETSc SBR / Pragmatic).  marktime + meshbuildtime <= refinetime;
        # see the AMR-step comment below for why they need not sum exactly.
        cumamrtime += refinetime
        amrtimes.append(cumamrtime)
        cummarktime += marktime
        marktimes.append(cummarktime)
        cummeshbuildtime += meshbuildtime
        meshbuildtimes.append(cummeshbuildtime)
        if writecsvs:
            csvfile.write(
                f"{i},{Nv},{Ne},{hmin:.5f},{hmax:.5f},{errnorm:.3e},{errnorm_recon:.3e},{errnorm_h1:.3e},{jaccard:.5f},{hausstr},"
                f"{refinetime:.3e},{marktime:.3e},{meshbuildtime:.3e}\n"
            )
        if Ne > targetelements:
            break

        # do an AMR level.  Each non-uni branch times two phases separately:
        #   "mark" = whatever VIAMR itself computes (marking fields, or AVM's
        #            metric construction including vcdmark())
        #   "meshbuild" = the external backend call that actually builds the
        #            new mesh (PETSc SBR via refinesbr2D(), or Pragmatic via
        #            animate.adapt()) -- see refinesbr2D()'s docstring for why
        #            that cost doesn't scale down with a sparser marking.
        comm.Barrier()  # sync ranks before timing a collective operation
        start_time = time.time()
        if amrtype == "uni":
            mesh = unimh[i + 1]
        elif amrtype == "avm":
            amr.setmetricparameters(
                target_complexity=targetsAVM[i + 1], h_min=1.0e-4, h_max=1.0
            )
            t_mark0 = time.time()
            # buildaveragedmetric() only builds the (averaged) metric; it never
            # calls animate.adapt() itself, so we call that separately here and
            # time each step on its own
            metric = amr.buildaveragedmetric(mesh, uh, lb)
            t_mark1 = time.time()
            mesh = animate.adapt(mesh, metric)
            t_meshbuild1 = time.time()
            marktime_local = t_mark1 - t_mark0
            meshbuildtime_local = t_meshbuild1 - t_mark1
        elif amrtype in ("nsv", "nsvsafe"):
            t_mark0 = time.time()
            g = Function(V).interpolate(g_ufl)
            safe = None
            if amrtype == "nsvsafe":
                # compute safe *before* nsvmark(), so its eta values are
                # excluded from the marking threshold process
                safe = amr.safeactiveunmark(uh, lb, F_strong, psiUFL(r), Constant(0.0))
                print(f"  safeactiveunmark() excludes {amr.countmark(safe)} "
                      "active elements from the nsvmark() threshold")
            (mark, etainf, _, _, _) = amr.nsvmark(
                uh, lb, g, Constant(0.0), g_ufl, safe=safe
            )
            t_mark1 = time.time()
            # effectivity index vs NSV03's own target norm: max_T eta_inf(T)
            # is the whole-domain (not inactive-set-restricted) quantity its
            # pointwise theory bounds ||u-u_h||_infty by, in contrast to
            # BR78/BV00's energy-norm estimator below
            maxetainf = amr.scalarrange(etainf)[1]
            eff_nsv = maxetainf / errnorm_Linf if errnorm_Linf > 0 else np.nan
            print(f"  eff_{amrtype.upper()} (sup norm) = {eff_nsv:.3f}")
            eff_dofs.append(Nv)
            eff_vals.append(eff_nsv)
            mesh = amr.refinesbr2D(mesh, mark)
            t_meshbuild1 = time.time()
            marktime_local = t_mark1 - t_mark0
            meshbuildtime_local = t_meshbuild1 - t_mark1
        else:
            t_mark0 = time.time()
            mark = amr.udomark(uh, lb, n=1)
            residual = -div(grad(uh))
            (imark, _, tot_eta) = amr.brinactivemark(uh, lb, residual, theta=thetaBR, method=methodBR)
            mark = amr.unionmarks(mark, imark)
            t_mark1 = time.time()
            # effectivity index vs the SAME inactive set brinactivemark()
            # restricts its estimator to; matches the H^1 seminorm BR78's
            # unweighted estimator targets (see errornorm_H1semi_deg20)
            iamark = amr.eleminactive(uh, lb, strong=True)
            dus = inner(grad(uexactUFL(r) - uh), grad(uexactUFL(r) - uh))
            errH1_inactive = np.sqrt(assemble(dus * iamark * dx(degree=20)))
            eff_br = tot_eta / errH1_inactive if errH1_inactive > 0 else np.nan
            print(f"  eff_BR (inactive-set, energy norm) = {eff_br:.3f}")
            eff_dofs.append(Nv)
            eff_vals.append(eff_br)
            mesh = amr.refinesbr2D(mesh, mark)
            t_meshbuild1 = time.time()
            marktime_local = t_mark1 - t_mark0
            meshbuildtime_local = t_meshbuild1 - t_mark1
        if amrtype != "uni":
            # true cost of a collective AMR step is bounded by the slowest rank.
            # marktime + meshbuildtime is generally < refinetime: refinetime
            # times the whole step, but the NSV/NSVSAFE/UDOBR branches also do
            # effectivity-index bookkeeping (scalarrange(), assemble()) between
            # the mark and meshbuild calls, which isn't counted in either timer
            # (it's diagnostic-only, not part of what actually produces the
            # refined mesh) -- plus the usual load-imbalance slop from
            # allreducing marktime/meshbuildtime separately from refinetime.
            refinetime = comm.allreduce(time.time() - start_time, op=MPI.MAX)
            marktime = comm.allreduce(marktime_local, op=MPI.MAX)
            meshbuildtime = comm.allreduce(meshbuildtime_local, op=MPI.MAX)
            print(f"  AMR step: mark {marktime:.3f}s + meshbuild {meshbuildtime:.3f}s = {refinetime:.3f}s")
        meshHist.append(mesh)

    if writecsvs:
        csvfile.close()

    results[amrtype] = (
        dofs, errnorms, errnorm_recons, hausdorffs, errnorm_h1s, errnorm_Linfs,
        errnorm_Linf_recons, amrtimes, marktimes, meshbuildtimes,
    )
    effs[amrtype] = (eff_dofs, eff_vals)

    # generate .pvd output file, with AMR fields for final mesh
    outfile = "result_sphere_" + amrtype + ".pvd"
    print(f"done ... writing to {outfile} ...")
    gap = Function(V, name="gap = uh-lb").interpolate(uh - lb)
    uexact = Function(V, name="u_exact").interpolate(uexactUFL(r))
    error = Function(V, name="error = |uexact - uh|").interpolate(
        abs(uexact - uh)
    )
    fields = [uh, lb, gap, uexact, error]
    if amrtype == "udobr":
        residual = -div(grad(uh))
        imark, _, _ = amr.brinactivemark(uh, lb, residual, theta=thetaBR, method=methodBR)
        mark = amr.unionmarks(amr.udomark(uh, lb, n=1), imark)
        imark.rename("imark (BR)")
        mark.rename("mark")
        fields += [mark, imark]
    elif amrtype in ("nsv", "nsvsafe"):
        g = Function(V).interpolate(g_ufl)
        safe = None
        if amrtype == "nsvsafe":
            safe = amr.safeactiveunmark(uh, lb, F_strong, psiUFL(r), Constant(0.0))
            safe.rename("safe active unmarked")
        (mark, etainf, sigmah, _, etad) = amr.nsvmark(
            uh, lb, g, Constant(0.0), g_ufl, safe=safe
        )
        mark.rename("mark")
        fields += [mark, sigmah, etainf, etad]
        if amrtype == "nsvsafe":
            fields.append(safe)
    VTKFile(outfile).write(*fields)
    print("")

# convergence figures comparing all methods; one figure per quantity, all
# methods overlaid, following the convergence-figure convention in nsv.py
print("generating .png convergence/effectivity plots to compare algorithms ...")
if mesh.comm.rank == 0:
    import matplotlib.pyplot as plt

    d = 2  # spatial dimension
    stylemap = {"udobr": "ko", "nsv": "bs", "nsvsafe": "md", "uni": "rs", "avm": "g^"}

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
    # as in nsv.py.  The reconstructed-uh form (see errornorm_reconstructed_deg)
    # is kept in the same L^2 norm as the plain error, rather than H^1, because
    # its patched-together tilde u_h is not H^1-conforming.
    _convergence_plot(
        1,
        "||u_exact - u_h||_2",
        "solution norm vs DOFs",
        "sphere_unorm.png",
        -2.0 / d,
        "DOFs^(-2/d)",
    )
    _convergence_plot(
        2,
        "||u_exact - tilde u_h||_2",
        "reconstructed-uh norm vs DOFs",
        "sphere_unorm_reconstructed.png",
        -2.0 / d,
        "DOFs^(-2/d)",
    )
    # Hausdorff distance is a geometric (not squared-error) quantity, so the
    # natural reference is mesh-resolution scaling h ~ DOFs^(-1/d), not the
    # L^2 rate above.
    _convergence_plot(
        3,
        "hausdorff2D(Gamma_u, Gamma_uh)",
        "free-boundary Hausdorff distance vs DOFs",
        "sphere_hausdorff.png",
        -1.0 / d,
        "DOFs^(-1/d)",
    )
    # H^1 seminorm = energy norm: a priori theory for obstacle problem gives
    # first-order convergence, DOFs^(-1/d), BR78 estimator targets this norm
    _convergence_plot(
        4,
        "|u_exact - u_h|_{H^1}",
        "H^1 seminorm error vs DOFs",
        "sphere_h1.png",
        -1.0 / d,
        "DOFs^(-1/d)",
    )
    # sup (L^infty) norm: this is the norm NSV03's pointwise a posteriori
    # theory targets.  This figure gives an apples-to-apples true-error
    # comparison in that norm across *all* methods.  Reference rate
    # DOFs^(-2/d) matches the L^2 rate above: for smooth solutions, CG1
    # pointwise error is generically the same asymptotic order as L^2,
    _convergence_plot(
        5,
        "||u_exact - u_h||_infty",
        "sup-norm error vs DOFs",
        "sphere_supnorm.png",
        -2.0 / d,
        "DOFs^(-2/d)",
    )
    # reconstructed-uh companion to the plain sup-norm figure above (see
    # errornorm_Linf_reconstructed()): the right comparison for UDOBR and
    # NSVSAFE, which deliberately leave the active-set interior coarse.
    # Included for all methods, not just those two, as a consistency check;
    # NSV/UNI/AVM refine the active set, so their curves should barely move.
    _convergence_plot(
        6,
        "||u_exact - tilde u_h||_infty",
        "reconstructed-uh sup-norm error vs DOFs",
        "sphere_supnorm_reconstructed.png",
        -2.0 / d,
        "DOFs^(-2/d)",
    )

    # amrtime = cumulative AMR time (marking + mesh refinement only) vs
    #   reconstructed-uh error.  UNI is excluded: its time includes no
    #   marking/refinement step, thus uninformative here.  Levels with
    #   under 1s of cumulative AMR time are excluded as measurement noise.
    # marktime = VIAMR's own cost (marking fields, or AVM's metric-building
    #   via buildaveragedmetric())
    # meshbuildtime = the external backend's mesh-construction cost
    #   (PETSc SBR via refinesbr2D(), or Pragmatic via animate.adapt()).
    keepmasks = {}
    for amrtype in refinetypes:
        if amrtype == "uni":
            continue
        keepmasks[amrtype] = np.array(results[amrtype][7]) >= 1.0

    def _amrtime_plot(index, ylabel, title, outfile):
        plt.figure()
        for amrtype in refinetypes:
            if amrtype == "uni":
                continue
            rvals = np.array(results[amrtype][2])
            rtimes = np.array(results[amrtype][index])
            keep = keepmasks[amrtype]
            plt.loglog(rvals[keep], rtimes[keep], stylemap[amrtype], label=amrtype.upper())
        plt.legend()
        plt.grid(True)
        plt.xlabel("||u_exact - tilde u_h||_2")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(outfile)

    _amrtime_plot(
        7,
        "cumulative AMR time (s)",
        "cumulative AMR time vs reconstructed-uh norm",
        "sphere_amrtime.png",
    )
    _amrtime_plot(
        8,
        "cumulative mark time (s)",
        "cumulative marking/metric-building time vs reconstructed-uh norm",
        "sphere_marktime.png",
    )
    _amrtime_plot(
        9,
        "cumulative meshbuild time (s)",
        "cumulative mesh-build time vs reconstructed-uh norm",
        "sphere_meshbuildtime.png",
    )

    # effectivity index = estimator / (true error), in whichever norm the
    # estimator targets (energy norm for BR78, sup norm for NSV03)
    # a reliable and efficient estimator should stay O(1) as DOFs grow
    plt.figure()
    effstylemap = {"udobr": ("ko", "BR78 (inactive-set energy norm)"),
                    "nsv": ("bs", "NSV03 (sup norm)"),
                    "nsvsafe": ("md", "NSV03+safe (sup norm)")}
    for amrtype in ("udobr", "nsv", "nsvsafe"):
        if amrtype not in effs or len(effs[amrtype][0]) == 0:
            continue
        edofs, evals = np.array(effs[amrtype][0]), np.array(effs[amrtype][1])
        style, label = effstylemap[amrtype]
        plt.semilogx(edofs, evals, style, label=label)
    plt.axhline(1.0, color="k", linestyle=":", label="ideal eff=1")
    plt.legend()
    plt.grid(True)
    plt.xlabel("DOFs")
    plt.ylabel("effectivity ratio (estimator / true-error)")
    plt.title("effectivity index vs DOFs")
    plt.savefig("sphere_effectivity.png")
