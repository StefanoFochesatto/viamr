# This example attempts to do apples-to-apples comparisons of 5 algorithms
# on the "ball" problem:
#   1. UDOBR = unstructured dilation operator plus Babuska & Rheinboldt (1989) in the inactive set
#   2. NSV = Nochetto, Siebert, and Veeser (2003)
#   3. NSVSAFE = a promising algorithm: NSV, with VIAMR.safeactiveunmark()
#      excluding certified-safe active elements from nsvmark()'s marking
#      threshold (not merely from its output mark; see viamr.py's
#      nsvmark() docstring).  On this problem, the spherical-cap obstacle
#      is superharmonic throughout the active set with f=0, so
#      safeactiveunmark() is expected to certify the whole active set safe
#      at every level; NSVSAFE should then track NSV's accuracy while
#      avoiding its unbounded active-set refinement.
#   4. UNI = uniform refinement
#   5. AVM = averaged-metric mesh adaptation  <-- only runs if avm import succeeds
#
# We generate .pvd files: result_sphere_{udobr,nsv,nsvsafe,uni,avm}.pvd
#
# The exact solution is known and so we can compute norm convergence rates.
# We generate seven .png convergence figures comparing all methods:
#   sphere_convergence_unorm.png       ||u_exact - u_h||_2 vs DOFs
#   sphere_convergence_preferred.png   ||u_exact - tilde u_h||_2 vs DOFs
#   sphere_convergence_h1.png          |u_exact - u_h|_{H^1 seminorm} vs DOFs
#   sphere_convergence_supnorm.png     ||u_exact - u_h||_infty vs DOFs, for all methods
#   sphere_convergence_supnorm_preferred.png
#                                       ||u_exact - tilde u_h||_infty vs DOFs, for all methods
#   sphere_convergence_hausdorff.png   hausdorff2D(Gamma_u, Gamma_uh) vs DOFs
#   sphere_effectivity.png             estimator/true-error effectivity index
#                                       vs DOFs, for UDOBR, NSV, and NSVSAFE only
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


def errornorm_preferred_deg(r, uh, activeh, fixeddegree=6):
    """L^2 error norm against "preferred" form of numerical solution"""
    tildeuh = conditional(eq(activeh, 1.0), psiUFL(r), uh)  # preferred
    # high degree quadrature important in next line; setting it avoids TSFC warning
    normsq = assemble((uexactUFL(r) - tildeuh) ** 2 * dx(degree=fixeddegree))
    return np.sqrt(normsq)


def errornorm_H1semi_deg(u, uh, fixeddegree=6):
    """H^1 seminorm (i.e. Dirichlet-energy / grad-L^2) error norm of the plain
    (not "preferred") numerical solution, avoiding the TSFC warning.  This is
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


def errornorm_Linf_preferred(amr, r, uh, activeh, pdegree=4):
    """Sup-norm (L^infty) error against the "preferred" form of the
    numerical solution (see errornorm_preferred_deg), rather than against
    the plain uh.  This is the right comparison for methods (e.g. UDOBR,
    NSVSAFE) that deliberately leave the active-set interior coarse, since
    u=psi there regardless of resolution: the plain sup norm is then
    dominated by the (known, not-actually-erroneous) gap between the coarse
    uh and the true psi, rather than by genuine solution error.  Uses the
    same CG^p-interpolation + VIAMR.scalarrange() technique as
    errornorm_Linf(); see its docstring re. non-polynomial data."""
    tildeuh = conditional(eq(activeh, 1.0), psiUFL(r), uh)  # preferred
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

results = {}  # results[amrtype] = (dofs, errnorm, errnorm_pres, hausdorffs, errnorm_h1s,
# errnorm_Linfs, errnorm_Linf_pres), for the six convergence figures generated at the end
effs = {}  # effs[amrtype] = (eff_dofs, eff_vals), for udobr/nsv/nsvsafe only; see
# the effectivity-index figure generated at the end

for amrtype in refinetypes:
    print(f"solving by VIAMR using {amrtype.upper()} method ...")

    amr = VIAMR()
    dofs, errnorms, errnorm_pres, hausdorffs, errnorm_h1s, errnorm_Linfs, errnorm_Linf_pres = (
        [], [], [], [], [], [], []
    )
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
        errnorm = errornorm_deg(uexactUFL(r), uh)
        activeh = amr.elemactive(uh, lb)
        errnorm_pre = errornorm_preferred_deg(r, uh, activeh)
        errnorm_h1 = errornorm_H1semi_deg(uexactUFL(r), uh)
        errnorm_Linf = errornorm_Linf(amr, uexactUFL(r), uh)
        errnorm_Linf_pre = errornorm_Linf_preferred(amr, r, uh, activeh)
        print(f"  ||u_exact - u_h||_2 = {errnorm:.3e}")
        print(f"  ||u_exact - tilde u_h||_2 = {errnorm_pre:.3e}")
        print(f"  |u_exact - u_h|_{{H^1}} = {errnorm_h1:.3e}")
        print(f"  ||u_exact - u_h||_infty = {errnorm_Linf:.3e}")
        print(f"  ||u_exact - tilde u_h||_infty = {errnorm_Linf_pre:.3e}")
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
        errnorm_pres.append(errnorm_pre)
        hausdorffs.append(haus if haus is not None else np.nan)  # nan plots as a gap
        errnorm_h1s.append(errnorm_h1)
        errnorm_Linfs.append(errnorm_Linf)
        errnorm_Linf_pres.append(errnorm_Linf_pre)
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
        elif amrtype in ("nsv", "nsvsafe"):
            g = Function(V).interpolate(g_ufl)
            safe = None
            if amrtype == "nsvsafe":
                # compute safe *before* nsvmark(), so its eta values are
                # excluded from the marking threshold itself, not merely
                # filtered from the mark afterward (see nsvmark() docstring)
                safe = amr.safeactiveunmark(uh, lb, F_strong, psiUFL(r), Constant(0.0))
                print(f"  safeactiveunmark() excludes {amr.countmark(safe)} "
                      "active elements from the nsvmark() threshold")
            (mark, etainf, _, _, _) = amr.nsvmark(
                uh, lb, g, Constant(0.0), g_ufl, safe=safe
            )
            # effectivity index vs NSV03's own target norm: max_T eta_inf(T)
            # is the whole-domain (not inactive-set-restricted) quantity its
            # pointwise theory bounds ||u-u_h||_infty by, in contrast to
            # BR78/BV00's energy-norm estimator below; errnorm_Linf was
            # already computed above, for this same uh
            maxetainf = amr.scalarrange(etainf)[1]
            eff_nsv = maxetainf / errnorm_Linf if errnorm_Linf > 0 else np.nan
            print(f"  eff_{amrtype.upper()} (sup norm) = {eff_nsv:.3f}")
            eff_dofs.append(Nv)
            eff_vals.append(eff_nsv)
            mesh = amr.refinemarkedelements(mesh, mark)
        else:
            mark = amr.udomark(uh, lb, n=1)
            residual = -div(grad(uh))
            (imark, _, tot_eta) = amr.brinactivemark(uh, lb, residual, theta=thetaBR, method=methodBR)
            mark = amr.unionmarks(mark, imark)
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
            mesh = amr.refinemarkedelements(mesh, mark)
        if amrtype != "uni":
            refinetime = time.time() - start_time
        meshHist.append(mesh)

    if writecsvs:
        csvfile.close()

    results[amrtype] = (
        dofs, errnorms, errnorm_pres, hausdorffs, errnorm_h1s, errnorm_Linfs, errnorm_Linf_pres
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
        "hausdorff2D(Gamma_u, Gamma_uh)",
        "sphere problem: free-boundary Hausdorff distance vs DOFs",
        "sphere_convergence_hausdorff.png",
        -1.0 / d,
        "DOFs^(-1/d)",
    )
    # H^1 seminorm = energy norm: a priori theory for obstacle problem gives
    # first-order convergence, DOFs^(-1/d), BR78 estimator targets this norm
    _convergence_plot(
        4,
        "|u_exact - u_h|_{H^1}",
        "sphere problem: H^1 seminorm error vs DOFs",
        "sphere_convergence_h1.png",
        -1.0 / d,
        "DOFs^(-1/d)",
    )
    # sup (L^infty) norm: this is the norm NSV03's pointwise a posteriori
    # theory targets (see errornorm_Linf's docstring), so unlike the
    # effectivity-index figure below (estimator/true-error ratio, NSV-family
    # methods only) this figure gives an apples-to-apples true-error
    # comparison, in that same norm, across *all* methods.  Reference rate
    # DOFs^(-2/d) matches the L^2 rate above: for smooth solutions, CG1
    # pointwise error is generically the same asymptotic order as L^2
    # (up to log factors), and NSV03's own effectivity results (see
    # sphere_effectivity.png) are consistent with that scaling here.
    _convergence_plot(
        5,
        "||u_exact - u_h||_infty",
        "sphere problem: sup-norm error vs DOFs",
        "sphere_convergence_supnorm.png",
        -2.0 / d,
        "DOFs^(-2/d)",
    )
    # "preferred"-form companion to the plain sup-norm figure above (see
    # errornorm_Linf_preferred()): the right comparison for UDOBR and
    # NSVSAFE, which deliberately leave the active-set interior coarse, so
    # their plain sup norm above is dominated by the known (not genuinely
    # erroneous) gap between coarse uh and the true psi there.  Included for
    # all methods, not just those two, as a consistency check: NSV/UNI/AVM
    # already refine the active set, so their curves should barely move.
    _convergence_plot(
        6,
        "||u_exact - tilde u_h||_infty",
        "sphere problem: preferred-form sup-norm error vs DOFs",
        "sphere_convergence_supnorm_preferred.png",
        -2.0 / d,
        "DOFs^(-2/d)",
    )

    # effectivity index = estimator / true error, in whichever norm the
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
    plt.title("sphere problem: effectivity index vs DOFs")
    plt.savefig("sphere_effectivity.png")
