# This example attempts to do apples-to-apples comparisons of 5 algorithms
# on a classical obstacle problem with a hemispherical obstacle:
#
#   1. UDOBR = unstructured dilation operator plus Babuska & Rheinboldt (1989)
#              in the inactive set
#   2. NSV = Nochetto, Siebert, and Veeser (2003)
#   3. NSVSAFE = NSV plus VIAMR.safeactiveunmark(), excluding certified-safe
#                active elements from fixedratemark()'s marking threshold
#   4. UNI = uniform refinement
#   5. AVM = averaged-metric mesh adaptation
#
# The AVM method only runs if "import animate" succeeds.
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
#   sphere_meshbuildtime.png          same, but mesh-construction cost (PETSc or Mmg/ParMmg)
#   sphere_effectivity.png            estimator/true-error effectivity index vs DOFs
#
# Optionally we generate .csv files for norm and Jaccard convergence rates.
#
# Optional -dimples mode (with -fconst) replaces the plain spherical cap
# by that cap with scattered downward gaussian dimples subtracted at
# buckyball-vertex centers, and/or turns on a nonzero constant source f.  In this
# mode there is no closed-form exact solution inside r<=r0, so norm/effectivity figures
# are skipped; only produces:
#   sphere_dimple_activefraction.png  active-element fraction vs DOFs
#   sphere_dimple_inactivecount.png   inactive elements found inside dimples vs DOFs
# See obstacleUFL() and dimple_centers().
#
# Optional -porous mode replaces the classical operator (Lu = -div(grad(u))) by
# the degenerate porous-media operator Lu = -div((u-psi)^{gamma-1} grad(u)).
# The degenerating coefficient is regularized by eps-continuation.  There is
# no closed-form exact solution for this operator, so norm/effectivity figures
# are skipped.  Also, NSV and NSVSAFE assume the Laplacian internally (see
# nsvmark()) and so are not meaningful here; only UDOBV, UNI, and AVM are
# compared.
#
# Three suggested runs to get familiar with the major cases:
#   python3 sphere.py                        exact soln known; Laplacian
#   python3 sphere.py -dimples -fconst -1.0  no exact soln; Laplacian, dimples on obstacle
#   python3 sphere.py -porous                no exact soln; porous-type operator


from argparse import ArgumentParser

parser = ArgumentParser(
    description="Apples-to-apples AMR comparison on a sphere obstacle problem with a known exact solution.  Optional form has a dimpled obstacle and allows a constant source term."
)
parser.add_argument(
    "-dimpleamp",
    type=float,
    default=0.08,
    metavar="X",
    help="gaussian dimple depth [default=0.08]",
)
parser.add_argument(
    "-dimples",
    action="store_true",
    default=False,
    help="add gaussian dimples (buckyball vertices) to the obstacle",
)
parser.add_argument(
    "-dimplesigma",
    type=float,
    default=0.07,
    metavar="X",
    help="gaussian dimple width [default=0.07]",
)
parser.add_argument(
    "-epsfinal",
    type=float,
    default=0.0005,
    metavar="X",
    help="-porous only: final eps in eps-continuation [default=0.0005]",
)
parser.add_argument(
    "-epsshrink",
    type=float,
    default=0.3,
    metavar="X",
    help="-porous only: geometric shrink factor in eps-continuation [default=0.3]",
)
parser.add_argument(
    "-epsstart",
    type=float,
    default=0.1,
    metavar="X",
    help="-porous only: initial eps in eps-continuation [default=0.1]",
)
parser.add_argument(
    "-fconst",
    type=float,
    default=0.0,
    metavar="X",
    help="constant source term f, e.g. in strong form -div(grad(u))=f [default=0.0]",
)
parser.add_argument(
    "-gamma",
    type=float,
    default=2.0,
    metavar="X",
    help="-porous only: exponent in the porous coefficient (u-psi)^(gamma-1) [default=2.0]",
)
parser.add_argument(
    "-m0",
    type=int,
    default=10,
    metavar="N",
    help="initial mesh is m0 x m0, except AVM [default=10]",
)
parser.add_argument(
    "-maxlevels",
    type=int,
    default=15,
    metavar="N",
    help="backstop on AMR levels [default=15]",
)
parser.add_argument(
    "-methodBR",
    type=str,
    default="total",
    metavar="X",
    choices=["total", "max"],
    help="fixedratemark() method, for BR in inactive set [default=total]",
)
parser.add_argument(
    "-porous",
    action="store_true",
    default=False,
    help="use the degenerate porous-media operator -div((u-psi)^(gamma-1) grad(u)) - f, "
    "via eps-continuation, in place of the classical Laplacian; restricts the comparison "
    "to udobr (weighted BV00)/uni/avm, since nsv/nsvsafe require the Laplacian [default=False]",
)
parser.add_argument(
    "-targetelements",
    type=float,
    default=1.0e5,
    metavar="X",
    help="stop refining once this many elements is met [default=1.0e5]",
)
parser.add_argument(
    "-thetaBR",
    type=float,
    default=0.5,
    metavar="X",
    help="fixedratemark() threshold, for BR in inactive set [default=0.5]",
)
parser.add_argument(
    "-uniformlevels",
    type=int,
    default=5,
    metavar="N",
    help="levels of uniform refinement for UNI [default=5]",
)
parser.add_argument(
    "-writecsvs",
    action="store_true",
    default=False,
    help="generate .csv files of convergence rates [default=False]",
)
args, passthroughoptions = parser.parse_known_args()
# uexactUFL only solves the classical (non-porous) Laplace obstacle problem
exact_known = (not args.dimples) and (args.fconst == 0.0) and (not args.porous)

import time
import numpy as np
import petsc4py

petsc4py.init(passthroughoptions)
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
refinetypes = ["udobr", "uni"]
if not args.porous:
    refinetypes += ["nsv", "nsvsafe"]  # need Laplacian operator
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
initialhAVM = 4.0 / args.m0
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


r0 = 0.9  # cap radius: psi is a spherical cap for r<=r0, glued to a linear far-field beyond


def psiUFL(r):
    """obstacle as UFL, from UFL expression for r"""
    psi0 = np.sqrt(1.0 - r0 * r0)
    dpsi0 = -r0 / psi0
    return conditional(le(r, r0), sqrt(1.0 - r * r), psi0 + dpsi0 * (r - r0))


def buckyball_vertices():
    """The 60 vertices of a truncated icosahedron (buckyball), projected onto
    the unit sphere.  Standard construction: all vertices generated from the
    three base triples (0,1,3phi), (1,2+phi,2phi), (phi,2,2phi+1), phi the
    golden ratio, by cyclic permutation and all sign choices on nonzero entries."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    base = [(0.0, 1.0, 3.0 * phi), (1.0, 2.0 + phi, 2.0 * phi), (phi, 2.0, 2.0 * phi + 1.0)]
    verts = []
    for a, b, c in base:
        for p, q, s in [(a, b, c), (b, c, a), (c, a, b)]:
            for sp in ([1.0] if p == 0.0 else [1.0, -1.0]):
                for sq in ([1.0] if q == 0.0 else [1.0, -1.0]):
                    for ss in ([1.0] if s == 0.0 else [1.0, -1.0]):
                        verts.append((sp * p, sq * q, ss * s))
    verts = np.array(verts)
    return verts / np.linalg.norm(verts, axis=1, keepdims=True)


def dimple_centers(zmargin):
    """(x,y,z) unit-sphere centers for gaussian dimples: buckyball vertices
    lying well inside the spherical-cap region r<=r0, i.e. with z >= z0 +
    zmargin where z0 = sqrt(1-r0^2), keeping dimples clear of the C^1 gluing
    radius r0."""
    verts = buckyball_vertices()
    z0 = np.sqrt(1.0 - r0 * r0)
    return verts[verts[:, 2] >= z0 + zmargin]


def chordal2_ufl(x, y, vk):
    """Squared chordal distance, on the unit sphere, between the surface
    point above planar (x,y) -- i.e. (x, y, sqrt(1-r^2)) -- and vertex vk =
    (xk,yk,zk).  Equals 2*(1-cos(angular distance)), so it agrees with
    (angular distance)^2 for small separations, but stays smooth (no
    derivative singularity) all the way down to zero separation.  The mesh domain
    extends well outside the unit disk (r up to 2*sqrt(2)), where 1-x^2-y^2
    is negative.  Clamped to 0 there since obstacleUFL() only ever uses this
    inside r<=r0<1, to avoid a NaN from sqrt(negative) contaminating the rest
    of the (otherwise unrelated) UFL expression."""
    xk, yk, zk = vk
    r2 = x * x + y * y
    z = sqrt(conditional(le(r2, 1.0), 1.0 - r2, 0.0))
    dot = x * xk + y * yk + z * zk
    return 2.0 - 2.0 * dot


def obstacleUFL(x, y, r, dimples=None):
    """Obstacle as UFL: the spherical cap from psiUFL(), optionally with
    round "golf ball" gaussian dimples subtracted at buckyball-vertex
    centers.  Dimples are isotropic *on the sphere*, so they stay circular
    near the cap edge instead of stretching into ellipses.  A deep enough
    dimple makes the cap locally subharmonic (-div(grad(psi))<0) at its
    center, which forces a small inactive island there whenever f is not too
    negative.  The dimple correction is gated to r<=r0, matching where
    psiUFL()'s spherical formula applies."""
    psi = psiUFL(r)
    if dimples is not None:
        dimplesum = Constant(0.0)
        for vk in dimples:
            dimplesum = dimplesum + args.dimpleamp * exp(
                -chordal2_ufl(x, y, vk) / (2.0 * args.dimplesigma**2)
            )
        psi = psi - conditional(le(r, r0), dimplesum, Constant(0.0))
    return psi


def dimple_region_ufl(x, y, dimples, radius):
    """Indicator (as UFL) of the union of geodesic disks of the given
    angular radius around each dimple center.  Diagnostic only, for counting
    inactive elements discovered inside dimples; not used in constructing
    the obstacle."""
    ind = Constant(0.0)
    for vk in dimples:
        ind = conditional(le(chordal2_ufl(x, y, vk), radius**2), Constant(1.0), ind)
    return ind


def uexactUFL(r):
    """exact solution (and boundary conditions) as UFL, from UFL expression for r"""
    afree = 0.697965148223374
    A, B = 0.680259411891719, 0.471519893402112
    return conditional(le(r, afree), psiUFL(r), -A * ln(r) + B)


def F_strong(u, f):
    """Strong form of the classical obstacle problem operator: L(u) - f.
    Used by VIAMR.safeactiveunmark() for the NSVSAFE method.  Not used
    if -porous option is set."""
    return -div(grad(u)) - f


def porous_residual_Z(uh, lb):
    """Strong-form residual and BV00 weight for the degenerate porous-type
    operator -div((u-psi)^{gamma-1} grad(u)) - f.  The residual is
    regularized by args.epsfinal because div() differentiates symbolically,
    producing gamma-2 as a power; for gamma<2 that is negative and uh==lb
    throughout the active set, giving 0**(negative) = inf/nan.  The weight
    Z, passed as the alpha kwarg for the weighted quasi-norm in
    brinactivemark(), is regularized by the mesh-native cell diameter."""
    gap = uh - lb
    Zeps = abs(gap + Constant(args.epsfinal)) ** (args.gamma - 1.0)
    residual = -div(Zeps * grad(uh)) - Constant(args.fconst)
    hT = CellDiameter(uh.function_space().mesh())
    Zqn = abs(gap + hT) ** (args.gamma - 1.0)
    return residual, Zqn


def activeexactUFL(r):
    """exact active set as UFL, from UFL expression for r (when exact_known)"""
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


def applybr(uh, lb):
    """avoid code duplication by encapsulating steps associated
    to residual estimator marking in inactive set"""
    if args.porous:
        residual, Z = porous_residual_Z(uh, lb)
    else:
        residual = -div(grad(uh)) - Constant(args.fconst)
        Z = None
    (imark, _, tot_eta) = amr.brinactivemark(
        uh, lb, residual, theta=args.thetaBR, method=args.methodBR, alpha=Z
    )
    return imark, tot_eta


# solver parameters for VI
sp = {
    "snes_type": "vinewtonrsls",
    "snes_vi_zero_tolerance": 1.0e-12,
    "snes_linesearch_type": "bt" if args.porous else "basic",
    "snes_linesearch_order": 1,  # not used if type is "basic"
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

dimples = dimple_centers(2.0 * args.dimplesigma) if args.dimples else None
if args.dimples:
    print(
        f"dimpled obstacle: {len(dimples)} gaussian dimples (buckyball vertices), "
        f"amp={args.dimpleamp}, sigma={args.dimplesigma} rad, f={args.fconst}"
    )
if args.porous:
    print(
        f"porous-media operator: gamma={args.gamma}, eps-continuation "
        f"{args.epsstart} -> {args.epsfinal} by shrink={args.epsshrink}"
    )
    print(f"comparison restricted to {[t.upper() for t in refinetypes]}")

# do runs and collect data
results = {}
effs = {}
diagresults = {}
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
    activefracs = []  # active-element fraction vs level; used only if not exact_known
    dimpleinactivecounts = []  # inactive elements found inside dimples; -dimples & not exact_known only

    # freeboundarygraph2D(), used below for the Hausdorff-distance figure,
    # needs PARALLEL_OVERLAP to give correct results in parallel
    dp = VIAMR.PARALLEL_OVERLAP

    # create initial mesh
    if amrtype == "avm":
        geo = SplineGeometry()
        geo.AddRectangle(p1=(-2, -2), p2=(2, 2), bc="rectangle")
        ngmsh = geo.GenerateMesh(maxh=initialhAVM)
        mesh0 = Mesh(ngmsh, distribution_parameters=dp)
    else:
        mesh0 = RectangleMesh(
            args.m0,
            args.m0,
            Lx=2.0,
            Ly=2.0,
            originX=-2.0,
            originY=-2.0,
            diagonal="crossed",
            distribution_parameters=dp,
        )
    meshHist = [mesh0]

    if amrtype == "uni":
        unimh = MeshHierarchy(mesh0, args.uniformlevels)

    if args.writecsvs:
        csvfile = open(f"sphere_{amrtype}.csv", "w")
        csvfile.write(
            "I,NV,NE,HMIN,HMAX,ENORM,ENORMRECON,ENORMH1,JACCARD,HAUSDORFF,"
            "REFINETIME,MARKTIME,MESHBUILDTIME\n"
        )

    for i in range(args.maxlevels + 1):
        print(f"solving on mesh {i} ...")
        mesh = meshHist[i]
        comm = mesh.comm  # fixed across this level, even once `mesh` is reassigned below
        amr.meshreport(mesh)
        V = FunctionSpace(mesh, "CG", 1)
        _, _, hmin, _ = amr.meshsizes(mesh)
        if i == 0:
            uh = Function(V, name="u_h")
            refinetime = 0.0
            marktime = 0.0
            meshbuildtime = 0.0
        else:
            # initialize by cross-mesh interpolation to fine mesh; uses OLD
            # data uh, lb (still bound to the previous mesh at this point)
            uUFL = conditional(uh < lb, lb, uh)
            uh = Function(V, name="u_h").interpolate(uUFL)

        # obstacle and initial iterate on the current mesh
        x, y = SpatialCoordinate(mesh)
        r = sqrt(x * x + y * y)
        psi_expr = obstacleUFL(x, y, r, dimples=dimples)
        lb = Function(V, name="psi").interpolate(psi_expr)
        ub = Function(V).interpolate(Constant(PETSc.INFINITY))
        if args.porous:
            # push the iterate strictly above the (current-mesh) obstacle,
            # needed for the degenerate coefficient
            uh = Function(V, name="u_h").interpolate(
                conditional(uh < lb, lb, uh) + Constant(args.epsstart)
            )

        # weak form
        v = TestFunction(V)
        g_ufl = uexactUFL(r)
        bcs = DirichletBC(V, g_ufl, "on_boundary")
        if args.porous:
            # weak form for the degenerate porous-media operator, with epsilon
            # regularization; epsC is updated in-place by the continuation loop
            epsC = Constant(args.epsstart)
            Z = abs(uh - lb + epsC) ** (args.gamma - 1.0)
            F = (Z * inner(grad(uh), grad(v)) - Constant(args.fconst) * v) * dx
        else:
            F = (inner(grad(uh), grad(v)) - Constant(args.fconst) * v) * dx

        # solve
        problem = NonlinearVariationalProblem(F, uh, bcs)
        solver = NonlinearVariationalSolver(
            problem, solver_parameters=sp, options_prefix="s"
        )
        if args.porous:
            # eps-continuation: step eps down geometrically from epsstart to
            # epsfinal, warm-starting uh from the previous step.
            # The eps floor is raised when the mesh is finer than epsfinal: below that
            # scale Z=(u-psi+eps)^(gamma-1) varies by orders of magnitude *within a
            # single element* near the free boundary, which causes Newton to diverge
            # on fine meshes.  Compare porous.py.
            epsfinal_eff = max(args.epsfinal, 0.2 * hmin ** (2.0 / args.gamma))
            if epsfinal_eff > args.epsfinal:
                print(f"  eps-continuation floor raised to {epsfinal_eff:.2e} > epsfinal={args.epsfinal:.2e}")
            epsval = args.epsstart
            while True:
                epsC.assign(epsval)
                solver.solve(bounds=(lb, ub))
                if epsval <= epsfinal_eff:
                    break
                epsval = max(epsfinal_eff, epsval * args.epsshrink)
        else:
            solver.solve(bounds=(lb, ub))

        # report, and break if target complexity met
        Nv, Ne, hmin, hmax = amr.meshsizes(mesh)
        dofs.append(Nv)
        activeh = amr.elemactive(uh, lb)
        Na = amr.countmark(activeh)
        print(f"  active elements: {Na} ({100.0 * Na / Ne:.2f} %)")

        # compute error norms if possible
        if exact_known:
            errnorm = errornorm_deg(uexactUFL(r), uh)
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
            print(f"  jaccard(A_uexact, A_uh) = {jaccard:.5f}")
            uexact = Function(V, name="u_exact").interpolate(uexactUFL(r))
            _, fbexact = amr.freeboundarygraph2D(uexact, lb)
            _, fb = amr.freeboundarygraph2D(uh, lb)
            # hausdorff2D() returns None for an empty free boundary (e.g. a coarse
            # initial mesh); format accordingly
            haus = amr.hausdorff2D(fbexact, fb)
            hausstr = f"{haus:.5f}" if haus is not None else "n/a"
            print(f"  hausdorff2D(Gamma_u, Gamma_uh) = {hausstr}")
            errnorms.append(errnorm)
            errnorm_recons.append(errnorm_recon)
            hausdorffs.append(haus if haus is not None else np.nan)  # nan plots as a gap
            errnorm_h1s.append(errnorm_h1)
            errnorm_Linfs.append(errnorm_Linf)
            errnorm_Linf_recons.append(errnorm_Linf_recon)
        else:
            activefracs.append(Na / Ne)
            if args.dimples:
                iamark = amr.eleminactive(uh, lb, strong=True)
                W0 = iamark.function_space()
                # angular radius for the "inside a dimple" diagnostic region:
                # 3 sigma captures each dimple's meaningful footprint
                dimregion = Function(W0).interpolate(
                    dimple_region_ufl(x, y, dimples, 3.0 * args.dimplesigma)
                )
                n_dimple_inactive = amr.countmark(
                    Function(W0).interpolate(iamark * dimregion)
                )
                dimpleinactivecounts.append(n_dimple_inactive)
                print(f"  inactive elements inside dimples: {n_dimple_inactive}")

        # refinetime is the AMR cost (marking + refinement) that produced THIS
        # mesh from the previous one (0.0 at i=0); accumulating it here.
        # marktime/meshbuildtime split it into VIAMR-controlled cost (marking,
        # or AVM's metric-building) vs the external backend's mesh-construction
        # cost (PETSc SBR / Mmg/ParMmg).  marktime + meshbuildtime <= refinetime;
        # see the AMR-step comment below for why they need not sum exactly.
        cumamrtime += refinetime
        amrtimes.append(cumamrtime)
        cummarktime += marktime
        marktimes.append(cummarktime)
        cummeshbuildtime += meshbuildtime
        meshbuildtimes.append(cummeshbuildtime)
        if args.writecsvs and exact_known:
            csvfile.write(
                f"{i},{Nv},{Ne},{hmin:.5f},{hmax:.5f},{errnorm:.3e},{errnorm_recon:.3e},{errnorm_h1:.3e},{jaccard:.5f},{hausstr},"
                f"{refinetime:.3e},{marktime:.3e},{meshbuildtime:.3e}\n"
            )
        if Ne > args.targetelements:
            break

        # Do an AMR level.  Each non-uni branch times two phases separately:
        #   "mark" = whatever VIAMR itself computes (marking fields, or AVM's
        #            metric construction including vcdmark())
        #   "meshbuild" = the external backend call that actually builds the
        #            new mesh (PETSc SBR via refinesbr2D(), or Mmg/ParMmg via
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
                safe = amr.safeactiveunmark(uh, lb, F_strong, psi_expr, Constant(args.fconst))
                print(f"  safeactiveunmark() excludes {amr.countmark(safe)} "
                      "active elements from the nsvmark() threshold")
            (mark, etainf, _, Eh, _) = amr.nsvmark(
                uh, lb, g, Constant(args.fconst), g_ufl, safe=safe
            )
            t_mark1 = time.time()
            if exact_known:
                # effectivity index vs NSV03's own target norm: Eh is the estimator
                # Etilde_h of (7.1), the whole-domain (not inactive-set-restricted)
                # bound which the pointwise theory gives for ||u-u_h||_infty
                eff_nsv = Eh / errnorm_Linf if errnorm_Linf > 0 else np.nan
                print(f"  eff_{amrtype.upper()} (sup norm) = {eff_nsv:.3f}")
                eff_dofs.append(Nv)
                eff_vals.append(eff_nsv)
            mesh = amr.refinesbr2D(mesh, mark)
            t_meshbuild1 = time.time()
            marktime_local = t_mark1 - t_mark0
            meshbuildtime_local = t_meshbuild1 - t_mark1
        else:
            # UDO + BR78/BV00 case
            t_mark0 = time.time()
            mark = amr.udomark(uh, lb, n=1)
            imark, tot_eta = applybr(uh, lb)
            mark = amr.unionmarks(mark, imark)
            t_mark1 = time.time()
            if exact_known:
                # effectivity index vs the SAME inactive set brinactivemark()
                # restricts its estimator to; matches the H^1 seminorm BR78's
                # unweighted estimator targets (see errornorm_H1semi_deg)
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
            # true cost of a collective AMR step is bounded by the slowest rank
            refinetime = comm.allreduce(time.time() - start_time, op=MPI.MAX)
            marktime = comm.allreduce(marktime_local, op=MPI.MAX)
            meshbuildtime = comm.allreduce(meshbuildtime_local, op=MPI.MAX)
            # refinetime may slightly exceed mark+meshbuild because of bookkeeping
            print(f"  AMR step: mark {marktime:.3f}s + meshbuild {meshbuildtime:.3f}s <= {refinetime:.3f}s")
        meshHist.append(mesh)

    if args.writecsvs:
        csvfile.close()

    if exact_known:
        results[amrtype] = (
            dofs, errnorms, errnorm_recons, hausdorffs, errnorm_h1s, errnorm_Linfs,
            errnorm_Linf_recons, amrtimes, marktimes, meshbuildtimes,
        )
        effs[amrtype] = (eff_dofs, eff_vals)
    else:
        diagresults[amrtype] = (
            dofs, activefracs, dimpleinactivecounts, amrtimes, marktimes, meshbuildtimes,
        )

    # generate .pvd output file, with AMR fields for final mesh
    outfile = "result_sphere_" + amrtype + ".pvd"
    print(f"done ... writing to {outfile} ...")
    gap = Function(V, name="gap = uh-lb").interpolate(uh - lb)
    fields = [uh, lb, gap]
    if exact_known:
        uexact = Function(V, name="u_exact").interpolate(uexactUFL(r))
        error = Function(V, name="error = |uexact - uh|").interpolate(
            abs(uexact - uh)
        )
        fields += [uexact, error]
    if amrtype == "udobr":
        imark, tot_eta = applybr(uh, lb)
        mark = amr.unionmarks(amr.udomark(uh, lb, n=1), imark)
        imark.rename("imark (BR)")
        mark.rename("mark")
        fields += [mark, imark]
    elif amrtype in ("nsv", "nsvsafe"):
        g = Function(V).interpolate(g_ufl)
        safe = None
        if amrtype == "nsvsafe":
            safe = amr.safeactiveunmark(uh, lb, F_strong, psi_expr, Constant(args.fconst))
            safe.rename("safe active unmarked")
        (mark, etainf, sigmah, _, etad) = amr.nsvmark(
            uh, lb, g, Constant(args.fconst), g_ufl, safe=safe
        )
        mark.rename("mark")
        fields += [mark, sigmah, etainf, etad]
        if amrtype == "nsvsafe":
            fields.append(safe)
    VTKFile(outfile).write(*fields)
    print("")

# Generate convergence/diagnostic figures comparing all methods.  One figure
# per quantity, all methods overlaid.  Accuracy figures need the closed-form
# exact solution.  For -dimples or -porous or a nonzero -fconst there
# isn't one, so we generate lightweight diagnostic figures instead.
print("generating .png plots to compare algorithms ...")
if mesh.comm.rank == 0:
    import matplotlib.pyplot as plt

    d = 2  # spatial dimension
    stylemap = {"udobr": "ko", "nsv": "bs", "nsvsafe": "md", "uni": "rs", "avm": "g^"}

    if exact_known:

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
        # as in nsv.py.  The reconstructed-uh form uses the same L^2 norm as the
        # plain error, rather than H^1, because patched tilde u_h is not H^1-conforming.
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
        # natural reference is mesh-resolution scaling h ~ DOFs^(-1/d).
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
        # Included for all methods, not just those two, for consistency.
        # NSV/UNI/AVM refine the active set, so their curves should match
        # ||u-u_h||_2 earlier.
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
        #   (PETSc SBR via refinesbr2D(), or Mmg/ParMmg via animate.adapt()).
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

    else:
        print("  no closed-form exact solution (-dimples, -porous, and/or -fconst != 0)")
        print("  generating diagnostic figures for active/inactive in center area ...")
        plt.figure()
        for amrtype in refinetypes:
            rdofs = np.array(diagresults[amrtype][0])
            rfrac = 100.0 * np.array(diagresults[amrtype][1])
            plt.semilogx(rdofs, rfrac, stylemap[amrtype], label=amrtype.upper())
        plt.legend()
        plt.grid(True)
        plt.xlabel("DOFs")
        plt.ylabel("active elements (%)")
        plt.title("active-set fraction vs DOFs")
        plt.savefig("sphere_dimple_activefraction.png")

        if args.dimples:
            plt.figure()
            for amrtype in refinetypes:
                rdofs = np.array(diagresults[amrtype][0])
                rcount = np.array(diagresults[amrtype][2])
                plt.semilogx(rdofs, rcount, stylemap[amrtype], label=amrtype.upper())
            plt.legend()
            plt.grid(True)
            plt.xlabel("DOFs")
            plt.ylabel("inactive elements found inside dimples")
            plt.title("dimple-hole discovery vs DOFs")
            plt.savefig("sphere_dimple_inactivecount.png")
