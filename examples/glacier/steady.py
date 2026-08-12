# Solves a 2D steady, isothermal shallow ice approximation glacier obstacle problem.
#
# A basic run to illustrate capabilities:
#   python3 steady.py -refine 4
# Then use paraview 3D visualization, with warp by scalar on the surface elevation,
# on the output file result.pvd.
#
# For more info see README.md or run
#   python3 steady.py -h


from clargs import parser

args, passthroughoptions = parser.parse_known_args()
assert args.m >= 1, "at least one cell in mesh"
assert args.refine >= 0, "cannot refine a negative number of times"
assert args.pcount >= 1, "at least one Picard iteration required"
assert args.udo_n >= 0, "cannot use UDO with negative levels"
assert args.picard or not args.elevdepend, "Picard iteration required for -elevdepend"

import numpy as np
import petsc4py

petsc4py.init(passthroughoptions)
from firedrake import *
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from firedrake.dmhooks import pop_appctx, push_appctx

pprint = PETSc.Sys.Print  # parallel print
from viamr import VIAMR
from realdata import DataNetCDF, ZeroBelowSeaLevel
from physics import (
    secpera,
    n,
    Gamma,
    H2u,
    u2H,
    Phi_u,
    weakform_u,
    residual_u_ufl,
    weakform_s,
    residual_s_ufl,
    surfacevelocity,
    glaciermeshreport,
    solve_params,
    debug,
)
from synthetic import (
    L,
    dome_a_ufl,
    bumps_b_ufl,
    model_a_ufl,
)

# validate -opvdsub and -box options
if args.opvdsub:
    bx = args.box  # [x_left, x_right, y_lower, y_upper]
    assert bx[0] < bx[1] and bx[2] < bx[3], "-box not valid"
    assert args.bdata or (0.0 <= bx[0] < bx[1] <= L), "x range not valid for [0,L]"
    assert args.bdata or (0.0 <= bx[2] < bx[3] <= L), "y range not valid for [0,L]"

# generate mesh, possibly by reading data for bed topography
if args.bdata:
    pprint(f"reading topg from {args.bdata} and ignoring -prob choice ...")
    args.prob = None
    topg_nc = DataNetCDF(args.bdata, "topg")
    # topg_nc.preview()  # generate matplotlib figure
    topg_nc.describe_grid(print=PETSc.Sys.Print, indent=4)
    topg = topg_nc.function(delnear=100.0e3, degree=2)  # topg is in CG2 on the data mesh
    # generate mesh compatible with data mesh, but at user (-m) resolution, typically lower
    mesh, mx, my = topg_nc.rectmesh(args.m)
    pprint(f"putting topg onto Firedrake structured {mx} x {my} data mesh ...")
else:
    pprint(
        f"generating {args.m} x {args.m} initial mesh on square domain, for problem {args.prob} ..."
    )
    mesh = RectangleMesh(
        args.m,
        args.m,
        L,
        L,
        diagonal="crossed",
        distribution_parameters=VIAMR.PARALLEL_OVERLAP,
    )


# encapsulate solve to avoid code duplication
def vinewtonsolve(G, w, bcs=None, lower=None, upper=None):
    problem = NonlinearVariationalProblem(G, w, bcs=bcs)
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=solve_params, options_prefix="steady"
    )
    solver.solve(bounds=(lower, upper))
    return None


# outer mesh refinement loop
amr = VIAMR(debug=True, activetol=1.0)
for i in range(args.refine + 1):
    pprint(f"solving using variable {args.primal} on mesh level {i}:")
    amr.meshreport(mesh, indent=2)

    # describe current mesh
    ne, hmin = glaciermeshreport(amr, mesh)

    # space for most functions
    V, DG0 = amr.spaces(mesh)  # V = CG1
    x = SpatialCoordinate(mesh)

    # bedrock on current mesh
    if args.bdata:
        b = Function(V).project(topg)  # cross-mesh projection from data mesh
        # see ZeroBelowSeaLevel class for how the following works
        _ = pop_appctx(mesh.coordinates.function_space().dm)
        push_appctx(mesh.coordinates.function_space().dm, {"b": b})
    else:
        b = Function(V).interpolate(bumps_b_ufl(x, problem=args.prob))  # synthetic bed
    b.rename("b = bedrock topography")

    # surface mass balance function on current mesh; depends on b in one case
    if args.bdata:
        # SMB from linear model based on lapse rate; from linearizing dome case
        c0 = -3.4e-8
        c1 = (6.3e-8 - c0) / 3.6e3
        a_lapse = c0 + c1 * topg
        a = Function(V).interpolate(a_lapse)
    elif args.elevdepend:
        # initialize from s = b assumption
        a = Function(V).interpolate(model_a_ufl(b, sELA=args.sELA))
    else:
        a = Function(V).interpolate(dome_a_ufl(x, problem=args.prob))
    a.rename("a = accumulation")

    # initialize geometry variable u or s; depends on b and a
    if i == 0:
        # build pile of ice from accumulation
        pileage = 400.0  # years
        Hinit = pileage * secpera * conditional(a > 0.0, a, 0.0)
    if args.primal == "u":
        if i == 0:
            uold = Function(V).interpolate(H2u(Hinit))
        else:  # cross-mesh interpolation of previous solution
            uold = Function(V).interpolate(u)
            # remove sign (admissibility) flaws from cross-mesh interpolation
            #   note: u = H^(8/3) < 1 represents *very* little ice
            uold = Function(V).interpolate(conditional(uold < 1.0, 0.0, uold))
        assert amr.checkadmissible(uold, Constant(0.0)), "uold must be non-negative"
        assert assemble(u2H(uold) * dx) > 0, "uold must correspond to positive ice volume"
    else:
        if i == 0:
            sold = Function(V).interpolate(Hinit + b)
        else:  # cross-mesh interpolation of previous solution
            sold = Function(V).interpolate(s)
            # remove admissibility flaws from cross-mesh interpolation
            sold = Function(V).interpolate(conditional(sold < b, b, sold))
        assert amr.checkadmissible(sold, b), "sold must be >= b"
        assert assemble((sold - b) * dx) > 0, "sold must correspond to positive ice volume"

    # set-up for solve on current mesh
    if args.primal == "u":
        u = Function(V, name="u = transformed thickness").interpolate(uold)
        lb = Function(V).interpolate(Constant(0.0))
        bcs = [
            DirichletBC(V, Constant(0.0), "on_boundary"),
        ]
        if args.bdata:
            bcs.append(ZeroBelowSeaLevel(V, Constant(0.0)))
    else:
        s = Function(V, name="s = surface elevation").interpolate(sold)
        lb = b
        bcs = [
            DirichletBC(V, b, "on_boundary"),
        ]
        if args.bdata:
            bcs.append(ZeroBelowSeaLevel(V, b))
    ub = Function(V).interpolate(Constant(PETSc.INFINITY))

    # solve, by Picard solve loop or directly by Newton
    if args.picard:
        if args.primal == "u":
            # both freeze-tilt iteration, and a=a(s) iteration if -elevdepend
            for k in range(args.pcount):
                if args.elevdepend:
                    sold = b + u2H(uold)
                    a = Function(V).interpolate(model_a_ufl(sold, sELA=args.sELA))
                    a.rename("a = accumulation")
                Ztilt = Phi_u(uold, b)
                F = weakform_u(u, a, b, Z=Ztilt)
                vinewtonsolve(F, u, bcs=bcs, lower=lb, upper=ub)
                uold = Function(V).interpolate(u)
        else:
            # only a=a(s) iteration if -elevdepend
            for k in range(args.pcount):
                if args.elevdepend:
                    a = Function(V).interpolate(model_a_ufl(sold, sELA=args.sELA))
                    a.rename("a = accumulation")
                F = weakform_s(s, a, b)
                vinewtonsolve(F, s, bcs=bcs, lower=lb, upper=ub)
                sold = Function(V).interpolate(s)
    else:
        # solve directly by Newton; not attempted if -elevdepend
        if args.primal == "u":
            F = weakform_u(u, a, b)
            vinewtonsolve(F, u, bcs=bcs, lower=lb, upper=ub)
        else:
            F = weakform_s(s, a, b)
            vinewtonsolve(F, s, bcs=bcs, lower=lb, upper=ub)

    # update geometry variables
    if args.primal == "u":
        H = Function(V, name="H = thickness").interpolate(u2H(u))
        s = Function(V, name="s = surface elevation").interpolate(b + H)
    else:
        H = Function(V, name="H = thickness").interpolate(s - b)
        H = H.interpolate(conditional(H > 1.0, H, 0.0))  # force exact zero outside ice
        u = Function(V, name="u (transformed thickness)").interpolate(H2u(H))

    # report glaciated area and Jaccard inactive set agreement
    vol = assemble(H * dx)
    ei = amr.eleminactive(H, Constant(0.0))
    area = assemble(ei * dx)
    pprint(
        f"  ice area {area / 1000.0**4:.3f} million km^2;  ice vol = {vol / 1000.0**5:.3f} million km^3",
        end="",
    )
    if i > 0:
        jac = amr.jaccard(ei, oldei, submesh=True)
        pprint(f";  Jaccard({i-1},{i}) = {100*jac:.2f}%")
    else:
        pprint("")
    oldei = ei

    # mark and refine based on constraint H >= 0 (or u >= 0 or s >= b)
    uni = i < args.uniform
    if uni:
        mark = Function(DG0).interpolate(Constant(1.0))
    else:
        if args.primal == "u":
            fbmark = amr.udomark(H, Constant(0.0), n=args.udo_n)
            res, Z = residual_u_ufl(u, a, b)
            imark, _, _ = amr.brinactivemark(
                u, Constant(0.0), res, theta=args.theta, method="total", alpha=Z
            )
        else:
            fbmark = amr.udomark(s, b, n=args.udo_n)
            res, Z = residual_s_ufl(s, a, b)
            # use u > 0 when calling brinactivemark(), so "jump(grad(u))" is for u, even though eta calculated with s
            imark, _, _ = amr.brinactivemark(
                s, b, res, theta=args.theta, method="total", alpha=Z
            )
        if args.hmin > 0.0:
            fbmark = amr.lowerboundcelldiameter(fbmark, args.hmin)
            imark = amr.lowerboundcelldiameter(imark, args.hmin)
        mark = amr.unionmarks(fbmark, imark)
        # report percentages of elements marked
        pfb = 100.0 * amr.countmark(fbmark) / ne
        pin = 100.0 * amr.countmark(imark) / amr.countmark(ei)

        # report on AMR
        pprint(
            f"  elements marked by UDO+BR weighted: {pfb:.2f}% free-bdry, {pin:.2f}% of inactive"
        )

    # refine (when proceeding to next mesh)
    if i < args.refine:
        pprint(f"  refining" + (" uniformly" if uni else "") + " ...")
        mesh = amr.refinemarkedelements(mesh, "uniform" if uni else mark)

if args.opvdsub:  # note boxind gets written into -opvd file
    x, y = SpatialCoordinate(mesh)
    bx = args.box  # [x_left, x_right, y_lower, y_upper]
    ibx = conditional(x >= bx[0], conditional(x <= bx[1], 1.0, 0.0), 0.0)
    iby = conditional(y >= bx[2], conditional(y <= bx[3], 1.0, 0.0), 0.0)
    _, DG0 = amr.spaces(mesh)
    boxind = Function(DG0, name="box indicator").interpolate(ibx * iby)

if args.opvd:
    Us = Function(VectorFunctionSpace(mesh, "CG", degree=2))
    Us.project(secpera * surfacevelocity(s, H))  # smoother than .interpolate()
    Us.rename("Us = surface velocity (m/a)")
    q = Function(FunctionSpace(mesh, "BDM", 1)).interpolate(Us * H)
    q.rename("q = UH (ice flux)")
    grads = Function(VectorFunctionSpace(mesh, "DG", degree=0))
    grads.interpolate(grad(s))
    grads.rename("Gs = grad(s)")
    fields = [H, s, u, Us, q, a, b, grads, mark, fbmark, imark]
    if args.opvdsub:
        fields.append(boxind)
    if mesh.comm.size > 1:
        rank = Function(FunctionSpace(mesh, "DG", 0))
        rank.dat.data[:] = mesh.comm.rank
        rank.rename("rank")
        fields.append(rank)
    pprint("writing solution fields to %s ..." % args.opvd)
    VTKFile(args.opvd).write(*fields)

if args.opvdsub:
    mesh.mark_entities(boxind, 99)
    mesh = RelabeledMesh(mesh, [boxind], [99])
    subm = Submesh(mesh, mesh.topological_dimension, 99)
    subV = FunctionSpace(subm, "CG", 1)
    subu = Function(subV, name="u = transformed thickness").interpolate(u)
    subH = Function(subV, name="H = thickness").interpolate(H)
    subs = Function(subV, name="s = surface elevation").interpolate(s)
    suba = Function(subV, name="a = accumulation").interpolate(a)
    subb = Function(subV, name="b = bedrock topography").interpolate(b)
    pprint("writing submesh quantities, for -box, to %s ..." % args.opvdsub)
    VTKFile(args.opvdsub).write(subu, subH, subs, suba, subb)
