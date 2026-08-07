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
assert args.newton or args.pcount >= 1, "at least one Picard iteration required"
assert args.udo_n >= 0, "cannot use UDO with negative levels"
assert (
    not args.elevdepend or args.prob != "dome"
), "combination invalid: -elevdepend & -prob dome"
assert not args.newton or not args.elevdepend, "-newton invalid (unstable) for -elevdepend"

import numpy as np
import petsc4py

petsc4py.init(passthroughoptions)
from firedrake import *
from firedrake.petsc import PETSc
from pyop2.mpi import MPI

pprint = PETSc.Sys.Print  # parallel print
from viamr import VIAMR
from datanetcdf import DataNetCDF
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
    debug,
)
from synthetic import (
    L,
    dome_L,
    dome_H0,
    dome_s_ufl,
    dome_a_ufl,
    dome_normerrors,
    dome_radiuserror,
    bumps_b_ufl,
    model_a_ufl,
)

if args.opvdsub:
    bx = args.box  # [x_left, x_right, y_lower, y_upper]
    assert bx[0] < bx[1] and bx[2] < bx[3], "-box not valid"
    assert args.bdata or (0.0 <= bx[0] < bx[1] <= L), "x range not valid for [0,L]"
    assert args.bdata or (0.0 <= bx[2] < bx[3] <= L), "y range not valid for [0,L]"

# set up .csv if generating numerical error data
if args.ocsv:
    assert args.prob == "dome", "option -ocsv only valid for -prob dome"
    csvfile = open(args.ocsv, "w")
    print("REFINE,NE,HMIN,UERRH1,HERRINF,DRMAX", file=csvfile)

# read data for bed topography
if args.bdata:
    pprint(f"reading topg from {args.bdata} and ignoring -prob choice ...")
    args.prob = None
    topg_nc = DataNetCDF(args.bdata, "topg")
    # topg_nc.preview()
    topg_nc.describe_grid(print=PETSc.Sys.Print, indent=4)
    pprint(f"putting topg onto matching Firedrake structured data mesh ...")
    topg, nearb = topg_nc.function(delnear=100.0e3)
else:
    pprint(
        f"generating synthetic {args.m} x {args.m} initial mesh for problem {args.prob} ..."
    )

if args.bdata:
    # generate mesh compatible with data mesh, but at user (-m) resolution, typically lower
    mesh = topg_nc.rectmesh(args.m)
else:
    # generate [0,L]^2 mesh via Firedrake
    mesh = RectangleMesh(args.m, args.m, L, L, diagonal="crossed")

# solver parameters
sp = {
    "snes_type": "vinewtonrsls",
    "snes_vi_zero_tolerance": 1.0e-2,  # max u=H^{8/3} is about 10^9; tol is 1 part in 10^-11
    "snes_rtol": 1.0e-6,
    "snes_atol": 1.0e-10,
    "snes_stol": 0.0,
    "snes_linesearch_type": "bt",
    "snes_linesearch_order": "1",
    "snes_max_it": 1000,
    # "snes_max_funcs": 10000,
    "snes_converged_reason": None,
    # "snes_monitor": None,
    # "snes_vi_monitor": None,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def glaciermeshreport(amr, mesh, indent=2):
    nv, ne, hmin, hmax = amr.meshsizes(mesh)
    hmin /= 1000.0
    hmax /= 1000.0
    indentstr = indent * " "
    PETSc.Sys.Print(
        f"{indentstr}current mesh: {nv} vertices, {ne} elements, h in [{hmin:.3f},{hmax:.3f}] km"
    )
    return None


# encapsulate solve to avoid code duplication
def vinewtonsolve(G, w, bcs=None, lower=None, upper=None):
    problem = NonlinearVariationalProblem(G, w, bcs=bcs)
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=sp, options_prefix="s"
    )
    solver.solve(bounds=(lower, upper))
    return None


# outer mesh refinement loop
amr = VIAMR(debug=True, activetol=1.0)
for i in range(args.refine + 1):
    # describe current mesh
    nv, ne, hmin, hmax = amr.meshsizes(mesh)
    pprint(f"solving problem {args.prob} on mesh level {i}:")
    glaciermeshreport(amr, mesh)

    # space for most functions
    V, DG0 = amr.spaces(mesh)  # V = CG1
    x = SpatialCoordinate(mesh)

    # bedrock on current mesh
    if args.bdata:
        b = Function(V).project(topg)  # cross-mesh projection from data mesh
    else:
        if args.prob == "dome":
            b = Function(V).interpolate(Constant(0.0))
        else:
            b = Function(V).interpolate(bumps_b_ufl(x, problem=args.prob))
    b.rename("b = bedrock topography")

    # surface mass balance function on current mesh; depends on b in one case
    if args.bdata:
        # SMB from linear model based on lapse rate; from linearizing dome case
        c0 = -3.4e-8
        c1 = (6.3e-8 - c0) / 3.6e3
        a_lapse = c0 + c1 * topg
        a = Function(V).interpolate(
            conditional(nearb > 0.0, -1.0e-6, a_lapse)
        )  # also cross-mesh re nearb
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
    else:
        s = Function(V, name="s = surface elevation").interpolate(sold)
        lb = b
        bcs = [
            DirichletBC(V, b, "on_boundary"),
        ]
    ub = Function(V).interpolate(Constant(PETSc.INFINITY))

    # solve, or Picard solve loop
    if args.newton:
        if args.primal == "u":
            F = weakform_u(u, a, b)
            vinewtonsolve(F, u, bcs=bcs, lower=lb, upper=ub)
        else:
            F = weakform_s(s, a, b)
            vinewtonsolve(F, s, bcs=bcs, lower=lb, upper=ub)
    else:
        # outer loop for Picard iteration
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
        fbmark = amr.udomark(H, Constant(0.0), n=args.udo_n)
        if args.primal == "u":
            res, Z = residual_u_ufl(u, a, b)
            imark, _, total_eta = amr.brinactivemark(
                u, Constant(0.0), res, theta=args.theta, method="total", alpha=Z
            )
        else:
            res, Z = residual_s_ufl(s, a, b)
            # use u > 0 when calling brinactivemark(), so "jump(grad(u))" is for u, even though eta calculated with s
            imark, _, total_eta = amr.brinactivemark(
                u, Constant(0.0), res, theta=args.theta, method="total", alpha=Z
            )
        if args.hmin > 0.0:
            fbmark = amr.lowerboundcelldiameter(fbmark, args.hmin)
            imark = amr.lowerboundcelldiameter(imark, args.hmin)
        mark = amr.unionmarks(fbmark, imark)
        # report percentages of elements marked
        pfb = 100.0 * amr.countmark(fbmark) / ne
        pin = 100.0 * amr.countmark(imark) / amr.countmark(ei)
        pprint(
            f"  elements marked by UDO+BR weighted: {pfb:.2f}% free-bdry, {pin:.2f}% of inactive"
        )

    # report numerical errors if exact solution known
    if not args.bdata and args.prob == "dome":
        uerr_H1_semi, uerr_H1_rel, Herr_Linf, uexact = dome_normerrors(u, H)
        vfb, _ = amr.freeboundarygraph(u, Function(V).interpolate(0.0))
        drmax = dome_radiuserror(mesh, vfb)
        pprint(
            f"  |u-uexact|_H1rel = {uerr_H1_rel:.3e};  |H-Hexact|_Linf = {Herr_Linf:.3f} m;  |dr|_Linf = {drmax/1000.0:.3f} km",
            end="",
        )
        pprint("" if uni else f";  eff_H1 = {total_eta / uerr_H1_semi:.3f}")
        if args.ocsv:
            print(
                f"{i:d},{ne:d},{hmin:.2f},{uerr_H1_rel:.3e},{Herr_inf:.3f},{drmax:.3f}",
                file=csvfile,
            )

    # refine (when proceeding to next mesh)
    if i < args.refine:
        pprint(f"  refining" + (" uniformly" if uni else "") + " ...")
        mesh = amr.refinemarkedelements(mesh, mark, isUniform=uni)

if args.ocsv:
    csvfile.close()

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
    fields = [H, s, u, Us, q, a, b, grads, mark]
    if not args.bdata and args.prob == "dome":
        uerr = Function(uexact.function_space()).interpolate(u - uexact)
        uerr.rename("uerr = u-u_exact")
        fields.append(uerr)
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
