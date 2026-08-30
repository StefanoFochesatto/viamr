# Compare AMR methods on the two obstacle problems used by Nochetto, Siebert,
# and Veeser to demonstrate their pointwise a posteriori estimators.  Both have
# the Laplacian as the operator and a unilateral lower obstacle.
#
#   -prob easy     "7.2 Example: Constant Obstacle" from NSV03 =
#                    Nochetto, R. H., Siebert, K. G., & Veeser, A. (2003).
#                    Pointwise a posteriori error control for elliptic obstacle
#                    problems.  Numer. Math. 95(1), 163-195.
#                  The obstacle is chi = 0 on the square (or cube) of side 2,
#                  and the exact solution is known, so this problem also
#                  generates convergence and effectivity .png figures.
#
#   -prob pyramid  "3.2 Pyramid obstacle" from NSV05 =
#                    Nochetto, R. H., Siebert, K. G., & Veeser, A. (2005). Fully
#                    localized a posteriori error estimators and barrier sets for
#                    contact problems.  SIAM J. Numer. Anal. 42(5), 2118-2135.
#                  The obstacle is a pyramid on the diamond domain.  No exact
#                  solution is known, so only an estimator .png figure is
#                  generated, and there are no error norms or effectivities.
#
# usage:
#   python3 nsv.py                  easy problem in 2D; generates .png figures
#   python3 nsv.py -dim 3           easy problem in 3D; needs Netgen
#   python3 nsv.py -prob pyramid    pyramid problem

from argparse import ArgumentParser

parser = ArgumentParser(
    description="Compare AMR by UDO+BR, NSV03, and NSV05 on the two obstacle problems from Nochetto, Siebert, & Veeser (2003, 2005)."
)
parser.add_argument(
    "-dim",
    type=int,
    default=2,
    metavar="N",
    choices=[2, 3],
    help="-prob easy only: spatial dimension [default=2]",
)
parser.add_argument(
    "-maxlevels",
    type=int,
    default=12,
    metavar="N",
    help="backstop on AMR levels [default=12]",
)
parser.add_argument(
    "-prob",
    type=str,
    default="easy",
    metavar="X",
    choices=["easy", "pyramid"],
    help="which obstacle problem to solve [default=easy]",
)
parser.add_argument(
    "-targetnodes",
    type=float,
    default=1.0e4,
    metavar="X",
    help="stop refining once this many nodes is met [default=1.0e4]",
)
args, passthroughoptions = parser.parse_known_args()

import numpy as np
import petsc4py

petsc4py.init(passthroughoptions)
from firedrake import *
from viamr import VIAMR
from firedrake.petsc import PETSc

print = PETSc.Sys.Print  # enables correct printing in parallel

methods = ["UDOBR", "NSV03", "NSV05"]

# parameters
nUDO = 0  # observe that {sigma_h * u_h > 0} is same as UDO mark with nUDO=0
dualtol = 1.0e-8  # used for admissibility (sigma_h >= -dualtol) *and* in estimator
marktheta = 0.5
markmethod = "total"

# problem-dependent setup: spatial dimension, methods compared, marking
# strategy, initial mesh, and a data() function which builds the UFL expressions
# and the discrete obstacle on the current mesh
if args.prob == "easy":
    # The domain is the square (-1,1)^2, or the cube (-1,1)^3 with -dim 3.  The
    # obstacle is chi = 0, the boundary values are g = (|x|^2 - r^2)^2 with
    # r = 0.7, and f is chosen so that the exact solution is
    #     u = 0                    for |x| <= r,
    #     u = (|x|^2 - r^2)^2      for |x| >  r.
    # The free boundary is thus the sphere |x| = r, and it is known exactly, so
    # this problem reports error norms and effectivity indices, and generates
    # .png figures.
    #
    # Features of this problem worth knowing:
    #   * u is C^1 but not C^2, since its Hessian jumps across the free boundary.
    #     The two branches of f agree there, both giving -8 r^2, so f is
    #     continuous but only Lipschitz.
    #   * The inactive set is large, and u is smooth and nonzero on most of it,
    #     so much of the error is ordinary P1 interpolation error away from the
    #     contact set.  That is what makes this problem, rather than the pyramid,
    #     the one which pins down the constant in front of the NSV05 residual
    #     term; see the nsv05mark() doc string.
    d = args.dim

    # initial mesh
    if d == 2:
        m = 3  # initial mesh resolution
        mesh0 = RectangleMesh(
            m,
            m,
            1.0,
            1.0,
            originX=-1.0,
            originY=-1.0,
            diagonal="crossed",
            distribution_parameters=VIAMR.PARALLEL_OVERLAP,
        )
    else:
        # 3D SBR refinement needs Netgen mesh and Netgen refinement (and produces bad meshes)
        from netgen.occ import *

        box = Box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
        h0 = 0.5  # initial mesh resolution
        ngmesh = OCCGeometry(box, dim=3).GenerateMesh(maxh=h0)
        mesh0 = Mesh(ngmesh, distribution_parameters=VIAMR.PARALLEL_OVERLAP)

    r = 0.7  # parameter in defining problem

    def data(mesh, V):
        x = SpatialCoordinate(mesh)
        x2 = inner(x, x)
        circle = x2 - r ** 2
        f_ufl = conditional(
            x2 <= r ** 2, -8.0 * r ** 2 * (1.0 - circle), -4.0 * (2.0 * x2 + d * circle)
        )
        g_ufl = circle ** 2  # note this is quartic, so P4 interpolation should be exact
        lb = Function(V, name="chi_h (obstacle)").interpolate(Constant(0.0))
        u_ufl = conditional(x2 <= r ** 2, 0.0, circle ** 2)
        return f_ufl, g_ufl, lb, u_ufl

else:
    # The domain is the diamond Omega, the square (-1,1)^2 rotated 45 degrees,
    # equivalently the \ell^1 unit ball.  The obstacle is the pyramid
    # chi(x) = dist(x, boundary of Omega) - 1/5, the source is f = -5, and the
    # boundary values are g = 0.  No exact solution is known, so this problem
    # only reports the estimators.
    #
    # Features of this problem worth knowing:
    #   * chi is concave and piecewise affine, with ridges along the coordinate
    #     axes, which are the diagonals of the diamond.  Element edges lie on both
    #     axes at every level, because the initial mesh puts them there and SBR
    #     refinement never straddles a ridge, so I_h chi = chi *exactly* and the
    #     obstacle terms of the estimators vanish for a real reason rather than by
    #     assumption.
    #   * The ridge nodes satisfy both sign conditions defining NSV05's
    #     full-contact nodes C_h, namely f <= 0 and a concave kink in u_h = chi_h
    #     giving J_h = -[[grad u_h]].n <= 0, so the ridges are swallowed by
    #     Omega_h^0, where the NSV05 residual indicator vanishes.  The mesh
    #     therefore stays coarse along the diagonals, and refinement there would
    #     mean the sign convention on J_h had been inverted.
    #
    # Figure 3.4 of NSV05 reports DOFs = 5, 381, 3073 at adaptive steps 0, 5, 10,
    # while we get 5, 425, 15642.  Do not read much into the gap: they refine by
    # bisection via ALBERT and mark with the maximum strategy, while we use SBR
    # and Doerfler marking at marktheta, and nsv05mark() keeps the |log h_min|^2
    # factor which section 3 of NSV05 folds into its constant, which marks more.
    d = 2

    # initial mesh: We want the diamond as 4 triangles around the origin, thus 5
    # nodes, matching "step = 0, DOFs = 5" in Figure 3.4 of NSV05.  We build the
    # DMPlex from an explicit cell list, rather than by moving the coordinates of
    # a crossed 1x1 RectangleMesh, because VIAMR.refinesbr2D() transforms
    # mesh.topology_dm and so takes the *DMPlex* coordinates, not the Firedrake
    # coordinate Function; moving only the latter would give a diamond at step 0
    # and squares thereafter.
    cells = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]], dtype=np.int32)
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    plex = PETSc.DMPlex().createFromCellList(2, cells, coords, comm=COMM_WORLD)
    mesh0 = Mesh(plex, distribution_parameters=VIAMR.PARALLEL_OVERLAP)

    def data(mesh, V):
        # dist(x, boundary) = (1 - |x|_1)/sqrt(2) on the ell^1 ball, so the
        # obstacle is a concave pyramid of height 1/sqrt(2) - 1/5 with ridges on
        # the coordinate axes
        x = SpatialCoordinate(mesh)
        f_ufl = Constant(-5.0)
        g_ufl = Constant(0.0)
        chi_ufl = (1.0 - abs(x[0]) - abs(x[1])) / sqrt(2.0) - 0.2
        # discrete obstacle chi_h = I_h chi; exact here, see comment above
        lb = Function(V, name="chi_h (obstacle)").interpolate(chi_ufl)
        return f_ufl, g_ufl, lb, None

# an exact solution is known only for the easy problem, so only it can report
# error norms, effectivity indices, and convergence figures
exact_known = args.prob == "easy"

# all methods use same VI solver
sp = {
    "snes_type": "vinewtonrsls",
    "snes_converged_reason": None,
    # "snes_monitor": None,
    # "snes_view": None,
    "snes_atol": 1.0e-12,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def errornorm_Linf(amr, u, uh):
    """Approximate sup-norm (L^infty) error, via interpolation of the
    (generally non-polynomial) exact-minus-computed difference into a
    higher-degree CG space, then VIAMR.scalarrange() for a parallel-safe
    max; same technique nsv03mark() itself uses internally for non-polynomial
    data (e.g. its bdryerr term).  This is the norm NSV03's "pointwise a
    posteriori error control" theory targets, in contrast to BR78/BV00's
    energy (H^1 seminorm) norm."""
    W = FunctionSpace(uh.function_space().mesh(), "CG", 4)
    err = Function(W).interpolate(abs(u - uh))
    return amr.scalarrange(err)[1]


print(f"solving the {args.prob} problem ({d}D case) ...")
results = {}
for method in methods:
    print("")
    print(f"using AMR by {method} method ...")
    mesh = mesh0
    dofs, errsl2, errsinf, errsH1ia, ests, eff_vals = [], [], [], [], [], []
    rhs, hausds, hausplains = [], [], []  # interface quantities; 2D easy only
    for j in range(args.maxlevels):
        V = FunctionSpace(mesh, "CG", 1)
        amr = VIAMR(debug=True)

        # UFL expressions for the data, and the discrete obstacle chi_h
        f_ufl, g_ufl, lb, u_ufl = data(mesh, V)

        # Initialize by cross-mesh interpolation, i.e. do mesh sequencing, then
        # lift onto the obstacle so the initial iterate is admissible.  For the
        # pyramid the corners of the diamond are exactly on the source mesh
        # boundary, so point location can miss them; allow_missing_dofs leaves
        # those dofs at zero, which is the boundary value there anyway.
        uh = Function(V, name="u_h (solution)")
        if j == 0:
            uh.interpolate(Constant(0.0))
        else:
            uh.interpolate(uh_prev, allow_missing_dofs=True)
        uh.interpolate(max_value(uh, lb))

        # state the problem
        vh = TestFunction(V)
        F = inner(grad(uh), grad(vh)) * dx - f_ufl * vh * dx
        g = Function(V).interpolate(g_ufl)  # = I_h g in NSV03 and NSV05
        bcs = DirichletBC(V, g, "on_boundary")
        problem = NonlinearVariationalProblem(F, uh, bcs)
        INFupper = Function(V).interpolate(Constant(PETSc.INFINITY))

        # solve the problem
        solver = NonlinearVariationalSolver(
            problem, solver_parameters=sp, options_prefix="s"
        )
        solver.solve(bounds=(lb, INFupper))
        uh_prev = uh

        _, nelements, _, _ = amr.meshsizes(mesh)
        dofs.append(V.dim())
        print(f"  level {j}: nodes = {dofs[-1]}, elements = {nelements}")

        # compute error norms, disregarding relevance to method
        if exact_known:
            errsl2.append(float(errornorm(u_ufl, uh)))  # default norm is L^2
            errsinf.append(float(errornorm_Linf(amr, u_ufl, uh)))
            iamark = amr.eleminactive(uh, lb, strong=True)
            dus = inner(grad(u_ufl - uh), grad(u_ufl - uh))
            errsH1ia.append(assemble(dus * iamark * dx(degree=6)) ** 0.5)
            print(f"    |u-u_h|_2 = {errsl2[-1]:.3e}, |u-u_h|_inf = {errsinf[-1]:.3e}")

        # marking, and the estimator each method targets: the l^2 accumulation of
        # BR78's inactive-set indicator, Etilde_h of (7.1) in NSV03, and E_h of
        # Theorem 2.7 in NSV05
        extra = ""
        if method == "UDOBR":
            fmark = amr.udomark(uh, lb, n=nUDO)
            residual = -div(grad(uh)) - f_ufl
            (imark, _, Eh) = amr.brinactivemark(
                uh, lb, residual, theta=marktheta, method=markmethod
            )
            mark = amr.unionmarks(fmark, imark)
            # BR78 targets the H^1 seminorm on the inactive set
            errtarget = errsH1ia[-1] if exact_known else 0.0
            estname = "eta_BR (inactive-set, energy norm)"
        elif method == "NSV03":
            (mark, etainf, etad, sigmah, Eh) = amr.nsv03mark(
                uh, lb, g, f_ufl, g_ufl, theta=marktheta, dualtol=dualtol,
                method=markmethod
            )
            # NSV03 targets ||u-u_h||_infty
            errtarget = errsinf[-1] if exact_known else 0.0
            estname = "Etilde_h (NSV03, sup norm)"
        elif method == "NSV05":
            (mark, eta, sz, fullcontact, Eh) = amr.nsv05mark(
                uh, lb, g, f_ufl, g_ufl, theta=marktheta, dualtol=dualtol,
                method=markmethod
            )
            # NSV05 targets ||u-u_h||_infty
            errtarget = errsinf[-1] if exact_known else 0.0
            estname = "E_h (NSV05, sup norm)"
            nfull = amr.countmark(fullcontact)
            extra = f", full-contact elements = {nfull} ({100 * nfull / nelements:.1f}%)"
        else:
            raise NotImplementedError
        ests.append(Eh)
        eff = Eh / errtarget if errtarget > 0 else float("nan")
        eff_vals.append(eff)
        effstr = f", effectivity = {eff:.3f}" if exact_known else ""
        print(f"    {estname} = {Eh:.3e}, marked = {amr.countmark(mark)}{extra}{effstr}")

        # Section 4 of NSV05 locates the exact free boundary F a posteriori.  By
        # Theorem 4.1, F lies within distance r_h of {u_h > chi_h + E_h}, which is
        # inside the computed inactive set.  Equation (4.4) in NSV05 gives
        #     r_h^2 = (2 d / lambda) (2 E_h + ||(chi_h - chi)^+||_inf).
        # Here chi_h = chi = 0, so that term drops.
        #
        # Theorem 4.1 uses the estimator only through reliability, ||u - u_h||_{0,inf} <= E_h,
        # so it applies to Etilde_h of (7.1) in NSV03, along with E_h of NSV05.
        #
        # The "condition number" (NSV05 name) lambda of the stability condition (4.1) is exactly
        # computable here.  With chi = 0, Remark 4.5 reads lambda = -sup f over
        # {u_h <= chi_h + E_h}, and f = -8 r^2 (1 - c) with c = |x|^2 - r^2 is
        # largest at c = 0; f only decreases outside the contact set, so
        # enlarging the set does not change the value, and lambda = 8 r^2.
        #
        # The comparison needs the boundary of {u_h > chi_h + E_h}.  Passing
        # lb + Eh straight to freeboundarygraph2D() does not give it: that routine
        # locates the *coincidence* set {|u_h - bound| < activetol} and asserts
        # u_h >= bound, so it trips the assertion, and returns an empty edge set
        # with debug off.  Clipping first fixes both, since
        #     {u_h > chi_h + E_h} = {v > chi_h + E_h}
        # when v = max(u_h, chi_h + E_h).  Note that v coincides with chi_h + E_h
        # over a whole region rather than nowhere.
        #
        # Every method also gets the plain distance from F to the computed free
        # boundary Gamma_h = boundary of {u_h = chi_h}, with no E_h offset, which
        # is the sphere.py comparison.  UDOBR has no sup-norm estimator and hence
        # neither r_h nor F_h, so this is the only interface quantity it has, and
        # it is the one which can be compared across all three methods.  Note
        # dist(F,Gamma_h) is several times smaller than dist(F,F_h), since
        # Gamma_h is inside F_h by E_h; the two are not interchangeable.
        if exact_known and d == 2:
            uexact = Function(V, name="u_exact").interpolate(u_ufl)
            _, fbexact = amr.freeboundarygraph2D(uexact, lb)
            _, fbplain = amr.freeboundarygraph2D(uh, lb)
            # hausdorff2D() returns None for an empty free boundary, e.g. on a
            # coarse initial mesh; format accordingly
            hplain = amr.hausdorff2D(fbexact, fbplain)
            # None means an empty free boundary, and an exact 0.0 means the two
            # nodal free boundaries coincide, as they do on the initial mesh;
            # neither goes on a log axis, so both become a gap in the figure
            hausplains.append(hplain if hplain else np.nan)
            line = f"    hausdorff2D(F, Gamma_h) = " + (
                f"{hplain:.3e}" if hplain is not None else "n/a"
            )
            if method in ("NSV03", "NSV05"):
                lam = 8.0 * r ** 2
                rh = (2.0 * d * 2.0 * Eh / lam) ** 0.5
                lbE = Function(V).interpolate(lb + Constant(Eh))
                uclip = Function(V).interpolate(max_value(uh, lbE))
                _, fbbarrier = amr.freeboundarygraph2D(uclip, lbE)
                haus = amr.hausdorff2D(fbexact, fbbarrier)
                hausstr = f"{haus:.3e}" if haus is not None else "n/a"
                line += (f", hausdorff2D(F, F_h) = {hausstr}, r_h = {rh:.3e}")
                rhs.append(rh)
                hausds.append(haus if haus else np.nan)
            print(line)

        # done with this method if we reach target complexity, or run out of levels
        if dofs[-1] > args.targetnodes or j == args.maxlevels - 1:
            break

        # get next mesh by refinement
        if d == 2:
            mesh = amr.refinesbr2D(mesh, mark)  # PETSc DM refinement
        else:
            mesh = mesh.refine_marked_elements(mark)  # Netgen refinement

    # for figures below
    results[method] = (dofs, errsl2, errsinf, errsH1ia, ests, eff_vals, rhs, hausds,
                       hausplains)

    # compute fields on final mesh (independent of method)
    gap = Function(V, name="gap = u_h - chi_h").interpolate(uh - lb)
    active = amr.elemactive(uh, lb)
    active.rename("active")
    tactive = amr.thinelemactive(uh, lb)
    tactive.rename("thin active")
    if mesh.comm.size > 1:
        rank = Function(FunctionSpace(mesh, "DG", 0))
        rank.dat.data[:] = mesh.comm.rank
        rank.rename("rank")

    # write Paraview-readable output
    outfile = f"result_{args.prob}_{method}.pvd"
    print(f"generating output file {outfile} ...")
    fields = [uh, lb, gap, active, tactive, mark]
    if exact_known:
        fields.append(
            Function(V, name="u_err = u_h - u_exact").interpolate(uh - u_ufl)
        )
    if method == "UDOBR":
        imark.rename("UDO inactive mark")
        fmark.rename("UDO FB mark")
        fields += [imark, fmark]
    elif method == "NSV03":
        fields += [sigmah, etainf, etad]
    elif method == "NSV05":
        fields += [eta, sz, fullcontact]
    if mesh.comm.size > 1:
        fields.append(rank)
    VTKFile(outfile).write(*fields)

# estimator figure; this is all the pyramid problem can plot
if not exact_known and mesh.comm.rank == 0:
    import matplotlib.pyplot as plt

    print("")

    # Without an exact solution only the estimators themselves can be plotted,
    # and they are three *different* quantities: an l^2 accumulation of the BR78
    # indicator for UDOBR, Etilde_h of (7.1) in NSV03, and E_h of Theorem 2.7 in
    # NSV05.  Their absolute values are not comparable; in particular nsv05mark()
    # keeps the |log h_min|^2 factor, which nsv03mark() has no analogue of, and
    # that inflates E_h by an order of magnitude.  Each curve is therefore
    # normalized by its own level-0 value, so what the figure compares is decay
    # rates.  Even so the curves target different norms, and so different optimal
    # rates, which is why both reference slopes are drawn: each estimator should
    # be read against the one for its own norm.  The NSV05 claim that full
    # localization reaches a given accuracy with far fewer DOFs is about the
    # meshes, so see result_pyramid_*.pvd for it.
    stylemap = {"UDOBR": "ko", "NSV03": "bs", "NSV05": "md"}
    figfile = f"nsv_{args.prob}_estimator.png"
    print(f"generating estimator figure {figfile} ...")
    plt.figure()
    alldofs = []
    for meth in methods:
        dofs, ests = np.array(results[meth][0]), np.array(results[meth][4])
        # an estimator can vanish on the coarsest meshes, as the BR78 one does
        # here while no element is yet strictly inactive; such levels cannot go
        # on log axes, and cannot set the normalization, so drop them
        keep = ests > 0.0
        dofs, ests = dofs[keep], ests[keep]
        alldofs.append(dofs)
        plt.loglog(dofs, ests / ests[0], stylemap[meth], label=meth + " estimator")
    alldofs = np.concatenate(alldofs)
    xx = np.array([alldofs.min(), alldofs.max()])
    for power, style, norm in [
        (-1.0 / d, "k--", "energy norm, i.e. UDOBR"),
        (-2.0 / d, "k:", "sup norm, i.e. NSV03 and NSV05"),
    ]:
        # normalized like the curves, so the reference slopes also start at 1
        plt.loglog(xx, (xx / xx[0]) ** power, style,
                   label=f"DOFs^({power * d:.0f}/d) for d={d} ({norm})")
    plt.legend()
    plt.grid(True)
    plt.xlabel("DOFs")
    plt.ylabel("estimator, relative to level 0")
    plt.title("estimators")
    plt.savefig(figfile)

# convergence and effectivity figures; only the easy problem has an exact solution
if exact_known and mesh.comm.rank == 0:
    import matplotlib.pyplot as plt

    print("")
    print(f"generating convergence and effectivity figures nsv_{args.prob}_*.png ...")
    markers = ["ko", "bs", "md"]

    figfile = f"nsv_{args.prob}_convergence_l2.png"
    plt.figure()
    for j in range(len(methods)):
        meth = methods[j]
        dofs, errs = np.array(results[meth][0]), np.array(results[meth][1])
        #print(np.polyfit(np.log(dofs), np.log(errs), 1))
        plt.loglog(dofs, errs, markers[j], label=meth)
        if j == len(methods) - 1:
            y = dofs ** (-2.0 / d)
            y = y * errs[0] / y[0]  # fix constant so that it aligns
            plt.loglog(dofs, y, "k:", label=f"DOFs^(-2/d) for d={d}")
    plt.legend()
    plt.grid(True)
    plt.xlabel("DOFs")
    plt.ylabel("norm error |u-u_h|_2")
    plt.title("L^2 errors")
    plt.savefig(figfile)

    figfile = f"nsv_{args.prob}_convergence_h1ia.png"
    plt.figure()
    for j in range(len(methods)):
        meth = methods[j]
        dofs, errs, ests = np.array(results[meth][0]), np.array(results[meth][3]), np.array(results[meth][4])
        #print(np.polyfit(np.log(dofs), np.log(errs), 1))
        plt.loglog(dofs, errs, markers[j], label=meth+' error')
        if meth == "UDOBR":
            plt.loglog(dofs, ests, markers[j], markerfacecolor='white', label=meth+' estimator')
        if j == len(methods) - 1:
            y = dofs ** (-1.0 / d)
            y = y * errs[0] / y[0]  # fix constant so that it aligns
            plt.loglog(dofs, y, "k:", label=f"DOFs^(-1/d) for d={d}")
    plt.legend()
    plt.grid(True)
    plt.xlabel("DOFs")
    plt.ylabel("|u-u_h|_H1 inactive")
    plt.title("H^1 errors restricted to inactive set")
    plt.savefig(figfile)

    figfile = f"nsv_{args.prob}_convergence_linf.png"
    plt.figure()
    for j in range(len(methods)):
        meth = methods[j]
        dofs, errs, ests = np.array(results[meth][0]), np.array(results[meth][2]), np.array(results[meth][4])
        #print(np.polyfit(np.log(dofs), np.log(errs), 1))
        plt.loglog(dofs, errs, markers[j], label=meth+' error')
        if meth in ("NSV03", "NSV05"):
            plt.loglog(dofs, ests, markers[j], markerfacecolor='white', label=meth+' estimator')
        if j == len(methods) - 1:
            y = dofs ** (-2.0 / d)
            y = y * errs[0] / y[0]  # fix constant so that it aligns
            plt.loglog(dofs, y, "k:", label=f"DOFs^(-2/d) for d={d}")
    plt.legend()
    plt.grid(True)
    plt.xlabel("DOFs")
    plt.ylabel("norm error |u-u_h|_inf")
    plt.title("L^inf errors (compare Figure 7.1 top in NSV03)")
    plt.savefig(figfile)

    # effectivity index = estimator / true error, in whichever norm the
    # estimator targets (energy norm for BR78, sup norm for NSV03); unlike
    # the norm-convergence figure above, there's no power-law rate to plot
    # against, since a reliable+efficient estimator should stay O(1) (ideally
    # near 1) as DOFs grow, so this uses a linear y-axis with a semilogx
    # x-axis, and a horizontal reference line at the ideal value of 1
    efffile = f"nsv_{args.prob}_effectivity.png"
    effstylemap = {"UDOBR": ("ko", "BR78 (inactive-set, energy norm)"),
                   "NSV03": ("bs", "NSV03 (sup norm)"),
                   "NSV05": ("md", "NSV05 (sup norm)")}
    plt.figure()
    for meth in methods:
        edofs, evals = np.array(results[meth][0]), np.array(results[meth][5])
        style, label = effstylemap[meth]
        plt.semilogx(edofs, evals, style, label=label)
    plt.axhline(1.0, color="k", linestyle=":", label="ideal eff=1")
    plt.legend()
    plt.grid(True)
    plt.xlabel("DOFs")
    plt.ylabel("effectivity ratio (estimator / true-error)")
    plt.title("effectivity index vs DOFs")
    plt.savefig(efffile)

    # Interface figure, in the shape of Figure 5.1 of NSV05: the strip width r_h
    # of (4.4), and the distance it is meant to control, namely the one between
    # the exact free boundary F and the computed barrier boundary
    # F_h = boundary of {u_h > chi_h + E_h}.  Only the two sup-norm estimators
    # have an r_h, and only 2D has hausdorff2D(); see the loop above.
    #
    # The plain distance dist(F,Gamma_h), to the computed free boundary itself,
    # is shown for all three methods, since it is the one interface quantity they
    # all have and so the only one which compares them.  Keep the two distances
    # apart when reading this: Gamma_h sits inside F_h by E_h, so dist(F,Gamma_h)
    # runs several times below dist(F,F_h), for reasons of definition rather than
    # of method quality.
    if d == 2:
        figfile = f"nsv_{args.prob}_interface.png"
        print(f"generating interface figure {figfile} ...")
        intstylemap = {"UDOBR": "ko", "NSV03": "bs", "NSV05": "md"}
        plt.figure()
        for meth in methods:
            idofs = np.array(results[meth][0])
            iplain = np.array(results[meth][8])
            plt.loglog(idofs, iplain, intstylemap[meth],
                       label=meth + " dist(F,Gamma_h)")
            if meth in ("NSV03", "NSV05"):
                irh = np.array(results[meth][6])
                ihaus = np.array(results[meth][7])
                plt.loglog(idofs, ihaus, intstylemap[meth], markerfacecolor="white",
                           label=meth + " dist(F,F_h)")
                plt.loglog(idofs, irh, intstylemap[meth][0] + "--",
                           label=meth + " r_h")
        # r_h ~ sqrt(E_h), so the interface converges at half the order of the sup
        # norm; compare the dotted line of slope -1/2 in Figure 5.1 of NSV05.
        # Unlike the other figures this one aligns the reference line at the
        # *finest* level, because on the coarsest meshes E_h is O(1), r_h then
        # exceeds the diameter of the contact set, and the barrier sets carry no
        # information; anchoring there would tilt the line for no good reason.
        ok = np.isfinite(ihaus)
        y = idofs ** (-1.0 / d)
        y = y * ihaus[ok][-1] / y[ok][-1]  # fix constant so that it aligns
        plt.loglog(idofs, y, "k:", label=f"DOFs^(-1/d) for d={d}")
        plt.legend()
        plt.grid(True)
        plt.xlabel("DOFs")
        plt.ylabel("distance")
        plt.title("free-boundary location: estimate r_h vs Hausdorff dist()")
        plt.savefig(figfile)
