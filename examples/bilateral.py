# Compare AMR methods on a *bilateral* classical obstacle problem, with
#
#     -1 <= u <= 1    everywhere
#     -Delta u = f    in inactive set
#
# Has an exact solution, with non-empty lower and upper active (contact) sets.
# We are not aware of such a verification case in the literature, thus
# this is an original construction.
#
# On Omega = (-2,2)^2, u=u(r) is radial in r = sqrt(x_1^2+x_2^2):
#
#     r <= 0.5        upper contact,  u = 1
#     0.5 <= r <= 1   free,           u = 1 - 6s^2 + 4s^3,          s = 2r - 1
#     1 <= r <= 1.5   lower contact,  u = -1
#     1.5 <= r <= 2   free,           u = -1 + 6t^2 - 8t^3 + 3t^4,  t = 2r - 3
#     r >= 2          free,           u = 0
#
# and the load is
#
#     f = 48                                     r <= 0.5
#     f = 48 - 96 s + 24 s(1-s)/r                0.5 <= r <= 1     (48 -> -48)
#     f = -48                                    1 <= r <= 1.5
#     f = -48 + 192 t - 144 t^2 - 24 t(1-t)^2/r  1.5 <= r <= 2     (-48 -> 0)
#     f = 0                                      r >= 2
#
# Note that the residual sigma = -Lap(u) - f satisfies sigma = -48 <= 0 on the
# upper-contact disc and sigma = +48 >= 0 on the lower-contact annulus, and
# sigma = 0 on the other three (inactive) regions.  Also u is C^1 across every
# free boundary, since u' = -24 s(1-s) and u' = 24 t(1-t)^2 both vanish at their
# endpoints, so no spurious measure sits on a free boundary.  The quartic on the
# outer annulus makes u'(2) = u''(2) = 0, which is what lets f be continuous
# everywhere, including where u meets the u = 0 corner region.  Every boundary
# point of the square has r >= 2, and g = 0.
#
# The free boundaries are circles, so no triangulation aligns with them, and the
# transition elements where sigma_h changes sign are genuinely present.  See
# doc/box.tex for discussion of the NSV03 method extension to box bounds as here.
#
# usage:
#   python3 bilateral.py                 both methods; generates .png figures
#   python3 bilateral.py -targetnodes 4e4

from argparse import ArgumentParser

parser = ArgumentParser(
    description="Compare AMR by UDO+BR and NSV03 on a bilateral obstacle problem, both methods in their box-constrained form."
)
parser.add_argument(
    "-maxlevels", type=int, default=10, metavar="N",
    help="backstop on AMR levels [default=10]",
)
parser.add_argument(
    "-targetnodes", type=float, default=1.0e4, metavar="X",
    help="stop refining once this many nodes is met [default=1.0e4]",
)
parser.add_argument(
    "-theta", type=float, default=0.5, metavar="X",
    help="marking parameter [default=0.5]",
)
args, passthroughoptions = parser.parse_known_args()

import numpy as np
import petsc4py

petsc4py.init(passthroughoptions)
from firedrake import *
from viamr import VIAMR
from firedrake.petsc import PETSc

print = PETSc.Sys.Print  # enables correct printing in parallel

methods = ["UDOBR", "NSV03"]

# parameters
nUDO = 0
dualtol = 1.0e-8
markmethod = "total"
m0 = 8  # initial mesh resolution

mesh0 = RectangleMesh(
    m0, m0, 2.0, 2.0, originX=-2.0, originY=-2.0, diagonal="crossed",
    distribution_parameters=VIAMR.PARALLEL_OVERLAP,
)


def data(mesh, V):
    """UFL expressions for the exact solution and the load, plus the discrete
    obstacles, on the current mesh.  See the header comment for the derivation."""
    x, y = SpatialCoordinate(mesh)
    r = sqrt(x ** 2 + y ** 2)
    rs = max_value(r, 0.25)  # guards the u'/r terms, only ever used where r >= 0.5
    s = 2 * r - 1
    t = 2 * r - 3
    u_ufl = conditional(
        r < 0.5, Constant(1.0),
        conditional(r < 1.0, 1 - 6 * s ** 2 + 4 * s ** 3,
        conditional(r < 1.5, Constant(-1.0),
        conditional(r < 2.0, -1 + 6 * t ** 2 - 8 * t ** 3 + 3 * t ** 4,
                    Constant(0.0)))))
    f_ufl = conditional(
        r < 0.5, Constant(48.0),
        conditional(r < 1.0, 48 - 96 * s + 24 * s * (1 - s) / rs,
        conditional(r < 1.5, Constant(-48.0),
        conditional(r < 2.0, -48 + 192 * t - 144 * t ** 2 - 24 * t * (1 - t) ** 2 / rs,
                    Constant(0.0)))))
    g_ufl = Constant(0.0)  # every boundary point has r >= 2
    lb = Function(V, name="lb (lower obstacle)").interpolate(Constant(-1.0))
    ub = Function(V, name="ub (upper obstacle)").interpolate(Constant(1.0))
    return f_ufl, g_ufl, lb, ub, u_ufl


sp = {
    "snes_type": "vinewtonrsls",
    "snes_converged_reason": None,
    "snes_atol": 1.0e-12,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def errornorm_Linf(amr, u, uh):
    """Approximate sup-norm error, the norm NSV03's theory targets; same
    technique nsv03mark() uses internally for non-polynomial data."""
    W = FunctionSpace(uh.function_space().mesh(), "CG", 4)
    return amr.scalarrange(Function(W).interpolate(abs(u - uh)))[1]


print("solving the bilateral problem ...")
results = {}
for method in methods:
    print("")
    print(f"using AMR by {method} method ...")
    mesh = mesh0
    dofs, errsl2, errsinf, errsH1ia, ests, eff_vals = [], [], [], [], [], []
    for j in range(args.maxlevels):
        V = FunctionSpace(mesh, "CG", 1)
        amr = VIAMR(debug=True)
        f_ufl, g_ufl, lb, ub, u_ufl = data(mesh, V)

        # mesh sequencing, then clip into the box so the initial iterate is admissible
        uh = Function(V, name="u_h (solution)")
        if j == 0:
            uh.interpolate(Constant(0.0))
        else:
            uh.interpolate(uh_prev, allow_missing_dofs=True)
        uh.interpolate(min_value(max_value(uh, lb), ub))

        vh = TestFunction(V)
        F = inner(grad(uh), grad(vh)) * dx - f_ufl * vh * dx
        g = Function(V).interpolate(g_ufl)
        bcs = DirichletBC(V, g, "on_boundary")
        problem = NonlinearVariationalProblem(F, uh, bcs)
        solver = NonlinearVariationalSolver(problem, solver_parameters=sp, options_prefix="s")
        solver.solve(bounds=(lb, ub))
        uh_prev = uh

        _, nelements, _, _ = amr.meshsizes(mesh)
        dofs.append(V.dim())
        nlo = amr.countmark(amr.elemactive(uh, lb, boxside="lower"))
        nup = amr.countmark(amr.elemactive(uh, ub, boxside="upper"))
        print(f"  level {j}: nodes = {dofs[-1]}, elements = {nelements}, "
              f"lower-active = {nlo}, upper-active = {nup}")

        errsl2.append(float(errornorm(u_ufl, uh)))
        errsinf.append(errornorm_Linf(amr, u_ufl, uh))
        # H^1 seminorm on the set inactive for *both* obstacles, which is what
        # brinactivemark() restricts its estimator to
        iamark = Function(amr.spaces(mesh)[1]).interpolate(
            amr.eleminactive(uh, lb, boxside="lower", strong=True)
            * amr.eleminactive(uh, ub, boxside="upper", strong=True)
        )
        dus = inner(grad(u_ufl - uh), grad(u_ufl - uh))
        errsH1ia.append(assemble(dus * iamark * dx(degree=6)) ** 0.5)
        print(f"    |u-u_h|_2 = {errsl2[-1]:.3e}, |u-u_h|_inf = {errsinf[-1]:.3e}")

        if method == "UDOBR":
            # box form: the free-boundary marking is the union of the two
            # one-sided UDO marks, as in examples/pollutant.py, while the
            # inactive-set estimator takes the pair, so that it restricts to the
            # elements touching *neither* obstacle
            fmark = amr.unionmarks(
                amr.udomark(uh, lb, boxside="lower", n=nUDO),
                amr.udomark(uh, ub, boxside="upper", n=nUDO),
            )
            residual = -div(grad(uh)) - f_ufl
            (imark, _, Eh) = amr.brinactivemark(
                uh, (lb, ub), residual, theta=args.theta, method=markmethod
            )
            mark = amr.unionmarks(fmark, imark)
            errtarget = errsH1ia[-1]
            estname = "eta_BR (inactive-set, energy norm)"
        elif method == "NSV03":
            (mark, etainf, etad, sigmah, Eh) = amr.nsv03mark(
                uh, (lb, ub), g, f_ufl, g_ufl, theta=args.theta,
                dualtol=dualtol, method=markmethod,
            )
            smin, smax = amr.scalarrange(sigmah)
            print(f"    sigma_h range = [{smin:.2f}, {smax:.2f}] (exact: [-48, 48])")
            errtarget = errsinf[-1]
            estname = "Etilde_h (NSV03 box, sup norm)"
        else:
            raise NotImplementedError
        ests.append(Eh)
        eff = Eh / errtarget if errtarget > 0 else float("nan")
        eff_vals.append(eff)
        print(f"    {estname} = {Eh:.3e}, marked = {amr.countmark(mark)}, "
              f"effectivity = {eff:.3f}")

        if dofs[-1] > args.targetnodes or j == args.maxlevels - 1:
            break
        mesh = amr.refinesbr2D(mesh, mark)

    results[method] = (dofs, errsl2, errsinf, errsH1ia, ests, eff_vals)

    # write to .pvd; skip writing lb,ub because they are constant
    uerr = Function(V, name="u_err = u_h - u_exact").interpolate(uh - u_ufl)
    gap_lo = Function(V, name="u_h - lb (lower gap)").interpolate(uh - lb)
    gap_up = Function(V, name="ub - u_h (upper gap)").interpolate(ub - uh)
    active_lo = amr.elemactive(uh, lb, boxside="lower")
    active_lo.rename("lower active")
    active_up = amr.elemactive(uh, ub, boxside="upper")
    active_up.rename("upper active")
    outfile = f"result_bilateral_{method}.pvd"
    print(f"generating output file {outfile} ...")
    fields = [uh, gap_lo, gap_up, active_lo, active_up, mark, uerr]
    if method == "NSV03":
        fields += [sigmah, etainf, etad]
    VTKFile(outfile).write(*fields)

if mesh.comm.rank == 0:
    import matplotlib.pyplot as plt

    print("")
    print("generating figures bilateral_*.png ...")
    markers = {"UDOBR": "ko", "NSV03": "bs"}

    for tag, idx, ylab, power in [
        ("l2", 1, "norm error |u-u_h|_2", -1.0),
        ("linf", 2, "sup norm error |u-u_h|_inf", -1.0),
    ]:
        plt.figure()
        for meth in methods:
            dd, ee = np.array(results[meth][0]), np.array(results[meth][idx])
            plt.loglog(dd, ee, markers[meth], label=meth)
        y = dd ** power
        plt.loglog(dd, y * ee[0] / y[0], "k:", label="DOFs^(-1) = O(h^2)")
        plt.legend()
        plt.grid(True)
        plt.xlabel("DOFs")
        plt.ylabel(ylab)
        plt.title(f"bilateral problem: {ylab}")
        plt.savefig(f"bilateral_convergence_{tag}.png")

    # Each method's estimator is compared to the norm that method targets: the
    # energy norm on the doubly-inactive set for BR78, and the sup norm for
    # NSV03.  The two effectivities are therefore not comparable to each other,
    # only each to the ideal value 1.
    plt.figure()
    for meth in methods:
        dd, ef = np.array(results[meth][0]), np.array(results[meth][5])
        plt.semilogx(dd, ef, markers[meth], label=meth)
    plt.legend()
    plt.grid(True)
    plt.xlabel("DOFs")
    plt.ylabel("effectivity = estimator / error in targeted norm")
    plt.title("bilateral problem: effectivity")
    plt.savefig("bilateral_effectivity.png")
