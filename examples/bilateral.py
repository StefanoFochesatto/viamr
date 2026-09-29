# Compare AMR methods on a *bilateral* classical obstacle problem, with
#
#     -1 <= u <= 1    everywhere
#     -Delta u = f    in inactive set
#
# On Omega = (-2,2)^2, the exact solution is u(r) for r = sqrt(x_1^2+x_2^2):
#
#     r <= 0.5        upper contact,  u = 1
#     0.5 <= r <= 1   inactive,       u = 1 - 6s^2 + 4s^3,          s = 2r - 1
#     1 <= r <= 1.5   lower contact,  u = -1
#     1.5 <= r <= 2   inactive,       u = -1 + 6t^2 - 8t^3 + 3t^4,  t = 2r - 3
#     r >= 2          inactive,       u = 0
#
# and the load f is continuous:
#
#     f = 48                                     r <= 0.5
#     f = 48 - 96 s + 24 s(1-s)/r                0.5 <= r <= 1     (48 -> -48)
#     f = -48                                    1 <= r <= 1.5
#     f = -48 + 192 t - 144 t^2 - 24 t(1-t)^2/r  1.5 <= r <= 2     (-48 -> 0)
#     f = 0                                      r >= 2
#
# The fixed Dirichlet boundary condition is g=0 on the boundary of the square,
# entirely in the (closure of the) inactive set.
#
# Note that the residual sigma = -Lap(u) - f satisfies sigma = -48 <= 0 on the
# upper-contact disc and sigma = +48 >= 0 on the lower-contact annulus, and
# sigma = 0 on the other three (inactive) regions.  Also u is C^1 across every
# free boundary, since u' = -24 s(1-s) and u' = 24 t(1-t)^2 both vanish at their
# endpoints, and thus no singular measure sits on a free boundary.  The quartic on
# the outer annulus makes u'(2) = u''(2) = 0, which is what lets f be continuous
# everywhere, including where u meets the u = 0 corner region.
#
# The free boundaries are circles, so no triangulation aligns with them, and also
# there exist transition elements where sigma_h changes sign.
#
# Both UDO+BR and NSV03/B26 methods are applied by default; see option -no_udo.
#
# Generates .pvd from final mesh for each method.  Generates .png figures showing
# convergence and effectivity as functions of dofs.  Generates .png
# showing solution and marking on final mesh.
#
# Runs:  python3 bilateral.py -h                       # help
#        python3 bilateral.py                          # good .pvd for paper
#        python3 bilateral.py -target 8.0e5 -no_udo    # performance figure
#        python3 bilateral.py -no_udo -target 1e9 -maxlevels N  # mesh marking figures N=4,7

from argparse import ArgumentParser

parser = ArgumentParser(
    description="Compare AMR by UDO+BR and NSV03/B26 on a bilateral obstacle problem."
)
parser.add_argument(
    "-maxlevels", type=int, default=15, metavar="N",
    help="stop on levels in mesh hierarchy from AMR [default=15]",
)
parser.add_argument(
    "-no_udo", action="store_true", default=False,
    help="do not include UDO+BR method (e.g. for paper figures)",
)
parser.add_argument(
    "-target", type=float, default=2.0e3, metavar="X",
    help="stop refining at this many nodes [default=2.0e3]",  # increase for performance study
)
parser.add_argument(
    "-theta", type=float, default=0.5, metavar="X",
    help="marking parameter [default=0.5]",
)
args, passthroughoptions = parser.parse_known_args()

import numpy as np
import matplotlib.pyplot as plt
import petsc4py

petsc4py.init(passthroughoptions)

from firedrake import *
from firedrake.petsc import PETSc

from viamr import VIAMR

print = PETSc.Sys.Print  # enables correct printing in parallel

methods = ["NSV03"] if args.no_udo else ["NSV03", "UDOBR"]

# parameters
nUDO = 0
dualtol = 1.0e-8
markmethod = "total"
m0 = 4  # initial mesh resolution

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
    """Approximate sup-norm error, the NSV03 target norm.  Same
    technique _nsv03mark() uses internally for non-polynomial data."""
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
        nlo = amr.countmark(amr.elemactive(uh, (lb, None)))
        nup = amr.countmark(amr.elemactive(uh, (None, ub)))
        print(f"  level {j}: nodes = {dofs[-1]}, elements = {nelements}, "
              f"lower-active = {nlo}, upper-active = {nup}")

        area = 16.0
        errsl2.append(float(errornorm(u_ufl, uh) / np.sqrt(area)))  # scaled
        errsinf.append(errornorm_Linf(amr, u_ufl, uh))
        assert errsl2[-1] <= errsinf[-1]  # because of l2 scaling
        # H^1 seminorm on the set inactive for *both* obstacles, which is what
        # inactivemark() restricts its estimator to
        iamark = amr.eleminactive(uh, (lb, ub), strong=True)
        dus = inner(grad(u_ufl - uh), grad(u_ufl - uh))
        errsH1ia.append(assemble(dus * iamark * dx(degree=6)) ** 0.5)
        print(f"    |u-u_h|_2 = {errsl2[-1]:.3e}, |u-u_h|_inf = {errsinf[-1]:.3e}")

        if method == "UDOBR":
            fmark, _, _ = amr.fbmark(uh, (lb, ub), udo_n=nUDO)
            residual = -div(grad(uh)) - f_ufl
            (imark, _, Eh) = amr.inactivemark(
                uh, (lb, ub), estimator="br78", res=residual, theta=args.theta, method=markmethod
            )
            mark = amr.unionmarks(fmark, imark)
            errtarget = errsH1ia[-1]
            estname = "eta_BR (inactive-set, energy norm)"
        elif method == "NSV03":
            (mark, nsvfields, Eh) = amr.nsvmark(
                uh, (lb, ub), g, f_ufl, g_ufl, estimator="nsv03", theta=args.theta,
                dualtol=dualtol, method=markmethod,
            )
            etainf, etad, sigmah = nsvfields["etainf"], nsvfields["etad"], nsvfields["sigmah"]
            #smin, smax = amr.scalarrange(sigmah)
            #print(f"    sigma_h range = [{smin:.2f}, {smax:.2f}] (exact: [-48, 48])")
            errtarget = errsinf[-1]
            estname = "Etilde_h (sup norm)"
        else:
            raise NotImplementedError
        ests.append(Eh)
        eff = Eh / errtarget if errtarget > 0 else float("nan")
        eff_vals.append(eff)
        print(f"    {estname} = {Eh:.3e}, marked = {amr.countmark(mark)}, "
              f"effectivity = {eff:.3f}")

        if dofs[-1] > args.target or j == args.maxlevels - 1:
            break
        mesh = amr.refinesbr2D(mesh, mark)

    results[method] = (dofs, errsl2, errsinf, errsH1ia, ests, eff_vals)

    # write to .pvd; skip writing lb,ub because they are constant
    uerr = Function(V, name="u_err = u_h - u_exact").interpolate(uh - u_ufl)
    gap_lo = Function(V, name="u_h - lb (lower gap)").interpolate(uh - lb)
    gap_up = Function(V, name="ub - u_h (upper gap)").interpolate(ub - uh)
    active_lo = amr.elemactive(uh, (lb, None))
    active_lo.rename("lower active")
    active_up = amr.elemactive(uh, (None, ub))
    active_up.rename("upper active")
    outfile = f"result_bilateral_{method}.pvd"
    print(f"generating output file {outfile} ...")
    fields = [uh, gap_lo, gap_up, active_lo, active_up, mark, uerr]
    if method == "NSV03":
        fields += [sigmah, etainf, etad]
    VTKFile(outfile).write(*fields)

if mesh.comm.rank == 0:
    print("")
    print("generating figures bilateral_*.png ...")

    markers = {"UDOBRl2": "ko",
               "UDOBRlinf": "ks",
               "NSV03l2": "bo",
               "NSV03linf": "bs"}
    plt.figure()

    def labelmaker(meth, basic):
        lab = basic
        if len(methods) > 1:
            lab = meth + " " + lab
        return lab

    for meth in methods:
        ddinf, eeinf = np.array(results[meth][0]), np.array(results[meth][2])
        plt.loglog(ddinf, eeinf, markers[meth+"linf"], label=labelmaker(meth, "|u-u_h|_inf"))
    for meth in methods:
        ddest, eeest = np.array(results[meth][0]), np.array(results[meth][4])
        lab = "|u-u_h|_inf" if len(methods) == 1 else meth + " |u-u_h|_inf"
        plt.loglog(ddest, eeest, markers[meth+"linf"], markerfacecolor="w", label=labelmaker(meth, "estimate"))
    for meth in methods:
        dd2, ee2 = np.array(results[meth][0]), np.array(results[meth][1])
        plt.loglog(dd2, ee2, markers[meth+"l2"], markerfacecolor="w", label=labelmaker(meth, "|u-u_h|_2"))
    y = ddinf ** -1.0  # use last linf value for normalization
    plt.loglog(ddinf, y * eeinf[0] / y[0], "k:", label="DOFs^(-1) = O(h^2)")
    plt.legend()
    plt.grid(True)
    plt.xlabel("DOFs")
    plt.ylabel("error norm")
    plt.savefig(f"bilateral_convergence.png")

    # Each method's estimator is compared to the norm that method targets:
    # energy norm on inactive set for BR78; sup norm for NSV03.
    markers = {"UDOBR": "ko",
               "NSV03": "bo"}
    for meth in methods:
        plt.figure()
        dd, ef = np.array(results[meth][0]), np.array(results[meth][5])
        plt.semilogx(dd, ef, markers[meth], label=meth)
        plt.semilogx([0.9*min(dd), 1.1*max(dd)], [1.0, 1.0], 'k:')
        plt.xlim(0.91*min(dd), 1.09*max(dd))
        plt.ylim(0.0, 1.1 * max(ef))
        plt.grid(True)
        #plt.axis('tight')
        plt.xlabel("DOFs")
        if meth == "NSV03":
            plt.ylabel("effectivity = estimator / |u-u_h|_inf")
        elif meth == "UDOBR":
            plt.ylabel("effectivity = estimator / |u-u_h|_H1[inactive]")
        plt.savefig(f"bilateral_effectivity_{meth}.png", bbox_inches='tight')


    if nelements > 1.0e5:
        print("[large mesh ... EXITing before trying to generate mesh figures ...]")
        import sys
        sys.exit(0)

    from firedrake.pyplot import tripcolor, triplot

    # triplot() issues:
    #   * pops "colors" from boundary_kw, so pass a copy of bkw, as "dict(bkw)"
    #   * logger warns spuriously about empty interior-facet subdomains, so drop warnings
    bkw = {"colors": 4 * ["k"],
           "linewidths": 1.0}
    set_log_level(ERROR)

    fig, axes = plt.subplots()
    tripcolor(uh, axes=axes, cmap='viridis')
    triplot(mesh, axes=axes, interior_kw={"linewidths": 0.4}, boundary_kw=dict(bkw))
    axes.set_aspect("equal")
    axes.set_axis_off()
    fig.savefig("bilateral_uh.png", bbox_inches='tight', dpi=300.0)
    plt.cla()

    fig, axes = plt.subplots()
    _, DG0 = amr.spaces(mesh)
    tripcolor(Function(DG0).interpolate(0.7 * mark), axes=axes, cmap='Greys', clim=(0.0,1.0))
    triplot(mesh, axes=axes, interior_kw={"linewidths": 0.3}, boundary_kw=dict(bkw))
    axes.set_aspect("equal")
    axes.set_axis_off()
    fig.savefig("bilateral_mark.png", bbox_inches='tight', dpi=300.0)
    plt.cla()
