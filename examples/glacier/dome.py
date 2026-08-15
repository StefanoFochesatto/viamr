# Solves the 2D steady, isothermal shallow ice approximation "dome" obstacle
# problem: flat bed (b=0), radially-symmetric surface mass balance, with a
# known exact solution (see synthetic.py).
#
# Compares the two primal-variable formulations of the physics, either in
# u = transformed thickness or s = surface elevation.
#
# The AMR strategy is UDO+BV00, so in the inactive set we use a weighted
# norm and estimator.
#
# This script has no command-line interface; edit the "major parameters" below.
#
# See steady.py for the general shallow ice approximation driver, with either
# a synthetic bumpy bed or a bed read from data.
#
# Generates:
#   result_dome_{u|s}.pvd          Paraview output, one per formulation
#   dome_convergence_H1.png        whole-domain |u-u_exact|_H1 vs DOFs
#   dome_convergence_L2.png        ||H-H_exact||_2 vs DOFs
#   dome_radius_error.png          free-boundary radius error vs DOFs
#   dome_effectivity.png           UDO+BV00 effectivity index vs DOFs

# major parameters
m0 = 10  # initial mesh is m0 x m0
levels = 6  # number of AMR refinement levels
theta = 0.5  # BR/UDO fixed-rate threshold in the inactive set
udo_n = 1  # udomark() dilation levels
primals = ["u", "s"]  # compare both primal-variable formulations

import numpy as np
from firedrake import *
from firedrake.petsc import PETSc

print = PETSc.Sys.Print  # enables correct printing in parallel

from viamr import VIAMR
from physics import (
    secpera,
    H2u,
    u2H,
    weakform_u,
    weakform_s,
    residual_u_ufl,
    residual_s_ufl,
    glaciermeshreport,
    solve_params,
)
from synthetic import L, dome_a_ufl, dome_s_ufl, dome_normerrors, dome_radiuserror

results = {}  # results[primal] = (dofs, uerrH1s, HerrL2s, drmaxes)
effs = {}  # effs[primal] = (eff_dofs, eff_vals); see effectivity note below

for primal in primals:
    print(f"solving dome problem using primal variable {primal} and UDO+BR weighted ...")
    mesh = RectangleMesh(
        m0,
        m0,
        L,
        L,
        diagonal="crossed",
        distribution_parameters=VIAMR.PARALLEL_OVERLAP,
    )
    amr = VIAMR(debug=True, activetol=1.0)
    dofs, uerrH1s, HerrL2s, drmaxes, eff_vals = [], [], [], [], []

    for i in range(levels + 1):
        print(f"mesh {i}:")
        amr.meshreport(mesh, indent=2)
        V, DG0 = amr.spaces(mesh)  # V=CG1
        x = SpatialCoordinate(mesh)
        a = Function(V, name="a = accumulation").interpolate(dome_a_ufl(x))

        # initial iterate: pile of ice from accumulation, or cross-mesh
        # interpolation of the previous level's solution
        # note w=u=H^{1/omega} or w=s=H according to primal
        if i == 0:
            pileage = 400.0  # years
            Hinit = pileage * secpera * conditional(a > 0.0, a, 0.0)
            if primal == "u":
                w = Function(V).interpolate(H2u(Hinit))
            else:
                w = Function(V).interpolate(Hinit)
        else:
            w = Function(V).interpolate(wold)  # cross-mesh interpolation
            # w = Function(V).interpolate(conditional(w < 1.0, 0.0, w))
        w.rename(
            "u = transformed thickness" if primal == "u" else "s = surface elevation"
        )
        bcs = [DirichletBC(V, Constant(0.0), "on_boundary")]
        lb = Function(V).interpolate(Constant(0.0))
        ub = Function(V).interpolate(Constant(PETSc.INFINITY))

        # solve directly by Newton; no Picard freeze-tilt loop is needed
        # because grad(b)=0 makes the tilt term Phi_u(u,b) identically zero
        F = weakform_u(w, a, lb) if primal == "u" else weakform_s(w, a, lb)
        problem = NonlinearVariationalProblem(F, w, bcs=bcs)
        solver = NonlinearVariationalSolver(
            problem, solver_parameters=solve_params, options_prefix="dome"
        )
        solver.solve(bounds=(lb, ub))

        # numerical errors vs exact solution
        # note dome_normerrors() takes both u and H
        if primal == "u":
            Hlike = Function(V, name="H (thickness)").interpolate(u2H(w))
            uerrH1semi, uerrH1rel, HerrLinf = dome_normerrors(w, Hlike)
            HerrL2 = assemble((Hlike - dome_s_ufl(x)) ** 2 * dx(degree=6)) ** 0.5
        else:
            ulike = Function(V, name="u (transformed thickness)").interpolate(H2u(w))
            uerrH1semi, uerrH1rel, HerrLinf = dome_normerrors(ulike, w)
            HerrL2 = assemble((w - dome_s_ufl(x)) ** 2 * dx(degree=6)) ** 0.5
        vfb, _ = amr.freeboundarygraph2D(w, lb)
        drmax = dome_radiuserror(mesh, vfb)
        print(
            f"  |u-uexact|_H1rel = {uerrH1rel:.3e};  |H-Hexact|_Linf = {HerrLinf:.3f} m;  |dr|_Linf = {drmax/1000.0:.3f} km"
        )

        dofs.append(V.dim())
        uerrH1s.append(float(uerrH1semi))
        HerrL2s.append(float(HerrL2))
        drmaxes.append(drmax)

        # mark and refine by UDO+BV00
        if primal == "u":
            res, Z = residual_u_ufl(w, a, lb)
        else:
            res, Z = residual_s_ufl(w, a, lb)
        fbmark = amr.udomark(w, Constant(0.0), n=udo_n)  # because of flat bed, psi=0 regardless of -primal
        imark, _, total_eta = amr.brinactivemark(
            w, Constant(0.0), res, theta=theta, method="total", alpha=Z
        )
        mark = amr.unionmarks(fbmark, imark)

        # report on AMR
        ne, hmin = glaciermeshreport(amr, mesh)
        ei = amr.eleminactive(w, Constant(0.0))
        pfb = 100.0 * amr.countmark(fbmark) / ne
        pin = 100.0 * amr.countmark(imark) / amr.countmark(ei)
        print(
            f"  elements marked: {pfb:.2f}% free-bdry, {pin:.2f}% of inactive"
        )

        # effectivity index = estimator / (true error)
        # Note brinactivemark(alpha=Z) restricts to the inactive set and,
        # per Bernardi & Verfurth (2000), targets the weighted energy norm.
        # Here Z is the (regularized) diffusivity of the equation we actually
        # solved (residual_u_ufl vs residual_s_ufl).  While eff_u
        # and eff_s are each computed correctly for their own formulation,
        # they are NOT directly comparable to each other in absolute terms.
        alli = amr.eleminactive(w, lb, strong=True)
        wexact = H2u(dome_s_ufl(x)) if primal == "u" else dome_s_ufl(x)
        dws = inner(grad(w - wexact), grad(w - wexact))
        errqn = assemble(Z * dws * alli * dx(degree=6)) ** 0.5
        eff = float(total_eta / errqn) if errqn > 0 else np.nan
        print(f"  eff_{primal} = {eff:.3f}  (using {primal} energy norm on inactive set)")
        eff_vals.append(eff)

        wold = w
        if i < levels:
            mesh = amr.refinesbr2D(mesh, mark)

    results[primal] = (dofs, uerrH1s, HerrL2s, drmaxes)
    effs[primal] = (dofs, eff_vals)

    # compute u error as a field
    uexact = H2u(dome_s_ufl(x))  # UFL
    if primal == "u":
        uerr = Function(w.function_space()).interpolate(w - uexact)
    else:
        uerr = Function(ulike.function_space()).interpolate(ulike - uexact)
    uerr.rename("uerr = u-u_exact")

    # Paraview output for this formulation's final mesh
    outfile = f"result_dome_{primal}.pvd"
    print(f"done with {primal}-form ... writing to {outfile} ...")
    imark.rename("imark (BV00)")
    fbmark.rename("free-boundary mark")
    mark.rename("mark")
    fields = [w, a, uerr, mark, fbmark, imark]
    fields.append(Hlike if primal == "u" else ulike)
    VTKFile(outfile).write(*fields)
    print("")

# convergence and effectivity figures, comparing u-form vs s-form
if mesh.comm.rank == 0:
    import matplotlib.pyplot as plt

    d = 2  # spatial dimension
    stylemap = {"u": "ko", "s": "bs"}

    def _convergence_plot(index, ylabel, title, outfile, rateexp, ratelabel):
        plt.figure()
        for primal in primals:
            rdofs, rvals = np.array(results[primal][0]), np.array(
                results[primal][index]
            )
            plt.loglog(rdofs, rvals, stylemap[primal], label=f"{primal}-form")
        rdofs, rvals = np.array(results[primals[0]][0]), np.array(
            results[primals[0]][index]
        )
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

    # H^1 seminorm: a priori theory for obstacle problem gives first-order
    # convergence, DOFs^(-1/d); BR78-type estimators target this norm
    _convergence_plot(
        1,
        "|u-u_exact|_{H^1}",
        "dome problem: H^1 seminorm error of u vs DOFs",
        "dome_convergence_H1.png",
        -1.0 / d,
        "DOFs^(-1/d)",
    )
    _convergence_plot(
        2,
        "||H-H_exact||_2",
        "dome problem: thickness L^2 error vs DOFs",
        "dome_convergence_L2.png",
        -2.0 / d,
        "DOFs^(-2/d)",
    )
    # free-boundary radius error is a geometric (not squared-error) quantity,
    # so the natural reference is mesh-resolution scaling h ~ DOFs^(-1/d)
    _convergence_plot(
        3,
        "max free-boundary radius error (m)",
        "dome problem: free-boundary radius error vs DOFs",
        "dome_radius_error.png",
        -1.0 / d,
        "DOFs^(-1/d)",
    )

    # effectivity index = estimator / true error, in whichever norm the
    # estimator targets for that formulation (see comment above); a
    # reliable and efficient estimator should stay O(1) as DOFs grow
    plt.figure()
    for primal in primals:
        edofs, evals = np.array(effs[primal][0]), np.array(effs[primal][1])
        plt.semilogx(
            edofs,
            evals,
            stylemap[primal],
            label=f"{primal}-form (own BV00 energy norm, inactive set)",
        )
    plt.axhline(1.0, color="k", linestyle=":", label="ideal eff=1")
    plt.legend()
    plt.grid(True)
    plt.xlabel("DOFs")
    plt.ylabel("effectivity index")
    plt.title("dome problem: UDO+BV00 effectivity index vs DOFs")
    plt.savefig("dome_effectivity.png")
