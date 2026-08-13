# Physical formulas for glaciers, and associated constants.
# Also some standardized solver parameters.

import numpy as np
import firedrake as fd
from firedrake.petsc import PETSc

pprint = PETSc.Sys.Print  # parallel print

debug = False  # debugging solver failures

# physical constants
secpera = 31556926.0  # seconds per year
n = 3.0  # exponent in Glen-Nye flow law
g = 9.81  # acceleration of gravity on earth
rho = 910.0  # density of ice
A = 1.0e-16 / secpera  # ice softness in flow law

# derived constants
p = n + 1  # typical:  p = 4
Gamma = 2 * A * (rho * g) ** n / (n + 2)  # coefficient in shallow ice approximation

# private constants of transformed SIA
_omega = (p - 1) / (2 * p)  #  \omega = 3/8
_mu = (p + 1) / (2 * p)  #  \mu = 5/8
_r = p / (p - 1)  #  r = 4/3


def u2H(u):
    return u ** _omega


def H2u(H):
    return H ** (1.0 / _omega)


def Phi_u(u, b):
    """Computes the tilting effect of the bed gradient:
      \Phi(u) = (1/\omega) u^\mu \grad b.
    In fact we regularize the power to avoid NaN or Inf from the fractional power:
      u^\mu --> (u + eps)^\mu."""
    eps = 1.0  # eps=1 regularization is small; typical u is 1e3 to 1e9
    return (1.0 / _omega) * (u + eps) ** _mu * fd.grad(b)


def weakform_u(u, a, b, Z=None, epsreg=0.01, qdegree=5):
    """Return the nonlinear weak form for the u (transformed thickness) form
    of the shallow ice approximation.

    When Z=None this is the weak form corresponding to (3) in METHOD.md.
    If Z is given then this is (4).  In either case a(x) is given.  (Elevation
    -dependent surface mass balance *must* be handled by an outer iteration.)

    Even for steep beds, setting the quadrature degree in the weak form, i.e.
    "dx(degree=Q)" with Q=4,5,6,7, seems to produce about the same result as
    the automatic mechanism.  However, Q=2 is distinctly worse."""
    v = fd.TestFunction(u.function_space())
    if Z is not None:
        du_tilt = fd.grad(u) + Z
    else:
        du_tilt = fd.grad(u) + Phi_u(u, b)
    if debug:
        dumag = fd.Function(u.function_space()).interpolate(
            fd.inner(du_tilt, du_tilt) ** 0.5
        )
        dumin, dumax = scalarrange(dumag)
        print(f"  DEBUG: {dumin:.2e} <= |du_tilt| <= {dumax:.2e}")
    Dp = (fd.inner(du_tilt, du_tilt) + epsreg) ** ((p - 2) / 2)
    C = Gamma * _omega ** (p - 1)
    return (
        C * Dp * fd.inner(du_tilt, fd.grad(v)) * fd.dx(degree=qdegree) - a * v * fd.dx
    )


def residual_u_ufl(u, a, b, epsreg = 0.01):
    """Return a UFL expression for the strong form residual of the
    u (transformed thickness) form of the shallow ice approximation.
    Also return the coefficient Z."""
    du_tilt = fd.grad(u) + Phi_u(u, b)
    Dp = (fd.inner(du_tilt, du_tilt) + epsreg) ** ((p - 2) / 2)
    C = Gamma * _omega ** (p - 1)
    Z = C * Dp
    res = - fd.div(Z * du_tilt) - a  # formally a Poisson equation here
    return res, Z


def weakform_s(s, a, b, qdegree=5, epsplap=1.0e-4, epsH=20.0, epsMass=0.0):
    """Return the nonlinear weak form for the s (transformed thickness) form
    of the shallow ice approximation.  There are two regularizations in the
    degenerate-diffusivity coefficient Dp: epsplap is a unitless surface slope
    regularization, and epsH in meters is a thickness regularization.

    epsMass (units 1/time) adds a zeroth-order term epsMass*(s-b)*v*dx: the
    mass term of an implicit time-step of the time-dependent SIA, used as a
    steady-solve "recovery strategy" in Bueler (2016).  Appendix B of Bueler
    (2025) explains *why* it works as a Newton-robustness aid: the steady
    surface motion map is not coercive where the SMB a~0 and the bed is
    flat or has a dip (Prop. B.1/B.2) -- filling such a dip with flat,
    stagnant ice and leaving it bare are then equally valid, a zero-gradient
    tie that defeats Newton.  epsMass>0 breaks the tie (filling now costs
    strictly more, proportional to epsMass*H) since, unlike epsplap/epsH,
    it acts on s itself rather than on the gradient.  epsMass=0 recovers
    the original (unregularized in this sense) form."""
    v = fd.TestFunction(s.function_space())
    gs = fd.grad(s)
    dsp = (fd.inner(gs, gs) + epsplap ** 2) ** ((p - 2) / 2)
    Z = Gamma * (s - b + epsH) ** (p + 1) * dsp
    F = Z * fd.inner(gs, fd.grad(v)) * fd.dx(degree=qdegree) - a * v * fd.dx
    if epsMass != 0.0:
        F += epsMass * (s - b) * v * fd.dx
    return F


def residual_s_ufl(s, a, b, epsplap=1.0e-4, epsH=20.0):
    """Return a UFL expression for the strong form residual of the
    s (surface elevation) form of the shallow ice approximation.
    Also return the coefficient Z."""
    gs = fd.grad(s)
    dsp = (fd.inner(gs, gs) + epsplap ** 2) ** ((p - 2) / 2)
    Z = Gamma * (s - b + epsH) ** (p + 1) * dsp
    res = - fd.div(Z * gs) - a  # formally a Poisson equation here
    return res, Z


def solve_s_continuation(
    s,
    a,
    b,
    bcs,
    lb,
    ub,
    epsH_final=20.0,
    epsH_start=0.0,
    epsplap=1.0e-4,
    dtau_years=20.0,
    qdegree=5,
    max_bisections=20,
    options_prefix="steady",
):
    """Solve the s (surface elevation) VI robustly on rough/real bed
    topography.  Combines two independent regularizations, needed for two
    different failure mechanisms (2026-08 diagnosis on eastgr.nc):

    1. epsH-continuation.  The coefficient (s-b+epsH)^(p+1) is nearly
       constant (an easy, near-linear problem) when epsH dominates the
       actual thickness field, and only becomes hard to solve by Newton
       once epsH drops to the same order as the real thickness: that's
       where Newton's Jacobian sensitivity (p+1)*(s-b+epsH)^p is largest.
       So we start epsH large, relative to the current iterate's own
       thickness scale, and step it down to epsH_final, warm-starting from
       the previous solve each time.  This mainly fixes a *cold-start*
       problem (a crude initial iterate far from the true thickness scale).

       The step is taken by bisection in log(epsH), between hi (log(epsH)
       of the most recent successful solve) and lo (log(epsH) of the
       nearest known failure).  While no failure has been recorded yet,
       each attempt jumps directly to epsH_final -- cheap, and empirically
       more robust than many small steps, since a fresh hi can sometimes
       cross the whole remaining gap in one solve.  Once a failure is
       recorded, we bisect the true bracket [lo, hi] and stop retrying
       epsH_final directly, since that retry is essentially always doomed
       once a nearby failure is known.  On a failed Newton solve, the field
       is rolled back to the last good checkpoint.  epsH_start <= 0 means
       auto-derive it from the current iterate's own thickness.

    2. epsMass, via dtau_years (a permanent regularization, not continued).
       This is the "recovery strategy" of Bueler (2016): add the mass term
       of an implicit time-step, epsMass=1/(dtau_years*secpera), to the
       steady weak form (see weakform_s).  Appendix B of Bueler (2025)
       explains why this specifically helps Newton: the steady surface
       motion map is not coercive wherever the SMB a~0 and the bed is flat
       or has a dip, and dtau_years=0 (epsMass=0) sits exactly at that
       non-unique, non-coercive limit -- so unlike epsH, epsMass should
       *not* be continued toward 0 (confirmed empirically: bridging to
       epsMass=0 fails even from dtau_years as large as 1e7).  Instead a
       fixed, moderate dtau_years is kept permanently, the way a vanishing-
       viscosity method fixes a small but nonzero viscosity to select a
       well-posed solution.  dtau_years=20 is a robust default in testing
       (dtau up to 200-500 years also worked in easier cases, but 20 was
       the only value reliably safe across all of them); dtau_years<=0
       disables this term.

    Structure: with epsMass active, a single direct solve at epsH_final is
    tried first (cheap, and often enough on its own); only on failure does
    epsH-continuation's escalate-then-bisect machinery kick in, still with
    epsMass active throughout.  Both mechanisms are independently useful --
    keep both rather than picking one.

    epsplap is not continued: ablation testing (eastgr.nc) showed it has
    negligible effect on convergence here, unlike epsH, so it stays fixed
    at its production value throughout."""
    epsMass = 0.0 if dtau_years <= 0.0 else 1.0 / (dtau_years * secpera)

    V = s.function_space()
    if epsH_start <= 0.0:
        H = fd.Function(V).interpolate(s - b)
        with H.dat.vec_ro as hv:
            hmax = hv.max()[1]
        epsH_start = max(0.5 * hmax, 20.0 * epsH_final)

    epsHc = fd.Constant(epsH_start)
    F = weakform_s(
        s, a, b, qdegree=qdegree, epsplap=epsplap, epsH=epsHc, epsMass=epsMass
    )
    problem = fd.NonlinearVariationalProblem(F, s, bcs=bcs)
    solver = fd.NonlinearVariationalSolver(
        problem, solver_parameters=solve_params, options_prefix=options_prefix
    )

    # Fast path: with epsMass active, a single direct solve at epsH_final
    # (no continuation at all) is often enough on its own -- confirmed
    # empirically to succeed even from a crude cold start.  epsH-bisection
    # below explores *intermediate* epsH values that were never validated
    # this way and can still stumble on the Appendix-B degeneracy even with
    # epsMass on, so try the direct route first and only fall back to
    # bisection if it fails.
    if epsMass != 0.0:
        checkpoint = fd.Function(V).interpolate(s)
        epsHc.assign(epsH_final)
        try:
            solver.solve(bounds=(lb, ub))
            pprint(f"    epsH-continuation: epsH={epsH_final:.3f} OK (direct, epsMass active)")
            return None
        except Exception:
            s.interpolate(checkpoint)
            pprint(f"    epsH-continuation: epsH={epsH_final:.3f} FAILED (direct), "
                   f"falling back to epsH-continuation")

    # Find a confirmed-good starting anchor: epsH_start (auto-derived or
    # given) is not guaranteed to already be safely in the trivial regime,
    # so if it fails, escalate upward (geometrically) until one succeeds.
    # This solve was previously unguarded -- a bug, since the whole
    # bisection below only works once there is a known-good anchor to
    # bisect against; escalating here closes that gap instead of raising
    # immediately on a too-small epsH_start.
    checkpoint = fd.Function(V).interpolate(s)
    trial = epsH_start
    escalations = 0
    while True:
        epsHc.assign(trial)
        try:
            solver.solve(bounds=(lb, ub))
        except Exception:
            escalations += 1
            pprint(f"    epsH-continuation: epsH={trial:.3f} FAILED (start), escalating")
            if escalations > max_bisections:
                raise RuntimeError(
                    f"epsH-continuation: could not find a working starting "
                    f"epsH after {max_bisections} escalations from "
                    f"epsH_start={epsH_start:.3f}"
                )
            s.interpolate(checkpoint)
            trial *= 4.0
            continue
        break
    pprint(f"    epsH-continuation: epsH={trial:.3f} OK (start)")
    epsH_start = trial
    checkpoint.interpolate(s)

    # Standard bisection on log(epsH), between hi (log(epsH) of the most
    # recent successful solve, checkpointed) and lo (log(epsH) of the
    # nearest known failure, or None if none found yet).  While lo is None
    # we probe the full jump straight to the target -- a fresh hi can
    # sometimes cross the whole remaining gap in one solve, as it did from
    # epsH_start.  Once a failure is recorded, we bisect the true bracket
    # [lo, hi] and stop retrying the target directly: that retry is
    # essentially always doomed once a nearby failure is already known, and
    # wastes a solve every round instead of halving the real gap.
    hi = np.log(epsH_start)
    logfinal = np.log(epsH_final)
    lo = None
    bisections = 0
    while hi > logfinal + 1.0e-9:
        if lo is None or hi - logfinal < 1.0e-3:
            cand = logfinal
        else:
            cand = 0.5 * (hi + lo)
        trial = float(np.exp(cand))
        epsHc.assign(trial)
        try:
            solver.solve(bounds=(lb, ub))
        except Exception:
            s.interpolate(checkpoint)
            lo = cand
            bisections += 1
            pprint(f"    epsH-continuation: epsH={trial:.3f} FAILED, bisecting")
            stuck = bisections > max_bisections
            # bracket collapse: hi and lo have converged to (near) the same
            # point without ever reaching epsH_final -- further bisection
            # cannot make progress (this is a stall, not slow convergence:
            # see the eastgr.nc mesh-level-1 diagnosis, 2026-08), so stop
            # and report it rather than exhausting max_bisections
            collapsed = (hi - lo) < 1.0e-6 and (cand - logfinal) > 1.0e-3
            if stuck or collapsed:
                reason = "bracket collapsed" if collapsed else "bisection budget exhausted"
                raise RuntimeError(
                    f"epsH-continuation stuck near epsH={trial:.3f} ({reason} "
                    f"after {bisections} bisections, never reached "
                    f"epsH_final={epsH_final}).  If dtau_years>0 was already "
                    f"in use, try a smaller dtau_years (stronger epsMass); "
                    f"see Appendix B, Bueler (2025)."
                )
            continue
        pprint(f"    epsH-continuation: epsH={trial:.3f} OK")
        hi = cand
        checkpoint.interpolate(s)
    return None


def surfacevelocity(s, H):
    CU = ((n + 2) / (n + 1)) * Gamma
    dss = fd.inner(fd.grad(s), fd.grad(s))
    return CU * H ** p * dss ** ((p - 2) / 2) * fd.grad(s)


def glaciermeshreport(amr, mesh, indent=2):
    nv, ne, hmin, hmax = amr.meshsizes(mesh)
    hmin /= 1000.0
    hmax /= 1000.0
    return ne, hmin


# parameters for a VI-adapted Newton method
solve_params = {
    "snes_type": "vinewtonrsls",
    "snes_vi_zero_tolerance": 1.0e-2,
    "snes_rtol": 1.0e-6,
    "snes_atol": 1.0e-10,
    "snes_stol": 0.0,
    "snes_linesearch_type": "bt",
    "snes_linesearch_order": "1",
    "snes_max_it": 1000,
    "snes_converged_reason": None,
    # "snes_max_funcs": 10000,
    # "snes_monitor": None,
    # "snes_vi_monitor": None,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}