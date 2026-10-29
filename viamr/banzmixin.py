# BS21: Banz & Schroeder (2021), J. Appl. Numer. Optim. 3(2):333-359, DOI:10.23952/jano.3.2021.2.07
# In notation of BS21, discrete primal FE approximation uh corresponds to v.
import numpy as np
from mpi4py import MPI
from firedrake import *
from firedrake.petsc import PETSc


def poincare_constant(mesh):
    """Certified Dirichlet Poincare constant C_P = L / pi for enclosing
    axis-aligned bounding box of longest side L.

    Zero extension of a function vanishing on the whole boundary of Omega
    into the axis-aligned bounding box gives ||u||_L2 <= (L / pi) ||grad u||_L2.
    The bounding box is computed with an allreduce over the mesh communicator,
    making it safe in parallel across any number of MPI ranks.
    """
    # Axis-aligned bounding box formula C_P = L/pi is an implementation choice for
    # the energy-norm setting (BS21 Remark 3.2), not a numbered paper formula.
    coords = mesh.coordinates.dat.data_ro
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    dim = coords.shape[1]
    L_max = 0.0
    for i in range(dim):
        if coords.shape[0] > 0:
            local_min = float(coords[:, i].min())
            local_max = float(coords[:, i].max())
        else:
            local_min = float("inf")
            local_max = float("-inf")
        g_min = mesh.comm.allreduce(local_min, op=MPI.MIN)
        g_max = mesh.comm.allreduce(local_max, op=MPI.MAX)
        L_max = max(L_max, g_max - g_min)
    return float(L_max / np.pi)


def banz_constants(
    beta=1.0,
    c=1.0,
    c_gamma=1.0,
    chat_gamma=1.0,
    delta=0.5,
    epsilon=1.0 / 3.0,
):
    """Calculate the abstract constants E, F, theta and derived upper bound
    coefficients c_aux, c_cons, c_comp for the Banz & Schroeder (2021)
    a posteriori error framework.

    Validates parameter intervals:
      beta > 0, c > 0, c_gamma > 0, chat_gamma > 0
      0 < delta < beta
      0 < epsilon < min(2, theta^-1)

    For default parameters (c = beta = c_gamma = chat_gamma = 1, delta = 0.5, epsilon = 1/3),
    yields E = 5, F = 6, theta = 1.5, c_aux = 10, c_cons = 36, c_comp = 12.
    """
    if beta <= 0.0:
        raise ValueError(f"beta must be positive, got {beta}")
    if c <= 0.0:
        raise ValueError(f"c must be positive, got {c}")
    if c_gamma <= 0.0:
        raise ValueError(f"c_gamma must be positive, got {c_gamma}")
    if chat_gamma <= 0.0:
        raise ValueError(f"chat_gamma must be positive, got {chat_gamma}")
    if not (0.0 < delta < beta):
        raise ValueError(f"delta must lie in (0, beta) = (0, {beta}), got {delta}")

    # Abstract constants E, F, theta from BS21 Theorem 2.3.
    E = (1.0 + 2.0 * (c ** 2) * (chat_gamma ** 2)) / (
        4.0 * delta * (beta - delta)
    ) + 2.0 * (chat_gamma ** 2)
    F = (1.0 + 2.0 * (c ** 2) * (chat_gamma ** 2)) / (beta - delta)
    theta = F * max(1.0, c_gamma ** 2) / 4.0

    max_eps = min(2.0, 1.0 / theta)
    if not (0.0 < epsilon < max_eps):
        raise ValueError(
            f"epsilon must lie in (0, min(2, theta^-1)) = (0, {max_eps}), got {epsilon}"
        )

    one_minus = 1.0 - theta * epsilon
    # Derived weights c_aux, c_cons from BS21 Theorem 2.3;
    # c_comp = epsilon * c_cons shorthand.
    c_aux = E / one_minus
    c_cons = F / (epsilon * one_minus)
    c_comp = epsilon * c_cons  # == F / one_minus

    return {
        "beta": beta,
        "c": c,
        "c_gamma": c_gamma,
        "chat_gamma": chat_gamma,
        "delta": delta,
        "epsilon": epsilon,
        "E": E,
        "F": F,
        "theta": theta,
        "c_aux": c_aux,
        "c_cons": c_cons,
        "c_comp": c_comp,
    }


def banz_aux_estimator(uh, kappa, f_ufl=None, estimator="br78", degree=3, **kwargs):
    """Auxiliary linear PDE error estimator for -Delta v_hat = f + kappa.

    Default 'br78' evaluates the cell volume residual and interior facet jumps
    across the whole domain.

    Returns (eta_aux, aux_eta2), where eta_aux is a DG0 Function of elementwise
    indicators and aux_eta2 is the scalar sum_T eta_aux_T^2.
    """
    if estimator != "br78":
        raise ValueError(f"estimator='{estimator}' not supported; must be 'br78'")

    mesh = uh.function_space().mesh()
    DG0 = FunctionSpace(mesh, "DG", 0)
    v = CellVolume(mesh)
    h = CellDiameter(mesh)
    n = FacetNormal(mesh)

    f = f_ufl if f_ufl is not None else Constant(0.0)
    if isinstance(f, (int, float)):
        f = Constant(f)
    if isinstance(kappa, (int, float)):
        kappa = Constant(kappa)

    # BS21 (2.5)/(3.6): auxiliary problem -Delta v_hat = f + kappa.
    # Its Galerkin condition is (2.14)/(3.4); res is the strong residual.
    res = f + kappa + div(grad(uh))

    eta_sq = Function(DG0)
    w = TestFunction(DG0)
    dx_m = dx(domain=mesh, degree=degree)
    dS_m = dS(domain=mesh, degree=degree)

    # Residual error indicator: cf. BS21 (3.7), using P1-style BR78 cell-diameter
    # weights and full data rather than literal hp-projected formula.
    G = inner(eta_sq / v, w) * dx_m
    G -= inner(h ** 2 * res ** 2, w) * dx_m
    G -= 0.5 * inner(h("+") * jump(grad(uh), n) ** 2, w("+")) * dS_m
    G -= 0.5 * inner(h("-") * jump(grad(uh), n) ** 2, w("-")) * dS_m

    sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
    solve(G == 0, eta_sq, solver_parameters=sp)

    eta_aux = Function(DG0, name="eta_aux").interpolate(sqrt(max_value(eta_sq, 0.0)))
    with eta_sq.dat.vec_ro as vec:
        local_sum = float(np.sum(np.maximum(vec.array_r, 0.0)))
    # Global auxiliary residual estimator sum eta_aux^2, BS21 (3.7).
    aux_eta2 = float(mesh.comm.allreduce(local_sum, op=MPI.SUM))

    return eta_aux, aux_eta2


def banz_weighted_functional(
    uh,
    kappa,
    psi,
    C_P,
    epsilon=1.0 / 3.0,
    cutoff="sco",
    solver_parameters=None,
    drop_cross_neg=True,
    degree=3,
    **kwargs,
):
    """Evaluate the Banz & Schroeder weighted functional Xi_eps with cut-off
    in ('sco', 'feco').

    Parameters
    ----------
    uh : Function
        Primal finite element approximation.
    kappa : Function or Constant
        Multiplier approximation.
    psi : Function, Constant, or UFL expression
        Lower obstacle.
    C_P : float
        Poincare constant for bounding the dual norm ||kappa_-||_H^-1_D.
    epsilon : float, optional
        Weighting parameter. Default is 1/3.
    cutoff : str, optional
        'sco': Standard Cut-Off (algebraic cut-offs mu = kappa_+, zeta = (psi - v)_+).
        'feco': Finite Element Cut-Off (solves discrete auxiliary VI for zeta_h^*,
               then truncates zeta_FECO = max(zeta_h^*, psi - v), mu_FECO = kappa_+).
    solver_parameters : dict, optional
        PETSc solver parameters for the auxiliary VI solve in 'feco'.
    drop_cross_neg : bool, optional
        Whether to drop the non-positive -eps * (kappa_-, (psi - v)_+) cross term
        for the upper bound. Default is True.
    degree : int, optional
        Quadrature degree for assembling integrals. Default is 3.

    Returns
    -------
    xi : float
        Scalar value of Xi_eps.
    components : dict
        Detailed breakdown:
        'N2': C_P^2 * ||kappa_-||_L2^2
        'grad_zeta2': ||grad zeta||_L2^2
        'comp_pairing': pairing term
        'xi': xi
        'xi_T': DG0 Function of cellwise contributions
        'zeta': zeta_h^* Function (for FECO) or None (for SCO)
    """
    if cutoff not in ("sco", "feco"):
        raise ValueError(f"unknown cutoff '{cutoff}', must be 'sco' or 'feco'")

    mesh = uh.function_space().mesh()
    DG0 = FunctionSpace(mesh, "DG", 0)
    v = CellVolume(mesh)
    w = TestFunction(DG0)
    dx_m = dx(domain=mesh, degree=degree)

    if isinstance(kappa, (int, float)):
        kappa = Constant(kappa)
    if isinstance(psi, (int, float)):
        psi = Constant(psi)

    # Multiplier negativity term: N_Omega^2 <= C_P^2 * ||kappa_-||_L2^2
    # Dual defect with mu = kappa_+ (BS21 (3.10)); C_P * ||kappa_-||_L2 provides an
    # L2 upper bound on the H^-1_D dual norm (BS21 Section 2.2).
    kappa_minus = max_value(-kappa, 0.0)
    N_integrand = Constant(C_P ** 2) * kappa_minus ** 2
    N2 = float(assemble(N_integrand * dx_m))

    if cutoff == "sco":
        zeta_field = None
        # By Stampacchia's theorem for (psi - uh)_+ = max(psi - uh, 0):
        # BS21 Section 2.2: zeta = (psi - uh)_+ belongs to K_v.
        # This corrects the reversed cut-off sign printed in (3.11).
        grad_integrand = conditional(
            psi - uh > 0.0, inner(grad(psi - uh), grad(psi - uh)), 0.0
        )
        grad_zeta2 = float(assemble(grad_integrand * dx_m))

        # Positive part decompositions for complementarity pairing, BS21 (3.10).
        uh_minus_psi_pos = max_value(uh - psi, 0.0)
        psi_minus_uh_pos = max_value(psi - uh, 0.0)
        kappa_plus = max_value(kappa, 0.0)

        # Expand the two pairings in Xi_eps (BS21 Section 2.2).
        # Dropping the non-positive cross-term enlarges the functional.
        if drop_cross_neg:
            pair_integrand = kappa_plus * uh_minus_psi_pos
        else:
            pair_integrand = kappa_plus * uh_minus_psi_pos - kappa_minus * psi_minus_uh_pos

        comp_pairing = float(assemble(pair_integrand * dx_m))

    elif cutoff == "feco":
        V = uh.function_space()
        zeta_h_star = Function(V, name="zeta_h_star").assign(0.0)
        y = TestFunction(V)
        # Discrete auxiliary VI for zeta_h^*, BS21 (3.16)/(3.13) under energy product (Remark 3.2).
        F_feco = (
            inner(grad(zeta_h_star), grad(y)) * dx_m
            - inner(-0.5 * Constant(epsilon) * kappa, y) * dx_m
        )
        bc = DirichletBC(V, Constant(0.0), "on_boundary")
        problem = NonlinearVariationalProblem(F_feco, zeta_h_star, bcs=bc)

        sp = {
            "snes_type": "vinewtonrsls",
            "snes_rtol": 1.0e-8,
            "snes_atol": 1.0e-12,
            "snes_stol": 1.0e-12,
        }
        if solver_parameters is not None:
            sp.update(solver_parameters)
        solver = NonlinearVariationalSolver(problem, solver_parameters=sp)
        # Discrete obstacle bounds defining K_v, BS21 (3.16).
        lb_zeta = Function(V).interpolate(psi - uh)
        ub_zeta = Function(V).interpolate(Constant(PETSc.INFINITY))
        solver.solve(bounds=(lb_zeta, ub_zeta))
        zeta_field = zeta_h_star

        # Stampacchia gradient for max(zeta_h_star, psi - uh)
        # Repaired cut-off zeta_FECO = max(zeta_h^*, psi - uh) ensures zeta in K_v (BS21 Theorem 3.1).
        grad_integrand = conditional(
            zeta_h_star >= psi - uh,
            inner(grad(zeta_h_star), grad(zeta_h_star)),
            inner(grad(psi - uh), grad(psi - uh)),
        )
        grad_zeta2 = float(assemble(grad_integrand * dx_m))

        # Retains mu = kappa_+ standard cut-off rather than multiplier problem BS21 (3.15).
        kappa_plus = max_value(kappa, 0.0)
        zeta_feco = conditional(zeta_h_star >= psi - uh, zeta_h_star, psi - uh)
        pair_integrand = kappa_plus * (uh - psi) + kappa * zeta_feco
        comp_pairing = float(assemble(pair_integrand * dx_m))

    # Scalar weighted functional Xi_eps: unnumbered definition in BS21 Section 2.2 (p. 337).
    xi = N2 + grad_zeta2 + epsilon * comp_pairing

    # Cellwise DG0 representation of xi
    integrand_xi = N_integrand + grad_integrand + Constant(epsilon) * pair_integrand
    xi_T = Function(DG0, name="xi_T")
    G_xi = inner(xi_T / v, w) * dx_m - inner(integrand_xi, w) * dx_m
    solve_sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
    solve(G_xi == 0, xi_T, solver_parameters=solve_sp)

    components = {
        "N2": N2,
        "grad_zeta2": grad_zeta2,
        "comp_pairing": comp_pairing,
        "xi": xi,
        "xi_T": xi_T,
        "zeta": zeta_field,
    }
    return xi, components


class BanzMarkingsMixin:
    r"""Mixed into class VIAMR (see viamr.py): marking by the a posteriori
    error estimator of Banz & Schroeder (2021) for obstacle problems.
    The public method is banzmark()."""

    def poincare_constant(self, mesh):
        """Certified Dirichlet Poincare constant C_P = L / pi for enclosing
        axis-aligned bounding box."""
        return poincare_constant(mesh)

    def banz_constants(
        self,
        beta=1.0,
        c=1.0,
        c_gamma=1.0,
        chat_gamma=1.0,
        delta=0.5,
        epsilon=1.0 / 3.0,
    ):
        """Calculate abstract constants E, F, theta, c_aux, c_cons, c_comp."""
        return banz_constants(
            beta=beta,
            c=c,
            c_gamma=c_gamma,
            chat_gamma=chat_gamma,
            delta=delta,
            epsilon=epsilon,
        )

    def banz_aux_estimator(
        self, uh, kappa, f_ufl=None, estimator="br78", degree=3, **kwargs
    ):
        """Auxiliary linear PDE error estimator across the entire domain."""
        return banz_aux_estimator(
            uh, kappa, f_ufl=f_ufl, estimator=estimator, degree=degree, **kwargs
        )

    def banz_weighted_functional(
        self,
        uh,
        kappa,
        psi,
        C_P,
        epsilon=1.0 / 3.0,
        cutoff="sco",
        solver_parameters=None,
        drop_cross_neg=True,
        degree=3,
        **kwargs,
    ):
        """Evaluate weighted functional Xi_eps with cut-off in ('sco', 'feco')."""
        return banz_weighted_functional(
            uh,
            kappa,
            psi,
            C_P,
            epsilon=epsilon,
            cutoff=cutoff,
            solver_parameters=solver_parameters,
            drop_cross_neg=drop_cross_neg,
            degree=degree,
            **kwargs,
        )

    def banzmark(
        self,
        uh,
        bounds,
        kappa,
        f_ufl=None,
        bounds_ufl=(None, None),
        cutoff="sco",
        aux_estimator="br78",
        method="total",
        theta=0.5,
        epsilon=1.0 / 3.0,
        delta=0.5,
        C_P=None,
        beta=1.0,
        c=1.0,
        c_gamma=1.0,
        chat_gamma=1.0,
        degree=3,
        solver_parameters=None,
        drop_cross_neg=True,
        **kwargs,
    ):
        """Mark elements according to the a posteriori error estimator of Banz &
        Schroeder (2021) for variational inequalities (obstacle problems).

        Parameters
        ----------
        uh : Function
            The computed finite element solution (e.g. in CG1).
        bounds : tuple or Function or Constant
            The obstacle constraint. Must represent a lower obstacle alone,
            either as a single object (lb) or a tuple (lb, None).
        kappa : Function or Constant
            The Lagrange multiplier approximation. REQUIRED (cannot be None).
        f_ufl : UFL expression or Function or Constant, optional
            The load term. If None, defaults to Constant(0.0).
        bounds_ufl : tuple, optional
            (lb_ufl, ub_ufl) continuum obstacle expressions, if known.
        cutoff : str, optional
            Cut-off strategy in ('sco', 'feco'). Default is 'sco'.
            'sco': Standard Cut-Off (purely algebraic).
            'feco': Finite Element Cut-Off (solves discrete auxiliary VI for zeta_h^*).
        aux_estimator : str, optional
            Auxiliary linear PDE estimator across the whole domain. Default is 'br78'.
        method : str, optional
            Marking strategy for fixedratemark ('total' or 'max'). Default is 'total'.
        theta : float, optional
            Refinement fraction threshold for fixedratemark. Default is 0.5.
        epsilon : float, optional
            Weighting parameter for Xi_eps. Default is 1/3.
        delta : float, optional
            Abstract parameter in (0, beta). Default is 0.5.
        C_P : float, optional
            Certified Dirichlet Poincare constant. If None, computed from bounding box.
        beta, c, c_gamma, chat_gamma : float, optional
            Abstract parameters. Default to 1.0.

        Returns
        -------
        mark : Function in DG0
            The element marking (1.0 for marked, 0.0 otherwise).
        fields : dict
            Dictionary of diagnostic fields and components:
            - 'eta': DG0 indicator with weighted consistency (sum_T eta_T^2 == eta_up^2)
            - 'eta_aux': DG0 auxiliary PDE error indicator
            - 'aux_eta2': scalar auxiliary error squared
            - 'xi': scalar weighted functional Xi_eps
            - 'xi_T': DG0 weighted functional indicator
            - 'N2': multiplier negativity contribution
            - 'grad_zeta2': non-penetration gradient contribution
            - 'comp_pairing': complementarity pairing contribution
            - 'c_aux', 'c_cons', 'c_comp': derived estimator constants
            - 'C_P': Poincare constant
            - 'ethresh': threshold returned by fixedratemark
            - 'eta_up': certified total error bound
            - 'zeta': auxiliary VI solution (for FECO) or None (for SCO)
            - 'cutoff': cutoff method used ('sco' or 'feco')
            - 'constants': dictionary of derived constants
        eta_up : float
            The certified upper bound on the total error norm:
            sqrt(c_aux * aux_eta2 + c_cons * xi).
        """
        if kappa is None:
            raise ValueError("kappa is a required parameter for banzmark")

        if isinstance(bounds, (tuple, list)):
            lb, ub = bounds
            if ub is not None:
                raise NotImplementedError(
                    "Banz-Schroeder estimator currently only supports unilateral lower obstacle problems (bounds=(lb, None))"
                )
        else:
            lb = bounds

        if isinstance(bounds_ufl, (tuple, list)):
            lb_ufl, ub_ufl = bounds_ufl
            if ub_ufl is not None:
                raise NotImplementedError(
                    "Banz-Schroeder estimator currently only supports unilateral lower obstacle problems"
                )
        else:
            lb_ufl = bounds_ufl

        if lb is None and lb_ufl is None:
            raise ValueError("bounds must specify a lower obstacle")

        psi = lb_ufl if lb_ufl is not None else lb

        mesh = uh.function_space().mesh()
        if C_P is None:
            C_P = poincare_constant(mesh)

        consts = banz_constants(
            beta=beta,
            c=c,
            c_gamma=c_gamma,
            chat_gamma=chat_gamma,
            delta=delta,
            epsilon=epsilon,
        )
        c_aux = consts["c_aux"]
        c_cons = consts["c_cons"]
        c_comp = consts["c_comp"]

        eta_aux, aux_eta2 = banz_aux_estimator(
            uh, kappa, f_ufl=f_ufl, estimator=aux_estimator, degree=degree, **kwargs
        )

        xi, xi_components = banz_weighted_functional(
            uh,
            kappa,
            psi,
            C_P,
            epsilon=epsilon,
            cutoff=cutoff,
            solver_parameters=solver_parameters,
            drop_cross_neg=drop_cross_neg,
            degree=degree,
            **kwargs,
        )

        # Weights from BS21 Theorem 2.3; (2.7) supplies the reliability factor
        # needed when replacing the auxiliary error by a residual estimator.
        eta_up2 = c_aux * aux_eta2 + c_cons * xi
        eta_up = float(np.sqrt(max(eta_up2, 0.0)))

        # Construct weighted DG0 indicator eta_T with weighted consistency
        # sum_T eta_T^2 == eta_up^2
        _, DG0 = self.spaces(mesh)
        eta_sq_T = Function(DG0)
        eta_sq_T.interpolate(
            Constant(c_aux) * eta_aux ** 2
            + Constant(c_cons) * xi_components["xi_T"]
        )
        eta_T = Function(DG0, name="eta (banz)").interpolate(
            sqrt(max_value(eta_sq_T, 0.0))
        )

        mark, ethresh = self.fixedratemark(eta_T, theta=theta, method=method)

        fields = {
            "eta": eta_T,
            "eta_aux": eta_aux,
            "aux_eta2": aux_eta2,
            "xi": xi,
            "xi_T": xi_components["xi_T"],
            "N2": xi_components["N2"],
            "grad_zeta2": xi_components["grad_zeta2"],
            "comp_pairing": xi_components["comp_pairing"],
            "c_aux": c_aux,
            "c_cons": c_cons,
            "c_comp": c_comp,
            "C_P": C_P,
            "ethresh": ethresh,
            "eta_up": eta_up,
            "zeta": xi_components["zeta"],
            "cutoff": cutoff,
            "constants": consts,
        }
        return mark, fields, eta_up
