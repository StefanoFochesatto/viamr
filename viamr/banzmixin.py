import math
import numpy as np
from mpi4py import MPI
from firedrake import *
from firedrake.petsc import PETSc
import ufl


class BanzFields(dict):
    r"""A dictionary of DG0 diagnostic indicator fields produced by banzmark(),
    with attributes for accessing scalar error components and branch dominance.

    Fields accessible by key:
      "eta":              total elementwise DG0 error indicator
      "load":             elementwise load/residual contribution (h_T * ||f + kappa||)
      "nonpenetration":   elementwise non-penetration contribution (||grad q_-||)
      "complementarity":  elementwise complementarity defect ((kappa_+, q_+)^0.5)

    Attributes:
      info:               dictionary of all raw and derived scalar quantities
      dominant_branch:    "auxiliary", "constraint", or "balanced"
      aux_part2:          auxiliary branch contribution c_aux * R^2
      constraint_part2:   constraint branch contribution (c_cons * (N^2 + ||grad q_-||^2) + c_comp * (kappa_+, q_+))
      eta_up:             global certified upper bound (sqrt(aux_part2 + constraint_part2))
    """

    @property
    def info(self):
        return self.get("info", {})

    @property
    def dominant_branch(self):
        return self.info.get("dominant_branch", "balanced")

    @property
    def aux_part2(self):
        return self.info.get("aux_part2", 0.0)

    @property
    def constraint_part2(self):
        return self.info.get("constraint_part2", 0.0)

    @property
    def eta_up(self):
        return self.info.get("eta_up", 0.0)

    def summary(self):
        """A one-line summary string of the estimator result."""
        return (
            f"eta_up = {self.eta_up:.3e} "
            f"(aux: {self.aux_part2:.3e}, constraint: {self.constraint_part2:.3e}, "
            f"dominant: {self.dominant_branch})"
        )


class BanzMarkingsMixin:
    r"""Mixed into class VIAMR (see viamr.py): a posteriori error estimation and
    marking for obstacle problems based on the abstract framework of Banz &
    Schroeder (2021) and Proximal Galerkin (Keith & Surowiec 2024, Keith,
    Masri, & Zeinhofer 2026).

    Theoretical basis:
      * Lothar Banz & Andreas Schroeder (2021). "A posteriori error control for
        variational inequalities with linear constraints in an abstract framework."
        Journal of Applied and Numerical Optimization 3(2), 333-359.
      * Brendan Keith & Thomas M. Surowiec (2024). "Proximal Galerkin: A
        structure-preserving finite element method for pointwise bound constraints."
        arXiv:2307.12444.
      * "A posteriori error control for proximal Galerkin approximations of
        variational inequalities" (2026 draft).

    Key property:
      The error bound holds for ANY primal-dual pair (v, kappa) in Z x Y*, without
      requiring v to solve a discrete VI or kappa to lie in a discrete cone.
      In Proximal Galerkin, the latent variable increment kappa = (s^{k-1} - s^k)/alpha
      satisfies Banz-Schroeder's Galerkin relation for free.
    """

    def _banz_constants(self, delta=0.5, epsilon=1.0 / 3.0):
        r"""Compute the Banz-Schroeder abstract framework constants for the
        obstacle problem in H^1_0(Omega) with the energy norm:
            c = beta = c_gamma = chat_gamma = 1.
        
        For delta in (0, beta) and epsilon in (0, min(2, theta^-1)):
            E = (1 + 2*c^2*chat_gamma^2) / (4*delta*(beta - delta)) + 2*chat_gamma^2
            F = (1 + 2*c^2*chat_gamma^2) / (beta - delta)
            theta = F * max(1, c_gamma^2) / 4
            c_aux  = E / (1 - theta*epsilon)
            c_cons = F / (epsilon * (1 - theta*epsilon))
            c_comp = epsilon * c_cons

        The standard defaults delta = 1/2, epsilon = 1/3 yield:
            E = 5, F = 6, theta = 3/2,
            c_aux = 10, c_cons = 36, c_comp = 12.
        """
        beta = c = c_gamma = chat_gamma = 1.0
        if not (0.0 < delta < beta):
            raise ValueError(f"delta must lie in (0, {beta}), got {delta}")
        E = (1.0 + 2.0 * c * c * chat_gamma * chat_gamma) / (
            4.0 * delta * (beta - delta)
        ) + 2.0 * chat_gamma * chat_gamma
        F = (1.0 + 2.0 * c * c * chat_gamma * chat_gamma) / (beta - delta)
        theta = F * max(1.0, c_gamma * c_gamma) / 4.0
        max_eps = min(2.0, 1.0 / theta)
        if not (0.0 < epsilon < max_eps):
            raise ValueError(
                f"epsilon must lie in (0, {max_eps:.4f}), got {epsilon}"
            )
        one_minus = 1.0 - theta * epsilon
        c_aux = E / one_minus
        c_cons = F / (epsilon * one_minus)
        c_comp = epsilon * c_cons
        return {
            "c": c,
            "beta": beta,
            "c_gamma": c_gamma,
            "chat_gamma": chat_gamma,
            "E": E,
            "F": F,
            "theta": theta,
            "c_aux": c_aux,
            "c_cons": c_cons,
            "c_comp": c_comp,
        }

    def _banz_poincare(self, mesh):
        r"""Compute the certified Dirichlet Poincare constant C_P = L/pi for an
        axis-aligned bounding box of side length L enclosing mesh Omega.
        This constant satisfies ||z||_{L^2} <= C_P ||grad z||_{L^2} for any
        z in H^1_0(Omega) with full Dirichlet boundary conditions.
        Parallel-safe across MPI ranks.
        """
        coords = mesh.coordinates.dat.data_ro
        if coords.ndim == 1:
            coords = coords.reshape(-1, 1)
        dim = coords.shape[1]
        lo = np.empty(dim)
        hi = np.empty(dim)
        for i in range(dim):
            local_min = (
                float(coords[:, i].min()) if coords.shape[0] > 0 else float("inf")
            )
            local_max = (
                float(coords[:, i].max()) if coords.shape[0] > 0 else float("-inf")
            )
            lo[i] = mesh.comm.allreduce(local_min, op=MPI.MIN)
            hi[i] = mesh.comm.allreduce(local_max, op=MPI.MAX)
        L = float(np.max(hi - lo))
        return L / math.pi

    def _banz_flux_bound(
        self, v, kappa, f_load, C_P, tau, mesh, quadrature_degree=8
    ):
        r"""Evaluate the auxiliary-branch bound R_Omega(v, kappa; tau) (draft Lemma 4.1):
            ||Bv - ell - kappa||_{Z*} <= R_Omega(v, kappa; tau)
                                      = ||grad v - tau||_{L^2} + C_P ||f + kappa + div tau||_{L^2}
        plus interior face jump terms when tau is discontinuous (tau='grad').
        """
        fcp = {"quadrature_degree": quadrature_degree}
        dxm = Measure("dx", domain=mesh)
        dSm = Measure("dS", domain=mesh)

        f_term = (
            Constant(0.0)
            if f_load is None
            else (
                Constant(f_load)
                if isinstance(f_load, (int, float))
                else f_load
            )
        )
        k_term = (
            Constant(kappa) if isinstance(kappa, (int, float)) else kappa
        )

        if tau == "zero":
            norm_grad_v = math.sqrt(
                float(
                    assemble(
                        inner(grad(v), grad(v)) * dxm,
                        form_compiler_parameters=fcp,
                    )
                )
            )
            norm_res = math.sqrt(
                float(
                    assemble(
                        (f_term + k_term) ** 2 * dxm,
                        form_compiler_parameters=fcp,
                    )
                )
            )
            return norm_grad_v + C_P * norm_res, 0.0

        elif tau == "grad":
            g = grad(v)
            div_g = div(g)
            norm_res = math.sqrt(
                float(
                    assemble(
                        (f_term + k_term + div_g) ** 2 * dxm,
                        form_compiler_parameters=fcp,
                    )
                )
            )
            # Interior face jumps: sum_F h_F ||[grad v . n]||^2
            if mesh.topological_dimension >= 2:
                n = FacetNormal(mesh)
                jump2 = float(
                    assemble(
                        FacetArea(mesh) * jump(g, n) ** 2 * dSm,
                        form_compiler_parameters=fcp,
                    )
                )
            else:
                jump2 = 0.0
            return C_P * norm_res + math.sqrt(max(jump2, 0.0)), jump2

        else:
            raise NotImplementedError(
                f"tau={tau!r} not supported; choose 'grad' or 'zero'"
            )

    def _banz_constraint_terms(
        self, v, kappa, phi, C_P, mesh, quadrature_degree=8
    ):
        r"""Evaluate the constraint branch terms using the admissible cut-off pair:
            mu = kappa_+ = max{kappa, 0} in G*
            zeta = q_- = max{phi - v, 0} in K_v
        where q = v - phi.

        Returns (N2, grad_q_minus2, comp_term, cross_neg):
          * N2 = C_P^2 ||kappa_-||^2_{L^2}
          * grad_q_minus2 = ||grad q_-||^2_{L^2} assembled via the Stampacchia
            identity conditional(q < 0, inner(grad(q), grad(q)), 0)
          * comp_term = (kappa_+, q_+)_{L^2}
          * cross_neg = (kappa_-, q_-)_{L^2} (non-negative)
        """
        fcp = {"quadrature_degree": quadrature_degree}
        dxm = Measure("dx", domain=mesh)

        k_val = Constant(kappa) if isinstance(kappa, (int, float)) else kappa
        phi_val = Constant(phi) if isinstance(phi, (int, float)) else phi

        q = v - phi_val
        kappa_pos = conditional(k_val > 0.0, k_val, 0.0)
        kappa_neg = conditional(k_val < 0.0, -k_val, 0.0)
        q_pos = conditional(q > 0.0, q, 0.0)
        q_neg = conditional(q < 0.0, -q, 0.0)

        k_neg2 = float(
            assemble(kappa_neg ** 2 * dxm, form_compiler_parameters=fcp)
        )
        N2 = (C_P ** 2) * k_neg2

        grad_q_minus2 = float(
            assemble(
                conditional(q < 0.0, inner(grad(q), grad(q)), 0.0) * dxm,
                form_compiler_parameters=fcp,
            )
        )

        comp_term = float(
            assemble(kappa_pos * q_pos * dxm, form_compiler_parameters=fcp)
        )
        cross_neg = float(
            assemble(kappa_neg * q_neg * dxm, form_compiler_parameters=fcp)
        )

        return N2, grad_q_minus2, comp_term, cross_neg

    def _banz_local_indicators(
        self, v, kappa, phi, f_load, mesh, quadrature_degree=8
    ):
        r"""Compute elementwise DG0 indicators:
            eta_{T,load}^2  = h_T^2 ||f + kappa||^2_{L^2(T)}
            eta_{T,nonpen}^2 = ||grad q_-||^2_{L^2(T)}
            eta_{T,comp}^2   = (kappa_+, q_+)_{L^2(T)}
            eta_T^2         = eta_{T,load}^2 + eta_{T,nonpen}^2 + eta_{T,comp}^2
        """
        _, DG0 = self.spaces(mesh)
        t0 = TestFunction(DG0)
        fcp = {"quadrature_degree": quadrature_degree}
        dxm = Measure("dx", domain=mesh)

        f_term = (
            Constant(0.0)
            if f_load is None
            else (
                Constant(f_load)
                if isinstance(f_load, (int, float))
                else f_load
            )
        )
        k_val = Constant(kappa) if isinstance(kappa, (int, float)) else kappa
        phi_val = Constant(phi) if isinstance(phi, (int, float)) else phi

        q = v - phi_val
        kappa_pos = conditional(k_val > 0.0, k_val, 0.0)
        q_pos = conditional(q > 0.0, q, 0.0)

        load2_vec = np.maximum(
            assemble(
                (f_term + k_val) ** 2 * t0 * dxm,
                form_compiler_parameters=fcp,
            ).dat.data_ro,
            0.0,
        )
        nonpen2_vec = np.maximum(
            assemble(
                conditional(q < 0.0, inner(grad(q), grad(q)), 0.0) * t0 * dxm,
                form_compiler_parameters=fcp,
            ).dat.data_ro,
            0.0,
        )
        comp_vec = np.maximum(
            assemble(
                kappa_pos * q_pos * t0 * dxm,
                form_compiler_parameters=fcp,
            ).dat.data_ro,
            0.0,
        )

        h_T = Function(DG0).interpolate(CellDiameter(mesh)).dat.data_ro

        load_eta = h_T * np.sqrt(load2_vec)
        nonpen_eta = np.sqrt(nonpen2_vec)
        comp_eta = np.sqrt(comp_vec)

        total_eta2 = load_eta ** 2 + nonpen_eta ** 2 + comp_eta ** 2
        total_eta = np.sqrt(total_eta2)

        def _make_field(values, name):
            fld = Function(DG0, name=name)
            fld.dat.data[:] = values
            return fld

        return BanzFields({
            "eta": _make_field(total_eta, "banz_indicator"),
            "load": _make_field(load_eta, "load_residual"),
            "nonpenetration": _make_field(nonpen_eta, "nonpenetration"),
            "complementarity": _make_field(comp_eta, "complementarity"),
        })

    def banz_aux_residual_gram_norm(
        self, uh, kappa, f_load=None, bcs=None, quadrature_degree=8
    ):
        r"""Compute the discrete auxiliary residual norm in the Z-Gram norm:
            rho = sup_{z in Z_h} |<r, z>| / ||z||_Z = sqrt(r^T A^{-1} r)
        where A is the stiffness Gram matrix (with Dirichlet boundary rows
        replaced by identity) and r is the residual vector with Dirichlet rows zeroed.
        """
        V = uh.function_space()
        mesh = V.mesh()
        fcp = {"quadrature_degree": quadrature_degree}
        dxm = Measure("dx", domain=mesh)
        z = TestFunction(V)

        f_term = (
            Constant(0.0)
            if f_load is None
            else (
                Constant(f_load)
                if isinstance(f_load, (int, float))
                else f_load
            )
        )
        k_val = Constant(kappa) if isinstance(kappa, (int, float)) else kappa

        r = assemble(
            (inner(grad(uh), grad(z)) - (f_term + k_val) * z) * dxm,
            form_compiler_parameters=fcp,
        )

        if bcs is None:
            # Default to full Dirichlet on boundary
            bcs = [DirichletBC(V, 0.0, "on_boundary")]
        elif not isinstance(bcs, (list, tuple)):
            bcs = [bcs]

        # Homogeneous Dirichlet BCs on V
        hom_bcs = [DirichletBC(V, 0.0, bc.sub_domain) for bc in bcs]
        bc_nodes = []
        for bc in hom_bcs:
            bc_nodes.extend(bc.nodes)
        bc_nodes = np.unique(bc_nodes)
        if len(bc_nodes) > 0:
            r.dat.data[bc_nodes] = 0.0

        a = inner(grad(TrialFunction(V)), grad(z)) * dxm
        A = assemble(a, bcs=hom_bcs)
        x = Function(V)
        solve(
            A,
            x,
            r,
            solver_parameters={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )
        return math.sqrt(max(float(np.dot(r.dat.data_ro, x.dat.data_ro)), 0.0))

    def banz_moment_defect(
        self, uh, phi, s=None, recon=None, W=None, quadrature_degree=8
    ):
        r"""Compute the moment defect d_{hp} = ||P_W(q - e^s)||_{L^2}:
            d_{hp} = sqrt(d^T H^{-1} d)
        where H is the mass matrix on the latent space W, and d is the moment residual.
        If recon is provided (e.g. e^s + phi), u - recon is used.
        """
        V = uh.function_space()
        mesh = V.mesh()
        if W is None:
            if s is not None and hasattr(s, "function_space"):
                W = s.function_space()
            elif recon is not None and hasattr(recon, "function_space"):
                W = recon.function_space()
            else:
                W = V

        fcp = {"quadrature_degree": quadrature_degree}
        dxm = Measure("dx", domain=mesh)
        w = TestFunction(W)

        phi_val = Constant(phi) if isinstance(phi, (int, float)) else phi
        if recon is not None:
            diff = uh - recon
        elif s is not None:
            diff = (uh - phi_val) - exp(s)
        else:
            raise ValueError("Either s (latent) or recon must be provided.")

        d_vec = assemble(diff * w * dxm, form_compiler_parameters=fcp)
        H = assemble(inner(TrialFunction(W), w) * dxm)
        y = Function(W)
        solve(
            H,
            y,
            d_vec,
            solver_parameters={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )
        return math.sqrt(
            max(float(np.dot(d_vec.dat.data_ro, y.dat.data_ro)), 0.0)
        )

    def banzmark(
        self,
        uh,
        bounds,
        kappa=None,
        f_ufl=None,
        bounds_ufl=(None, None),
        method="total",
        theta=0.5,
        C_P=None,
        tau="grad",
        delta=0.5,
        epsilon=1.0 / 3.0,
        quadrature_degree=8,
        bcs=None,
        s=None,
        recon=None,
        W=None,
        threshold_dom=2.0,
    ):
        r"""Compute element markings and the Banz-Schroeder a posteriori error
        bound for the classical obstacle problem:
            -Delta u = f  in Omega,   u >= phi,   u = 0 on dOmega.

        Parameters:
          uh:         Primal approximation Function in CG1/CGp.
                      (Alternatively, an LVPP solver object can be passed here).
          bounds:     Tuple (lb, ub). Lower obstacle lb must be given, ub must be None.
          kappa:      Multiplier approximation Function, Constant, or float.
                      In Proximal Galerkin, kappa = (s^{k-1} - s^k) / alpha.
          f_ufl:      Load term f (float, Constant, Function, or UFL expression).
          bounds_ufl: Tuple (lb_ufl, ub_ufl) of continuum obstacles if available.
          method:     Marking method passed to fixedratemark(): "total" (Doerfler/bulk)
                      or "max". Default "total".
          theta:      Marking parameter in (0, 1]. Default 0.5.
          C_P:        Dirichlet Poincare constant (default: bounding box L/pi).
          tau:        Auxiliary flux choice: "grad" (default) or "zero".
          delta:      Banz parameter in (0, 1). Default 0.5.
          epsilon:    Banz parameter in (0, 2/3). Default 1/3.
          quadrature_degree: Quadrature degree for assembled forms. Default 8.
          bcs:        Boundary conditions (optional, for Gram residual rho).
          s:          Latent variable Function (optional, for moment defect d_{hp}).
          recon:      Observable Function (optional, e.g. exp(s) + phi).
          W:          Latent function space (optional).
          threshold_dom: Dominance threshold ratio for auxiliary vs constraint
                      branches (default 2.0).

        Returns:
          (mark, fields, Eh)
          * mark:   DG0 element marking Function with values in {0.0, 1.0}.
          * fields: BanzFields dictionary of DG0 indicator fields ("eta", "load",
                    "nonpenetration", "complementarity") and branch diagnostic info.
          * Eh:     Scalar certified upper bound eta_up.
        """
        # Support passing an LVPP solver directly
        if hasattr(uh, "u_out") and hasattr(uh, "drift"):
            lvpp = uh
            uh = lvpp.u_out[0]
            if kappa is None and hasattr(lvpp, "drift") and len(lvpp.drift) > 0:
                kappa = lvpp.drift[0]
            if bounds is None and hasattr(lvpp, "constraints"):
                bounds = (lvpp.constraints[0].lower, None)
            if bcs is None and hasattr(lvpp, "bcs"):
                bcs = lvpp.bcs
            if recon is None and hasattr(lvpp, "_system") and hasattr(lvpp._system, "recon"):
                recon = lvpp._system.recon[0]
            if s is None and hasattr(lvpp, "psi_out") and len(lvpp.psi_out) > 0:
                s = lvpp.psi_out[0]
            if W is None and hasattr(lvpp, "latent_spaces") and len(lvpp.latent_spaces) > 0:
                W = lvpp.latent_spaces[0]

        lb, ub = bounds
        if ub is not None:
            raise NotImplementedError(
                "banzmark currently supports lower obstacle only (unilateral)."
            )
        lb_ufl = bounds_ufl[0] if bounds_ufl is not None else None
        phi = lb_ufl if lb_ufl is not None else lb
        if phi is None:
            raise ValueError("Lower obstacle bound must be provided in bounds or bounds_ufl.")

        if kappa is None:
            raise ValueError(
                "kappa (multiplier approximation or latent drift) must be provided."
            )

        V = uh.function_space()
        mesh = V.mesh()

        if C_P is None:
            C_P = self._banz_poincare(mesh)

        consts = self._banz_constants(delta=delta, epsilon=epsilon)
        c_aux = consts["c_aux"]
        c_cons = consts["c_cons"]
        c_comp = consts["c_comp"]

        R, jump2 = self._banz_flux_bound(
            uh, kappa, f_ufl, C_P, tau, mesh, quadrature_degree=quadrature_degree
        )
        N2, grad_q_minus2, comp_term, cross_neg = self._banz_constraint_terms(
            uh, kappa, phi, C_P, mesh, quadrature_degree=quadrature_degree
        )

        aux_part2 = c_aux * (R ** 2)
        constraint_part2 = c_cons * (N2 + grad_q_minus2) + c_comp * comp_term
        eta_up2 = aux_part2 + constraint_part2
        eta_up = math.sqrt(max(eta_up2, 0.0))

        # Full weighted functional Xi at the cut-off pair
        xi = N2 + grad_q_minus2 + epsilon * comp_term - epsilon * cross_neg

        # Branch dominance decision
        aux_safe = max(aux_part2, 1e-300)
        cons_safe = max(constraint_part2, 1e-300)
        if cons_safe >= threshold_dom * aux_safe:
            dominant = "constraint"
        elif aux_safe >= threshold_dom * cons_safe:
            dominant = "auxiliary"
        else:
            dominant = "balanced"

        # Optional health monitors
        rho = (
            self.banz_aux_residual_gram_norm(
                uh, kappa, f_ufl, bcs, quadrature_degree=quadrature_degree
            )
            if bcs is not None
            else None
        )
        d_hp = (
            self.banz_moment_defect(
                uh, phi, s=s, recon=recon, W=W, quadrature_degree=quadrature_degree
            )
            if (s is not None or recon is not None)
            else None
        )

        fields = self._banz_local_indicators(
            uh, kappa, phi, f_ufl, mesh, quadrature_degree=quadrature_degree
        )

        # Mark elements using VIAMR's existing fixedratemark
        mark, ethresh = self.fixedratemark(
            fields["eta"], theta=theta, method=method
        )

        branch_info = {
            "R": R,
            "jump2": jump2,
            "N2": N2,
            "grad_q_minus2": grad_q_minus2,
            "comp_term": comp_term,
            "cross_neg": cross_neg,
            "xi": xi,
            "aux_part2": aux_part2,
            "constraint_part2": constraint_part2,
            "eta_up2": eta_up2,
            "eta_up": eta_up,
            "dominant_branch": dominant,
            "C_P": C_P,
            "c_aux": c_aux,
            "c_cons": c_cons,
            "c_comp": c_comp,
            "rho": rho,
            "d_hp": d_hp,
            "ethresh": ethresh,
            "method": method,
            "theta": theta,
            "tau": tau,
        }
        fields["info"] = branch_info

        return mark, fields, eta_up
