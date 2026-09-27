import numpy as np
from firedrake import *
from firedrake.petsc import PETSc


class NSVMarkingsMixin:
    r"""Mixed into VIAMR (see viamr.py): marking by the pointwise a posteriori
    estimators of Nochetto, Siebert, & Veeser for classical obstacle problems.
    The public method is nsvmark(), which dispatches to _nsv03mark() or
    _nsv05mark().

    NSVMarkingsMixin is not usable separately from VIAMR, as it calls many
    VIAMR methods, e.g. fixedratemark(), nodalactive(), and _obstacleterms().
    """

    def nsvmark(
        self,
        uh,
        bounds,
        g,
        f_ufl,
        g_ufl,
        estimator="nsv03",
        bounds_ufl=(None, None),
        method="max",
        theta=0.5,
        dualtol=1.0e-10,
        fdegree=3,
        C0=None,
        C1=None,
        etadratio=None,
        rhotol=None,
        signtol=None,
    ):
        """For classical obstacle problems, with the Laplacian as the operator,
        compute marking on the entire domain according to a pointwise a posteriori
        estimator of Nochetto, Siebert, & Veeser.  The estimator is one of:

          estimator="nsv03":  the local "practical estimator" of NSV03, extended
          to box constraints bounds = (lb, ub); see _nsv03mark().

          estimator="nsv05":  the fully-localized, star-based estimator of NSV05,
          the successor of NSV03, for a lower obstacle only, i.e. bounds =
          (lb, None); see _nsv05mark().

        The inputs uh, bounds, g, f_ufl, g_ufl, bounds_ufl, method, theta,
        dualtol, and fdegree mean the same for both estimators.  Here g is the
        discrete boundary data, f_ufl the source term, g_ufl the boundary values,
        and bounds_ufl = (lb_ufl, ub_ufl) the continuum obstacles, if known.
        The marking strategy method and threshold theta are passed to
        fixedratemark().

        The remaining inputs are estimator-specific, and None means the default
        of the estimator.  C0 is used by both, with default 0.1 for "nsv03" and
        0.02 for "nsv05".  C1 and etadratio are used only by "nsv03", and rhotol
        and signtol only by "nsv05"; passing one to the other estimator raises
        ValueError.

        Returns (mark, fields, Eh).  Here mark is the DG0 element marking, Eh is
        the scalar estimator, which is the appropriate numerator for an
        effectivity index, and fields is a dict of the estimator's named
        diagnostic fields:
          "nsv03":  {"etainf", "etad", "sigmah"}
          "nsv05":  {"eta", "sz", "fullcontact"}
        See _nsv03mark() and _nsv05mark() for their meaning."""
        if estimator == "nsv03":
            own = {"C1": C1, "etadratio": etadratio}
            other = {"rhotol": rhotol, "signtol": signtol}
            impl = self._nsv03mark
        elif estimator == "nsv05":
            own = {"rhotol": rhotol, "signtol": signtol}
            other = {"C1": C1, "etadratio": etadratio}
            impl = self._nsv05mark
        else:
            raise ValueError(
                f"unknown estimator='{estimator}'; must be 'nsv03' or 'nsv05'"
            )
        for name, val in other.items():
            if val is not None:
                raise ValueError(f"{name} is not used by estimator='{estimator}'")
        # pass only the estimator-specific inputs actually given, so that the
        # others take the defaults in the signature of impl
        given = {k: v for k, v in dict(C0=C0, **own).items() if v is not None}
        return impl(
            uh,
            bounds,
            g,
            f_ufl,
            g_ufl,
            bounds_ufl=bounds_ufl,
            method=method,
            theta=theta,
            dualtol=dualtol,
            fdegree=fdegree,
            **given,
        )

    def _nsv03mark(
        self,
        uh,
        bounds,
        g,
        f_ufl,
        g_ufl,
        bounds_ufl=(None, None),
        method="max",
        theta=0.5,
        dualtol=1.0e-10,
        C0=0.1,
        C1=0.01,
        fdegree=3,
        etadratio=1.0,
    ):
        """For classical obstacle problems, with the Laplacian as the operator, compute marking on entire domain according to the local 'practical estimator' from NSV03:

            Nochetto, R. H., Siebert, K. G., & Veeser, A. (2003). Pointwise
            a posteriori error control for elliptic obstacle problems.
            Numerische Mathematik, 95(1), 163-195.

        With ub given this is the *box-constrained* form of that estimator, the
        one derived in doc/nsv-box/box.tex, which extends NSV03 to the two-sided
        constraint chi_lo <= u <= chi_up.  Both obstacles are then live, and the
        four terms below which refer to an obstacle come in symmetric pairs.  The
        default ub=None is the unilateral problem of NSV03 itself, in which the
        upper obstacle is absent and each pair collapses to its lower half.

        The per-element main formula, NSV03 (7.1) read box-constrained, is
            eta_infty =
                  C_0 h_T^2 ||R_infty||_infty                        [term 1]
                + ||(chi_lo - u_h)^+||_infty
                    + ||(u_h - chi_up)^+||_infty                     [term 2]
                + 1_{omega_bot} ||(u_h - chi_lo)^+||_infty
                    + 1_{omega_top} ||(chi_up - u_h)^+||_infty       [term 3]
                + ||g - I_h g||_{infty; partial Omega cap T}         [term 4]
        But there is a second per-element quantity, the L^d "quadrature indicator" of section 7.1:
            eta_d = C_1 h_T^2 ||grad(sigma_h)||_{d; Lambda_h cap T}    [term eta_d]
        Both eta_.. are computed on each triangle T in the mesh.

        Meaning:
          term 1:  Estimates the residual relevant to the VI problem; see below for the R_infty formula, which uses the discrete residual sigma_h below.  C_0=0.1 is used by NSV03.

          term 2:  The two *obstacle admissibility* terms, each zero unless the caller supplies the corresponding continuum obstacle lb_ufl (lower) or ub_ufl (upper), i.e. the entries of bounds_ufl.  Without it we take the continuum obstacle to be lb (resp. ub) itself, having only the obstacle on the current mesh, and then *asserting primal admissibility* makes the term vanish.  Passing a UFL expression for the continuum obstacle restores the term; see _obstacleterms().  It matters whenever chi is not representable in u_h's space, since then u_h touches chi_h rather than chi and the difference is a genuine part of the pointwise error.  A curved obstacle sampled on a coarse mesh is the case to worry about: there this term can be the whole of ||u - u_h||_{0,inf;Omega}.

          term 3:  The two *localized detachment* terms, which measure how far u_h has detached from an obstacle on the region where the discrete residual asserts that it is attached.  Each is localized to a star-dilated strict contact set, in the sign convention of sigma_h below,

              omega_bot = U_h({sigma_h > 0}),   omega_top = U_h({sigma_h < 0}),

          so an element counts if *some* vertex has a strictly signed multiplier.  NSV03 (7.1) instead uses the bare set {sigma_h < 0} in their opposite sign convention, requiring *every* vertex; the dilation is forced by the box constraint, because on a transition element sigma_h changes sign and the affine argument justifying the bare set fails.  The star dilation is explained in doc/nsv-box/box.tex, which also notes that the dilated set contains the bare one, so this stays reliable unilaterally.  One consequence: unlike the bare set, omega_bot does not force the term to zero when the continuum obstacle is absent, since an element with one strictly active vertex and two inactive ones has a nonzero gap.

          term 4:  Estimates the boundary interpolation error, and we use a formula which is correct if g is in CG4.  Being a sup norm over partial Omega cap T, it is computed as the elementwise sup of |g - I_h g| over the elements touching the boundary.

          term eta_d:  Controls the mass-lumping/quadrature error incurred when computing sigma_h (see sec. 6.3 and 7.1 in NSV03).  sigma_h is CG1, so grad(sigma_h) is elementwise constant, and its L^d(T) norm is |grad(sigma_h)|_T * |T|^{1/d}.  Localized to Lambda_h (the discrete contact set, approximated here by tactive below, the same "neighborhood active" indicator used for term 1's X) because sigma = 0 off the contact set (see (1.2) in NSV03), so grad(sigma_h) there is quadrature noise rather than signal.  Bilaterally Lambda_h is the union of the lower and upper element-wise contact sets.  C_1=0.01 is the practical value used by NSV03 in (7.1).

        Regarding eta_d, NSV03 sec. 7.1 notes that it "exhibits different accumulation" than eta_infty.  That is, as a genuine L^d(Lambda_h) norm, it aggregates over T by an L^d-type sum.  Mixing both into one scalar before marking would let eta_d's different scaling distort the max-based threshold.  Following NSV03, we therefore mark in two separate passes.  First on eta_infty, then on eta_d restricted to Lambda_h, and then take the union.  NSV03 further qualifies that the second pass only runs "provided quadrature dominates the estimator," so the second pass runs only if max(eta_d) > etadratio * max(eta_infty).  NSV03 does not give a precise numerical criterion for "dominates", so etadratio is exposed as a parameter.

        Returns (mark, fields, Eh), where fields = {"etainf", "etad", "sigmah"} holds the DG0 fields eta_infty and eta_d and the CG1 residual sigma_h.  Eh is the scalar estimator Etilde_h of (7.1) itself, so it is the quantity which bounds max(||u - u_h||_{0,inf;Omega}, ||sigma - sigmatilde_h||_{-2,inf;Omega}), and thus the right numerator for an effectivity index.  (Bilaterally, the reliability theorem of doc/nsv-box/box.tex gives the ||u - u_h||_{0,inf;Omega} half of that bound, and its residual estimate gives the other half, up to a constant.)  Each of its terms is accumulated in its own norm: the sup-norm terms are maximized separately, since (7.1) adds the global norms rather than maximizing their elementwise sum eta_infty; and eta_d is accumulated as an L^d-type sum of d-th powers.
        """
        # mesh quantities
        mesh = uh.function_space().mesh()
        CG1, DG0 = self.spaces(mesh)
        n = FacetNormal(mesh)
        hT = project(CellSize(mesh), DG0)  # versus mesh.cell_sizes(), which is in CG1

        # the discrete obstacles, and the continuum ones; either entry of either
        #   pair may be None, and an absent obstacle contributes nothing
        lb, ub = bounds
        lb_ufl, ub_ufl = bounds_ufl
        assert not (lb is None and ub is None), "bounds must constrain a side"

        # term 2
        # primal admissibility check; this is a property of the discrete solution,
        #   asserted against the discrete obstacle whether or not the continuum one
        #   is given.  With lb_ufl=None it is also what makes "(lb - u_h)^+" vanish,
        #   since then the continuum obstacle is lb itself, which is representable
        #   in u_h's space.  Symmetrically on the upper side.
        if lb is None:
            obstacleerrlo = Function(DG0)
            gaplo = Function(DG0)
        else:
            gaph = Function(CG1).interpolate(uh - lb)
            assert self._globalextreme(gaph, minimum=True) >= 0.0
            obstacleerrlo, gaplo = self._obstacleterms(uh, lb, lb_ufl, fdegree)
        if ub is None:
            obstacleerrup = Function(DG0)
            gapup = Function(DG0)
        else:
            gaphup = Function(CG1).interpolate(ub - uh)
            assert self._globalextreme(gaphup, minimum=True) >= 0.0
            obstacleerrup, gapup = self._obstacleterms(
                uh, ub, ub_ufl, fdegree, boxside="upper"
            )

        # compute residual sigmah in CG1 following section 2.1 of NSV03, page 169,
        #   but use opposite sign convention so sigmah >= 0.   complementarity is
        #   uh >= lb,  sigmah >= 0,  (uh - lb) sigmah = 0
        # bilaterally sigmah has no global sign.  The box-constrained theory in
        #   doc/nsv-box/box.tex gives instead a *nodal* complementarity, at an interior node:
        #   sigmah(z) >= 0 if uh(z) = lb(z), sigmah(z) <= 0 if uh(z) = ub(z),
        #   and sigmah(z) = 0 if neither.  So a strictly signed node is active,
        #   and the sign says against which obstacle.
        # step 1: residual as a cofunction; the "- inner(grad(uh), n) * phi * ds" term
        #   is NSV03's boundary flux correction (page 169) -- it vanishes identically
        #   at interior nodes z, since phi_h^z restricted to any boundary facet not
        #   incident to z is zero, so this one assembly is exact for interior dofs and
        #   gives the *uncorrected* boundary-node value handled in step 4 below.
        phi = TestFunction(CG1)
        res = assemble(
            (inner(grad(uh), grad(phi)) - f_ufl * phi) * dx
            - inner(grad(uh), n) * phi * ds
        )  # cofunction
        # step 2: create cofunction with values  s_i = int_Omega phi_i dx  for *all*
        #   nodes i; we *do not* want riesz_representation() here
        scale = assemble(phi * dx)
        # step 3: apply scale, that is, divide by s_i
        sigmah = Function(CG1, name="sigma_h (residual)")
        sigmah.dat.data[:] = res.dat.data_ro / scale.dat.data_ro  # divide numpy arrays
        # step 4: at boundary nodes z there is no perturbation freedom (v_h = I_h g is
        #   an equality constraint there), so sigmah(z) is not derived from stationarity
        #   as at interior nodes; following NSV03 page 169, it is instead defined as
        #   the *positive part* of the boundary-corrected residual above, and only kept
        #   nonzero where z's whole star U_h(z) is active -- otherwise sigmah(z) = 0.
        #   Bilaterally the same device applies to each obstacle, as in
        #   doc/nsv-box/box.tex: keep the positive part
        #   on a wholly lower-active star, the negative part on a wholly
        #   upper-active star, and zero elsewhere.  The two stars are disjoint
        #   because lb < ub, so the two contributions never overlap.
        bdry_ufl = Constant(0.0)
        if lb is not None:
            starlo = self._starwhollyactive(uh, (lb, None))
            bdry_ufl = bdry_ufl + starlo * conditional(sigmah > 0.0, sigmah, 0.0)
        if ub is not None:
            starup = self._starwhollyactive(uh, (None, ub))
            bdry_ufl = bdry_ufl + starup * conditional(sigmah < 0.0, sigmah, 0.0)
        bdryval = Function(CG1).interpolate(bdry_ufl)
        DirichletBC(CG1, bdryval, "on_boundary").apply(sigmah)

        # check dual admissiblity (up to tolerance).  Unilaterally that is the
        #   global sign sigmah >= 0.  Bilaterally it is the nodal complementarity
        #   recalled above, namely that a strictly signed node is active against
        #   the obstacle its sign names.
        if ub is None:
            assert self._globalextreme(sigmah, minimum=True) >= -dualtol
        elif lb is None:
            assert self._globalextreme(sigmah, minimum=False) <= dualtol
        else:
            nodallo = self.nodalactive(uh, (lb, None))
            nodalup = self.nodalactive(uh, (None, ub))
            signmisfit = Function(CG1).interpolate(
                conditional(sigmah > dualtol, 1.0 - nodallo, 0.0)
                + conditional(sigmah < -dualtol, 1.0 - nodalup, 0.0)
            )
            assert self._globalextreme(signmisfit, minimum=False) == 0.0

        # term 1
        # compute the R_\infty part of "practical estimator" in (7.1) in NSV03, from (3.7)
        # using p=\infty and p'=1:
        #    R_\infty = h_T^{-1} \|[[\partial_n u_h]]\|* + X
        # where by (2.3), with sign switch on sigma_h:
        #    X = |f + sigma_h| if element neighborhood of T is active
        #    X = |f|           otherwise
        # and where
        #    \|.\|* = \|.\|_{\infty; \partial T \setminus \partial \Omega},
        #             i.e. infinity norm along interior edges
        #    [[z]] is the jump in z along an edge
        v0 = TestFunction(DG0)
        # The jump must be a *sup over the facets of T of the jump value*, so it is
        # recovered per facet by _facetjump(), which divides by the facet measure,
        # and then maximized over each cell's own facets.
        # _facetjump() gives exterior facets the value zero, which is exactly the
        # "\setminus \partial \Omega" restriction wanted here.
        jumpu = self._elemmaxabs(self._facetjump(uh))
        # Lambda_h, the element-wise contact set; bilaterally it is the union
        # Lambda_h,bot cup Lambda_h^top of doc/nsv-box/box.tex, and each half is
        # one thinelemactive() call
        tlo = (
            Function(DG0)
            if lb is None
            else self.thinelemactive(uh, (lb, None))
        )
        tup = (
            Function(DG0)
            if ub is None
            else self.thinelemactive(uh, (None, ub))
        )
        tactive = Function(DG0).interpolate(max_value(tlo, tup))
        X_ufl = abs(f_ufl + tactive * sigmah)
        # note pages 188-189 in NSV03 regarding use of DG7, to deal with the fact
        # that f_ufl is generally not in CG1:
        #     "For terms involving non-polynomial data, the maximum norm is
        #      approximated by evaluating element point-values at the Lagrange
        #      nodes for 7th order polynomials.""
        # BUT using DG7 this way is really slow because it is so big, so we drop
        # to DG3 by default; DG3.dim() = 10*DG0.dim(), while DG7.dim() ~= 40*DG0.dim()
        # TODO: Firedrake's DG node set on simplices omits the vertices, so a max
        # over DG dofs under-samples a sup norm: interpolating x on the unit
        # triangle gives 0.897 at DG3 and 0.930 at DG4, against 1.0 for CG.  The
        # bias is toward under-estimation, which is the wrong direction for a
        # reliability bound.  CG is not a drop-in replacement here, since f_ufl
        # may be genuinely discontinuous (Example 7.2 builds it from
        # conditional()), so this needs a decision rather than a rename; the same
        # applies to the DGf sampling in _nsv05mark().  Contrast
        # _obstacleterms(), where the sampled quantities are continuous and CG is
        # used for exactly this reason.
        DGf = FunctionSpace(mesh, "DG", fdegree)
        Rinf = Function(DGf).interpolate((jumpu / hT) + X_ufl)
        Rinf = self._elemmaxabs(Rinf)

        # term 3
        # The two localized detachment terms, each measured over a star-dilated
        # strict contact set of doc/nsv-box/box.tex,
        #     omega_bot = U_h({sigma_h > 0}),   omega_top = U_h({sigma_h < 0}).
        # A star dilation of a nodal set is a single node -> element max, so an
        # element lands in omega_bot exactly when *some* vertex has a strictly
        # positive multiplier.  NSV03 (7.1) uses the bare set instead, an
        # element -> min requiring *every* vertex; doc/nsv-box/box.tex explains
        # why the box constraint forces the dilation, and notes that the dilated
        # set contains the bare one, so the unilateral estimator stays reliable.
        # Because of that enlargement the term does not degenerate when
        # continuum obstacle is absent, in contrast to the bare set, where
        # complementarity makes the gap vanish at exactly the nodes tested.  This is the situation of
        # NSV03's Remark 5.8, where this term drives the initial refinement.
        # Compare _nsv05mark(), whose Lambda_h of (2.20) is likewise a union of
        # whole stars.
        nodalstrictlo = Function(CG1).interpolate(
            conditional(sigmah > dualtol, 1.0, 0.0)
        )
        ombot = self._elemextreme(nodalstrictlo, minimum=False, defaultval=0.0)
        blockgaplo = Function(DG0).interpolate(ombot * gaplo)
        nodalstrictup = Function(CG1).interpolate(
            conditional(sigmah < -dualtol, 1.0, 0.0)
        )
        omtop = self._elemextreme(nodalstrictup, minimum=False, defaultval=0.0)
        blockgapup = Function(DG0).interpolate(omtop * gapup)

        # term 4
        # This is a sup norm over \partial \Omega \cap T, so it is computed as the
        # elementwise sup of |g - I_h g| on the elements which touch the boundary.
        # That overestimates the sup over the boundary facets themselves, but it is
        # an upper bound and so keeps the estimator reliable.
        CG4 = FunctionSpace(mesh, "CG", 4)  # CG4.dim() ~ 9*DG0.dim()
        adg = self._elemmaxabs(Function(CG4).interpolate(g_ufl - g))
        touchesbdry = Function(DG0)
        touchesbdry.dat.data[:] = assemble(v0 * ds).dat.data_ro
        bdryerr = Function(DG0).interpolate(
            conditional(touchesbdry > 0.0, 1.0, 0.0) * adg
        )

        # finally compute eta_inf; see doc string above for formula
        residterm = Function(DG0).interpolate(C0 * hT ** 2 * Rinf)
        etainf_ufl = (
            residterm
            + obstacleerrlo
            + obstacleerrup
            + blockgaplo
            + blockgapup
            + bdryerr
        )
        etainf = Function(DG0, name="eta_inf").interpolate(etainf_ufl)

        # first marking pass: eta_infty over the whole domain
        mark, _ = self.fixedratemark(etainf, theta, method)

        # term eta_d
        d = mesh.cell_dimension()
        gradsigmanorm = Function(DG0).interpolate(
            sqrt(inner(grad(sigmah), grad(sigmah)))
        )
        etad_ufl = (
            C1 * hT ** 2 * gradsigmanorm * CellVolume(mesh) ** (1.0 / d) * tactive
        )
        etad = Function(DG0, name="eta_d").interpolate(etad_ufl)

        # second marking pass: eta_d, but only when "quadrature dominates the
        # estimator" (NSV03 section 7.1)
        etainf_max = self._globalextreme(etainf, minimum=False)
        etad_max = self._globalextreme(etad, minimum=False)
        if etad_max > etadratio * etainf_max:
            markd, _ = self.fixedratemark(etad, theta, method)
            mark = self.unionmarks(mark, markd)

        # the estimator Etilde_h of (7.1), accumulating each term in its own norm;
        # see doc string above.  The sup-norm terms are maximized separately,
        # because (7.1) adds the global norms rather than maximizing their
        # elementwise sum eta_inf; and eta_d, being an L^d(Lambda_h) norm, is
        # accumulated as a sum of d-th powers.  The upper-obstacle terms are
        # identically zero when ub is None, so this is the box-constrained
        # estimator of doc/nsv-box/box.tex in both the bilateral and unilateral cases.
        Eh = (
            self._globalextreme(residterm, minimum=False)
            + self._globalextreme(obstacleerrlo, minimum=False)
            + self._globalextreme(obstacleerrup, minimum=False)
            + self._globalextreme(blockgaplo, minimum=False)
            + self._globalextreme(blockgapup, minimum=False)
            + self._globalextreme(bdryerr, minimum=False)
            + self._globalpnorm(etad, d)
        )
        fields = {"etainf": etainf, "etad": etad, "sigmah": sigmah}
        return (mark, fields, Eh)

    def _nsv05mark(
        self,
        uh,
        bounds,
        g,
        f_ufl,
        g_ufl,
        bounds_ufl=(None, None),
        method="max",
        theta=0.5,
        C0=0.02,
        fdegree=3,
        rhotol=1.0e-8,
        signtol=1.0e-10,
        dualtol=1.0e-10,
    ):
        """For classical obstacle problems, with the Laplacian as the operator, compute marking on the entire domain according to the *fully localized* estimator from NSV05:

            Nochetto, R. H., Siebert, K. G., & Veeser, A. (2005). Fully localized
            a posteriori error estimators and barrier sets for contact problems.
            SIAM Journal on Numerical Analysis, 42(5), 2118-2135.

        This is the successor of the NSV03 estimator implemented by _nsv03mark(); see the comparison below.

        The estimator is mostly nodal, and star-based at the nodes, but we must return an element-wise eta field for use in marking, via fixedratemark().  After the theory we address this practical implementation issue.

        ** Definitions **

        Let N_h be the set of all nodes on the current mesh.  Let phi_z denote the hat function at node z in N_h, omega_z = supp(phi_z) its star, and gamma_z = Gamma cap (omega_z)^o the set of facets lying in the *interior* of the star.  For simplices this is exactly the set of interior facets which contain z.  The other facets of the star lie in its boundary, and phi_z vanishes identically on them.

        For chi_h the obstacle (chi_h = lb), and J_h the facet jump of the solution u_h (computed by _facetjump()), define the set of full-contact *nodes* as

            C_h = {z in N_h : u_h = chi_h at z, and f <= 0 in omega_z, and J_h <= 0 on gamma_z}

        The discrete full-contact set is

            Omega_h^0 = (union of the elements all of whose vertices lie in C_h).

        This set is returned by the "fullcontact" DG0 marking from this method.  Let Omega_h^+ = Omega \setminus Omega_h^0 be its complement.  Note Omega_h^+ is open, so a facet lying in the boundary of Omega_h^0 is not in Gamma_h^+.  Thus a facet counts only if *both* its elements are outside the full-contact set.

        The residual indicator below is restricted to

            omega_z^+ = omega_z cap Omega_h^+
            gamma_z^+ = gamma_z cap Omega_h^+

        so it *vanishes identically* on Omega_h^0.  This is the "full localization" which distinguishes NSV05 from NSV03; it is why the mesh can stay coarse inside the contact set.

        Define, see equation (2.5), the *nodal residual*

            s_z = int_Omega f phi_z + int_Gamma J_h phi_z

        This nodal quantity is a tool for generating a set, not the indicator (i.e. eta_z below).  From equation (2.20), let

            Lambda_h = (union of stars omega_z over nodes z with s_z < 0,
                        where z is an interior node or a full-contact boundary node)

        This is a negative-residual set, used to gate inactive (noncontact) elements in the estimator.

        ** Estimator **

        The nodal star-based residual indicator (2.15) is

            eta_z = h_z^2 ||(f - fhat_z) phi_z||_{inf; omega_z^+}   data oscillation (excluding full contact)
                    + h_z ||J_h phi_z||_{inf; gamma_z^+}.           gradient jump (excluding full contact)

        Now the estimator E_h of Theorem 2.7 in NSV05 is

            E_h =   C_* |log h_min|^2 max_{z in N_h} eta_z    localized residual
                  + ||(chi - u_h)^+||_{inf; Omega}            admissibility violation vs continuum obstacle
                  + ||(u_h - chi)^+||_{inf; Lambda_h}         inactive overlap with negative-residual set
                  + ||g - I_h g||_{inf; partial Omega}        boundary data approximation

        ** Practical implementation **

        1. Section 3 of NSV05 replaces the whole prefactor C_* |log h_min|^2 by the single constant 0.02.  We keep the logarithmic factor instead, and treat C0 = 0.02 as a coefficient:

            C_* |log h_min|^2  -->  C0 |log(h_min / diam Omega)|^2

        Here h_min = min_z h_z as in NSV05.  Dividing by the bounding box makes the argument dimensionless, so the estimator is invariant under a rescaling of the geometry.  The reason for keeping the log factor is that a bare 0.02 is not reliable.  In the examples of NSV05 section 3 the solution vanishes in a neighborhood of the boundary, so the localized residual is never the binding term there.  However, example 7.2 of NSV03 (2003 paper), reveals the flaw: it has a large inactive set on which u is smooth and nonzero, and the pointwise error there is ordinary P1 interpolation error, ||u - u_h||_{inf;T} ~ h^2 |D^2 u| / 8.  Meanwhile eta_z ~ h^2 |D^2 u|, since the load enters only through its oscillation and the jump term has to carry the whole residual, so the prefactor on max_z eta_z must be at least about 1/8.  A bare 0.02 is roughly eight times too small.  On quasi-uniform meshes this stays hidden, because the obstacle term ||(u_h - chi)^+||_{inf;Lambda_h} is then the largest term in E_h, but adaptive refinement drives that term down much faster than the residual term, and the estimator then falls below the error.  With the logarithmic factor restored, |log(h_min / diam Omega)|^2 is 20 to 35 on the meshes that example generates, so the effective prefactor is 0.4 to 0.7 and E_h dominates at every level.

        2.  The obstacle chi is the *continuum* one, supplied as the UFL expression lb_ufl, the first entry of bounds_ufl; see _obstacleterms().  Both obstacle terms use it, since the barrier arguments of Propositions 2.5 and 2.6 compare u_h against chi itself.  With lb_ufl=None we assume instead that the a posteriori method only has access to the discrete obstacle on the current mesh, i.e. chi = chi_h = lb.  Then ||(chi - u_h)^+||_inf vanishes, because we *assert primal admissibility* against chi_h, and the blocked gap reduces to u_h - lb.  That default matters most for a curved obstacle, which is not representable in u_h's space: there u_h touches chi_h rather than chi, and the difference can be all of ||u - u_h||_{0,inf;Omega}, leaving the estimator blind to the dominant error.

        3. Note that E_h is a single number: a max over nodes plus global sup norms.  Localizing it to elements for marking purposes is not spelled out in NSV05, so we follow what section 7.1 of NSV03 does for its own estimator.  Define

            eta_T = CC max_{z a vertex of T} eta_z                localized residual on element
                    + ||(chi - u_h)^+||_{inf; T}                  obstacle approximation (zero if lb_ufl is None)
                    + ||(u_h - chi)^+||_{inf; Lambda_h cap T}     inactive overlap with negative residual set
                    + ||g - I_h g||_{inf; partial Omega cap T}.   boundary data approximation

        where

            CC = C0 |log(h_min / diam Omega)|^2

        ** Details **

        1.  Element residual:  By (2.9), only the *oscillation* f - fhat_z of the load enters, not f itself:

            fhat_z = (1/2)(min_{omega_z^+} f + max_{omega_z^+} f)   if rho_z = 0,
                     0                                              otherwise,

        with rho_z = int_{Omega_h^+} f phi_z + int_{Gamma_h^+} J_h phi_z the restriction of the nodal multiplier s_z to Omega_h^+.  By (2.2), the whole star of a node not in C_h lies in Omega_h^+, so rho_z = s_z there.  Combined with discrete complementarity (1.2), rho_z = s_z = 0 at every node where u_h > chi_h strictly, and "perhaps by chance otherwise" (NSV05 p. 2123).  Thus the rho_z = 0 branch applies outside of the contact set.  Since it is an exact equality in the theory but a floating-point comparison here, rho_z is tested against rhotol times the local scale int |f| phi_z.

        2. Weight phi_z on the oscillation:  On the facet term the hat function is handled exactly; see _tracetonodeextreme().  On the element term the bound phi_z <= 1 is used instead, which overestimates and so preserves reliability while giving the clean closed form

            ||f - fhat_z||_{inf; omega_z^+} = (1/2)(max_{omega_z^+} f - min_{omega_z^+} f)   if rho_z = 0,
                                              max_{omega_z^+} |f|                            otherwise,

        i.e. exactly half the oscillation of f over the star, computable from elementwise extremes of f.

        3. The "blocked gap" ||(u_h - chi)^+|| is restricted to the set Lambda_h of (2.20), which is the union of the stars omega_z over nodes z with s_z < 0, where z is an interior node or a full-contact boundary node.  Compare _nsv03mark(), which restricts the same quantity to {sigma_h < 0} elementwise, without dilating to stars.

        4. Boundary datum:  ||g - I_h g||_{inf; partial Omega} is computed with a formula which is correct if g is in CG4.  It is localized here as the elementwise sup over boundary-touching elements, which overestimates the sup over the boundary facets themselves.  (_nsv03mark() instead divides an assembled boundary integral by the cell volume, which does not have the units of a sup norm.)

        ** Marking **

        Under the default 'max' strategy, taking the max over the vertices of T is exactly equivalent to marking the whole star omega_z of every marked node z.  Note there is only *one* marking pass, in contrast to _nsv03mark().

        Returns (mark, fields, Eh), where
          * mark is the DG0 element marking for refinement
          * fields = {"eta", "sz", "fullcontact"} holds
              - eta, the DG0 elementwise estimator
              - sz, the CG1 nodal multiplier of (2.5)
              - fullcontact, the DG0 indicator of Omega_h^0
          * Eh is the scalar estimator E_h of Theorem 2.7 itself.

        Note that Eh is the quantity which bounds ||u - u_h||_{0,inf;Omega}, and thus it is the appropriate numerator for an effectivity index.  Its terms are separate global sup norms, so each is maximized on its own; this makes Eh >= max_T eta_T, with equality only if all the maxima happen to fall on one element.

        ** Differences from _nsv03mark() = NSV03 **

          * the theoretical indicator is star-based (per node z), not element-based
          * the residual is switched off entirely on Omega_h^0
          * the load enters only through its oscillation f - fhat_z
          * there is no quadrature estimator and no second marking pass
          * the blocked gap is restricted to a union of stars, not to elements
        """
        # the discrete obstacles, and the continuum ones.  NSV05's theory is
        #   unilateral, so only a lower obstacle is accepted here; _nsv03mark()
        #   is the box-constrained estimator.
        lb, ub = bounds
        lb_ufl, ub_ufl = bounds_ufl
        assert lb is not None, "estimator='nsv05' requires a lower obstacle"
        if ub is not None or ub_ufl is not None:
            raise NotImplementedError(
                "estimator='nsv05' is unilateral: NSV05's theory has no upper obstacle, "
                "so bounds must be (lb, None).  Use estimator='nsv03' for a box constraint."
            )

        # mesh quantities
        mesh = uh.function_space().mesh()
        CG1, DG0 = self.spaces(mesh)
        hT = project(CellSize(mesh), DG0)  # versus mesh.cell_sizes(), which is in CG1
        phi = TestFunction(CG1)
        v0 = TestFunction(DG0)

        # sample the (generally non-polynomial) load, then get its elementwise
        # extremes; note pages 188-189 in NSV03 regarding the use of DG7, and
        # see _nsv03mark() for why we drop to DG3 by default
        DGf = FunctionSpace(mesh, "DG", fdegree)
        fs = Function(DGf).interpolate(f_ufl)
        fmaxT = self._elemextreme(fs, minimum=False, defaultval=PETSc.NINFINITY)
        fminT = self._elemextreme(fs, minimum=True, defaultval=PETSc.INFINITY)
        fscale = self._globalextreme(
            Function(DGf).interpolate(abs(f_ufl)), minimum=False
        )

        # nodal multiplier s_z of (2.5), and the unrestricted facet jump J_h
        sz = self._nodalmultiplier(uh, f_ufl)
        Jh = self._facetjump(uh)

        # full-contact nodes C_h: nodally in contact, and both sign conditions
        # hold over the whole star.  The sign tests use tolerances relative to
        # the data scale, because in a flat contact region J_h is zero only up
        # to roundoff, and a spurious positive value there would cost us the
        # full localization we are after.
        ftol = signtol * fscale
        jtol = signtol * self._globalextreme(
            Function(DG0).interpolate(abs(self._elemmaxabs(Jh))), minimum=False
        )
        fmaxstar = self._elemtonodeextreme(
            fmaxT, CG1, minimum=False, defaultval=PETSc.NINFINITY
        )
        # max of J_h over gamma_z, which is the set of facets *containing* z; see
        # _tracetonodeextreme().  Exterior facets carry the value zero from
        # _facetjump(), which passes the "<= 0" test and so correctly does not
        # exclude boundary-adjacent nodes.
        Jmaxgamma = self._tracetonodeextreme(
            Jh, CG1, minimum=False, defaultval=PETSc.NINFINITY
        )
        Ch = Function(CG1, name="C_h (full-contact nodes)").interpolate(
            self.nodalactive(uh, bounds)
            * conditional(fmaxstar <= ftol, 1.0, 0.0)
            * conditional(Jmaxgamma <= jtol, 1.0, 0.0)
        )

        # discrete full-contact set Omega_h^0 = union of elements whose vertices
        # are all in C_h, and its complement Omega_h^+
        fullcontact = self._elemextreme(Ch, minimum=True, defaultval=1.0)
        fullcontact.rename("Omega_h^0 (full contact set)")
        fullplus = Function(DG0).interpolate(1.0 - fullcontact)

        # restrictions to Omega_h^+: the facet jump on Gamma_h^+, and the
        # elementwise extremes of f on omega_z^+.  Full-contact elements are
        # given finite sentinel values which can never win the reductions below;
        # hasplus records whether the star meets Omega_h^+ at all.
        Jplus = self._facetjump(uh, mask=fullplus)
        fbig = 1.0 + fscale
        fmaxplus = Function(DG0).interpolate(
            fullplus * fmaxT - (1.0 - fullplus) * fbig
        )
        fminplus = Function(DG0).interpolate(
            fullplus * fminT + (1.0 - fullplus) * fbig
        )
        fmaxplusz = self._elemtonodeextreme(
            fmaxplus, CG1, minimum=False, defaultval=-fbig
        )
        fminplusz = self._elemtonodeextreme(
            fminplus, CG1, minimum=True, defaultval=fbig
        )
        hasplus = self._elemtonodeextreme(fullplus, CG1, minimum=False, defaultval=0.0)

        # rho_z of (2.9), i.e. s_z with both integrals restricted to Omega_h^+;
        # phi_z is continuous, so avg(phi) is its value on the facet
        rho = Function(CG1)
        rho.dat.data[:] = assemble(
            fullplus * f_ufl * phi * dx
            - jump(grad(uh), FacetNormal(mesh))
            * fullplus("+")
            * fullplus("-")
            * avg(phi)
            * dS
        ).dat.data_ro
        rhoscale = Function(CG1)
        rhoscale.dat.data[:] = assemble(abs(f_ufl) * phi * dx).dat.data_ro

        # star-based residual indicator eta_z of (2.15); see doc string for the
        # closed form of the oscillation term
        hz = self._elemtonodeextreme(hT, CG1, minimum=False, defaultval=0.0)
        # the prefactor C0 |log(h_min / diam Omega)|^2 on the localized residual;
        # see the doc string for why the logarithmic factor is kept
        hmin = self._globalextreme(hz, minimum=True)
        Clog = C0 * np.log(hmin / self._domaindiameter(mesh)) ** 2
        osc_ufl = conditional(
            abs(rho) <= rhotol * rhoscale,
            0.5 * (fmaxplusz - fminplusz),
            max_value(abs(fmaxplusz), abs(fminplusz)),
        )
        Jgamma = self._tracetonodeextreme(
            Jplus, CG1, minimum=False, absolute=True, defaultval=0.0
        )
        etaz = Function(CG1, name="eta_z (star residual)").interpolate(
            hasplus * (hz ** 2 * osc_ufl + hz * Jgamma)
        )
        etaR = self._elemextreme(etaz, minimum=False, defaultval=0.0)

        # primal admissibility check; this is a property of the discrete solution,
        #   asserted against the discrete obstacle whether or not lb_ufl is given.
        #   With lb_ufl=None it is also what removes the obstacle-approximation
        #   term from the estimator, since the continuum obstacle is then lb
        #   itself, which is representable in u_h's space.
        gaph = Function(CG1).interpolate(uh - lb)
        assert self._globalextreme(gaph, minimum=True) >= 0.0
        obstacleerr, gap = self._obstacleterms(uh, lb, lb_ufl, fdegree)

        # dual admissibility (up to tolerance): NSV05 gives s_z <= 0 at interior
        # nodes.  The scale for s_z is that of int f phi_z.
        interior = Function(CG1).assign(1.0)
        DirichletBC(CG1, Constant(0.0), "on_boundary").apply(interior)
        szscale = max(
            self._globalextreme(Function(CG1).interpolate(abs(sz)), minimum=False),
            self._globalextreme(rhoscale, minimum=False),
        )
        sztol = dualtol * szscale
        assert (
            self._globalextreme(
                Function(CG1).interpolate(interior * sz), minimum=False
            )
            <= sztol
        )

        # blocked gap, restricted to Lambda_h of (2.20): the union of the stars
        # of the nodes with s_z < 0, taking interior nodes and full-contact
        # boundary nodes
        lamz = Function(CG1).interpolate(
            conditional(sz < -sztol, 1.0, 0.0) * max_value(interior, Ch)
        )
        blockgap = Function(DG0).interpolate(
            self._elemextreme(lamz, minimum=False, defaultval=0.0) * gap
        )

        # boundary datum approximation, as an elementwise sup over the elements
        # which touch the boundary; the CG4 sample is exact if g is in CG4
        CG4 = FunctionSpace(mesh, "CG", 4)  # CG4.dim() ~ 9*DG0.dim()
        touchesbdry = Function(DG0)
        touchesbdry.dat.data[:] = assemble(v0 * ds).dat.data_ro
        bdryerr = Function(DG0).interpolate(
            conditional(touchesbdry > 0.0, 1.0, 0.0)
            * self._elemmaxabs(Function(CG4).interpolate(g_ufl - g))
        )

        # elementwise estimator; see doc string above for the formula
        residterm = Function(DG0).interpolate(Clog * etaR)
        eta = Function(DG0, name="eta (NSV05)").interpolate(
            residterm + obstacleerr + blockgap + bdryerr
        )

        # the estimator E_h of Theorem 2.7, whose three terms are separate global
        # sup norms, so each is maximized on its own rather than maximizing their
        # elementwise sum eta.  This is the quantity the reliability theorem bounds
        # ||u - u_h||_{0,inf;Omega} by, hence the right numerator for an
        # effectivity index; it is >= max_T eta_T.
        Eh = (
            self._globalextreme(residterm, minimum=False)
            + self._globalextreme(obstacleerr, minimum=False)
            + self._globalextreme(blockgap, minimum=False)
            + self._globalextreme(bdryerr, minimum=False)
        )

        mark, _ = self.fixedratemark(eta, theta, method)
        fields = {"eta": eta, "sz": sz, "fullcontact": fullcontact}
        return (mark, fields, Eh)

    def _facetjump(self, uh, mask=None):
        """Compute the jump of the normal derivative of uh across mesh facets,
        returned as a Function in the lowest-order facet ("HDiv Trace" degree 0)
        space, which has exactly one degree of freedom per facet.  The sign
        convention is the one in Nochetto, Siebert, & Veeser (2005), namely

            J_h = [[grad(u_h)]] . n     with n pointing from T^- to T^+,

        which is the *negative* of UFL's jump(grad(uh), n).  (The convention is
        fixed by requiring that the nodal multiplier s_z of NSV05 (2.5) satisfy
        s_z = <f,phi_z> - <grad u_h, grad phi_z>; see _nodalmultiplier().)  The
        sign matters here, in contrast to _nsv03mark(), which only uses |J_h|.

        Facet values are recovered by dividing an assembled facet integral by the
        facet measure; this is exact because the trace space is elementwise
        constant, so its mass matrix is diagonal.  Exterior facets get the value
        zero, which is what _nsv03mark() wants, and also what the NSV05 theory
        wants: its facet set Gamma consists of interior facets only.

        The optional input mask is a DG0 {0,1} indicator.  When given, a facet
        gets a nonzero value only if *both* of its neighboring elements are in
        the mask.  This computes the restriction to Gamma_h^+ in NSV05, noting
        that Omega_h^+ is open, so a facet lying in the boundary of the
        full-contact set Omega_h^0 contributes nothing."""
        mesh = uh.function_space().mesh()
        T0 = FunctionSpace(mesh, "HDiv Trace", 0)
        w0 = TestFunction(T0)
        n = FacetNormal(mesh)
        Jh_ufl = -jump(grad(uh), n)  # the negative of UFL's jump
        if mask is not None:
            Jh_ufl = Jh_ufl * mask("+") * mask("-")
        # integrate over interior facets, as a CoFunction
        jhdual = assemble(Jh_ufl * w0("+") * dS)
        # facet measure on *all* facets, so the division below never divides by zero
        areaS = assemble(w0("+") * dS + w0 * ds)
        Jh = Function(T0, name="J_h (facet jump)")
        Jh.dat.data[:] = jhdual.dat.data_ro / areaS.dat.data_ro
        return Jh

    def _tracetonodeextreme(
        self, tracefield, nodalspace, minimum=False, absolute=False, defaultval=None
    ):
        """Gather a facet ("HDiv Trace" degree 0) field to the nodes of nodalspace
        (e.g. CG1), giving each node z the extreme value of tracefield over the
        facets which *contain* z.  Either computes the maximum or (optionally) the
        minimum, and optionally applies the absolute value; the user must set the
        default value, which initializes the reduction.

        The facets containing z are exactly the set gamma_z of NSV05, that is,
        Gamma cap int(omega_z), the interior facets lying in the interior of the
        star of z.  The remaining facets of the star are those opposite z in some
        element of the star, and they lie in the *boundary* of the star, not its
        interior.  Consistently, phi_z vanishes identically on them, while on the
        facets which do contain z it attains its maximum value of 1 at z itself.
        This is what makes the NSV05 facet term ||J_h phi_z||_{0,inf;gamma_z}
        exactly computable for piecewise linears, as a maximum of |J_h| over the
        facets containing z.

        The implementation relies on the simplex convention, which holds in
        FIAT/FInAT numbering, that local facet i is opposite local vertex i.

        Applies a PyOP2 parallel loop; correct in parallel via Firedrake's own halo
        exchange, with no DMPlex access needed."""
        assert defaultval is not None
        target = Function(nodalspace).assign(defaultval)
        kernel = op2.Kernel(
            """
        void trace_to_node_extreme(double *target, double const *source)
        {
        for (int j = 0; j < %(nnodes)s; j++) {
            double tmp = %(dval)s;
            for (int i = 0; i < %(nfacets)s; i++) {
                /* local facet i is opposite local vertex i, thus not in gamma_z */
                if (i == j) continue;
                double a = %(src)s;
                tmp = tmp %(compare)s a ? tmp : a;
            }
            target[j] = tmp;
        }
        }"""
            % {
                "nnodes": nodalspace.finat_element.space_dimension(),
                "nfacets": tracefield.function_space().finat_element.space_dimension(),
                "dval": float(defaultval),
                "compare": "<" if minimum else ">",
                "src": "fabs(source[i])" if absolute else "source[i]",
            },
            "trace_to_node_extreme",
        )
        op2.par_loop(
            kernel,
            nodalspace.mesh().cell_set,
            target.dat(op2.MIN if minimum else op2.MAX, target.cell_node_map()),
            tracefield.dat(op2.READ, tracefield.cell_node_map()),
        )
        return target

    def _nodalmultiplier(self, uh, f_ufl):
        """Compute the nodal multiplier s_z of NSV05 (2.5), namely

            s_z = int_Omega f phi_z + int_Gamma J_h phi_z,

        returned as a CG1 Function.  Integrating by parts elementwise, and using
        the sign convention of _facetjump(), this equals

            s_z = <f, phi_z> - <grad u_h, grad phi_z>,

        where the second term keeps its boundary flux contribution, so the formula
        is valid at boundary nodes too.  NSV05 shows s_z <= 0 for z in the interior
        nodes union the full-contact nodes C_h.

        Note s_z is *not* scaled by int phi_z, so it is the value of a functional
        rather than a nodal function value.  This is the NSV05 convention, and it
        is the opposite sign from _nsv03mark()'s sigma_h: s_z < 0 there corresponds to
        sigma_h > 0 here."""
        CG1, _ = self.spaces(uh.function_space().mesh())
        phi = TestFunction(CG1)
        n = FacetNormal(CG1.mesh())
        res = assemble(
            (inner(grad(uh), grad(phi)) - f_ufl * phi) * dx
            - inner(grad(uh), n) * phi * ds
        )  # cofunction
        sz = Function(CG1, name="s_z (nodal multiplier)")
        sz.dat.data[:] = -res.dat.data_ro
        return sz

    def _obstacleterms(self, uh, bound, bound_ufl, fdegree, boxside="lower"):
        """Compute the two obstacle-approximation quantities which appear in both
        NSV estimators, as elementwise sup norms in DG0.  For boxside="lower",
        where bound is the discrete lower obstacle,

            obstacleerr = ||(chi - u_h)^+||_{inf; T}
            gap         = ||(u_h - chi)^+||_{inf; T}

        and for boxside="upper", where bound is the discrete upper obstacle, the
        two positive parts are exchanged,

            obstacleerr = ||(u_h - chi)^+||_{inf; T}
            gap         = ||(chi - u_h)^+||_{inf; T}.

        Either way obstacleerr measures how far u_h violates the *continuum*
        obstacle on that side, and gap measures how far u_h has detached from it,
        so the caller uses them for the same two purposes on both sides.

        Here chi is the *continuum* obstacle, which is what NSV03 (7.1) and NSV05
        Theorem 2.7 both ask for.  The barrier arguments behind them (NSV05
        Propositions 2.5 and 2.6) compare u_h against chi itself, so substituting
        the discrete chi_h changes what is being estimated.

        With bound_ufl=None the continuum obstacle is taken to be the discrete
        one, i.e. bound itself.  Then primal admissibility makes obstacleerr
        vanish, and gap reduces to the elementwise sup of the gap to bound.  That
        is exactly what both methods computed before bound_ufl existed, so the
        default reproduces them.

        Otherwise bound_ufl is sampled at the Lagrange nodes of CG_fdegree, in the
        spirit of pages 188-189 of NSV03, which approximate a maximum norm of
        non-polynomial data by element point values at Lagrange nodes.  Both
        quantities are continuous, so CG is available, and it is what we want:
        Firedrake's DG node set on simplices does *not* include the vertices, so
        maximizing over it systematically under-samples a sup norm, whereas CG
        includes them and so at least sees the vertex values exactly.  This also
        makes bound_ufl agree with bound_ufl=None to roundoff whenever the
        continuum obstacle lies in u_h's space, since then both reduce to the
        same vertex maximum.

        Note that u_h >= chi_h does *not* imply u_h >= chi, so here both positive
        parts have to be taken explicitly, rather than relying on admissibility to
        fix a sign."""
        mesh = uh.function_space().mesh()
        _, DG0 = self.spaces(mesh)
        upper = boxside == "upper"
        if bound_ufl is None:
            obstacleerr = Function(DG0).interpolate(Constant(0.0))
            gaph = Function(uh.function_space()).interpolate(
                bound - uh if upper else uh - bound
            )
            return obstacleerr, self._elemmaxabs(gaph)
        CGc = FunctionSpace(mesh, "CG", fdegree)
        # violation of the continuum obstacle, signed so that its positive part
        # is obstacleerr and the positive part of its negative is gap
        viol_ufl = (uh - bound_ufl) if upper else (bound_ufl - uh)
        obstacleerr = self._elemmaxabs(
            Function(CGc).interpolate(max_value(viol_ufl, 0.0))
        )
        gap = self._elemmaxabs(
            Function(CGc).interpolate(max_value(-viol_ufl, 0.0))
        )
        return obstacleerr, gap

    def _domaindiameter(self, mesh):
        """Compute the diameter of the mesh's bounding box, collectively from the
        coordinate field.  This is a cheap stand-in for diam(Omega), used to
        nondimensionalize a mesh size before taking its logarithm; see
        _nsv05mark().  Correct in parallel, including when a process owns no local
        coordinate dofs."""
        xyz = mesh.coordinates.dat.data_ro
        if xyz.ndim == 1:
            xyz = xyz.reshape((-1, 1))
        comm = mesh.comm
        diamsqr = 0.0
        for j in range(xyz.shape[1]):
            empty = xyz.shape[0] == 0
            lo = comm.allreduce(
                PETSc.INFINITY if empty else float(xyz[:, j].min()), op=MPI.MIN
            )
            hi = comm.allreduce(
                PETSc.NINFINITY if empty else float(xyz[:, j].max()), op=MPI.MAX
            )
            diamsqr += (hi - lo) ** 2
        return diamsqr ** 0.5
