import numpy as np
from pyop2.mpi import MPI
from firedrake import *
from firedrake.petsc import PETSc
from firedrake.utils import IntType
import firedrake.cython.dmcommon as dmcommon

try:
    from petsctools import OptionsManager
except ImportError:
    from firedrake.petsc import OptionsManager

from .ama import AMAMixin, haveanimate
from .diagnostics import SetDiagnosticsMixin
from .nsv import NSVMarkingsMixin


class VIAMR(OptionsManager, AMAMixin, NSVMarkingsMixin, SetDiagnosticsMixin):
    r"""A VIAMR object computes a posteriori estimators for a Firedrake variational inequality (VI) solver, where the VI constraint set is defined by box bounds (lb <= u <= ub), and it manages adaptive mesh refinement (AMR) from such estimators.

    Central notions behind this class:
      * The methods generate DG0 element markings, to be passed to tag-and-refine mesh refinement methods.
      * We implement certain rigorous a posteriori estimators for the classical obstacle problem---see nsvmark() below.
      * Mathematical theory often does not provide existing, rigorous estimators for realistic problems.
      * Some methods in the library are partly-heuristic, based on a PDE-type a posteriori estimator in the computed inactive set, with additional refinement in the vicinity of the free boundary.
      * Refinement near the computed free boundary is compatible with the goals of certain VI models.
      * For some problems, refinement in the active set is worthwhile, but for some problems it is wasted effort.
      * We including geometrical set measures, specifically hausdorff2D() and jaccard(), so that users can assess their solutions using more than Sobolev norms.
      * One method, buildaveragedmetric(), generates a metric to be passed to the animate mesh adaptation library.

    The public mark-and-refine API of the VIAMR class consists of:

      fbmark():  marking method targeting refinement of the computed free boundaries, either by a purely-discrete unstructured-dilation operation (algorithm="udo") or by diffusing the computed active set indicator using mesh size in a variable coefficient (algorithm="vcd")

      inactivemark():  classical (PDE) a posteriori error estimator, applied in the computed inactive set, implementing either the method from Babushka & Rheinboldt (1978) (estimator="br78"), its weighted extension from Bernardi & Verfurth (2000) (estimator="bv00"), or CG1 recovery of the DG0 gradient (estimator="gradientrecovery")

      nsvmark():  mark using a pointwise a posteriori estimator for classical obstacle problems, either the "practical estimator" from Nochetto, Siebert, & Veeser (2003) = NSV03, extended to box constraints (estimator="nsv03"), or the fully-localized, star-based estimator from Nochetto, Siebert, & Veeser (2005) = NSV05, the successor of NSV03, implemented for unilateral obstacles only (estimator="nsv05")

      fixedratemark():  general-purpose thresholding of an elementwise DG0 estimator field by a fixed-rate ('max' or 'total'/bulk/Doerfler) criterion; used internally by inactivemark() and nsvmark(), but also usable directly

      unionmark():  a method for combining existing marks

      nodalactive():  nodal marking of the computed active set

      elemactive(), thinelemactive():  two versions of element marking of computed active sets

      eleminactive():  element marking of the computed inactive set

      lowerboundcelldiameter():  unmark elements with cell diameters below a minimum

      refinesbr2D():  a method which calls PETSc for skeleton-based-refinement (SBR)

    There are also diagnostic and measurement methods:

      freeboundarygraph2D():  for 2D obstacle problems, return the computed free boundary, an edge set

      hausdorff2D():  compute Hausdorff distance between edge sets E1, E2 in planar (2D) mesh

      jaccard():  compute Jaccard similarity index between two sets, e.g. active sets, each given by a DG0 indicator function or a UFL expression

    Some default calls to the most significant mark-and-refine methods are:

    .. code-block:: python3

      amr = VIAMR()

      fbmark, _, _ = amr.fbmark(uh, (lb, ub), ..)       # free-boundary-targeted marking
          algorithm="udo"                               # unstructured dilation operator
          algorithm="vcd"                               # same, but based on diffusion

      imark, _, _ = amr.inactivemark(uh, (lb, ub), ..)  # mark using classical PDE
                                                        #estimator in inactive set
          estimator="br78", res=res_ufl                 # BR78 estimator
          estimator="bv00", res=res_ufl, alpha=alpha    # weighted BV00 estimator
          estimator="gradientrecovery"                  # gradient recovery estimator

      mark, ethresh = amr.fixedratemark(eta, ..)        # threshold a DG0 estimator eta

      mark = amr.unionmarks(fbmark, imark)              # mark elements both ways

      mark, _, _ = amr.nsvmark(uh, (lb, ub), g, ..)     # estimators designed for the classical obstacle problem
          estimator="nsv03"                             # NSV03, extended to box constraints
          estimator="nsv05"                             # NSV05, unilateral; ub=None is required

      rmesh = amr.refinesbr2D(mesh, mark)               # PETSc's skeleton-based refinement

    Regarding the arguments (see individual methods for details): uh is a computed VI solution, lb is a lower-bound obstacle, ub is an upper-bound obstacle, res_ufl is a UFL expression for the residual (applicable in the inactive set), and alpha is a weighting field (see examples).

    Regarding returned values (see individual methods for details): fbmark, imark, and mark are element markings in DG0, i.e. indicator functions which are nonzero exactly on the marked elements, and rmesh is a refined mesh.

    Note that fbmark() marks the free boundary of each bound separately, and unions the two markings.  The separate markings are also returned; see fbmark().

    There are also some utility methods, including: spaces(), meshsizes(), meshreport(), scalarrange(), checkadmissible(), and countmark().  Other methods starting with an underscore are (roughly) intended to be private to the VIAMR class.

    The mesh adaptation API needs the animate library:

    .. code-block:: python3

      import animate
      metric = amr.buildaveragedmetric(mesh, uh, (lb, ub))      # VIAMR builds the metric ...
      amesh = animate.adapt(mesh, metric)                       # ... caller adapts the mesh with it

    Source layout:
      * buildaveragedmetric() is in AMAMixin (ama.py)
      * jaccard(), hausdorff2D(), and freeboundarygraph2D() are in SetDiagnosticsMixin (diagnostics.py),
      * nsvmark() is in NSVMarkingsMixin (nsv.py)

    Known limitations:
      * Functions which do not work in parallel: 1. jaccard(..., submesh=False) with two DG0 Function arguments.
      * Functions whose results depend on number of processes: 1. fbmark(algorithm="vcd"), 2. buildaveragedmetric() (via the same diffusion solve).
      * Functions which only work for 2D meshs: 1. freeboundarygraph2D(), 2. hausdorff2D(), 3. refinesbr2D()

    Regarding the last limitation, see the doc string of refinesbr2D(), and compare to refine_marked_elements() from NetGen/ngspetsc.  That ngspetsc method can be applied to DG0 markings from the current library; see the examples.
    """

    PARALLEL_OVERLAP = {
        "partition": True,
        "overlap_type": (DistributedMeshOverlapType.VERTEX, 1),
    }
    """distribution_parameters value needed for freeboundarygraph2D() to give
    correct results in parallel; see _checkparalleloverlap().  Usage:
      mesh = RectangleMesh(m, m, Lx, Ly, distribution_parameters=VIAMR.PARALLEL_OVERLAP)
    """

    def __init__(self, **kwargs):
        self.activetol = kwargs.pop("activetol", 1.0e-10)
        self.debug = kwargs.pop("debug", False)  # extra checks with debug=True
        self.metricparameters = None
        super().__init__({})

    def spaces(self, mesh, k=1):
        """Return CG_k and DG_k-1 spaces."""
        if self.debug:
            assert isinstance(k, int)
            assert k >= 1
        return FunctionSpace(mesh, "CG", k), FunctionSpace(mesh, "DG", k - 1)

    def _globalextreme(self, w, minimum=True):
        """Compute the collective (allreduce) extreme value of a generic scalar
        field's local dof values.  Either computes the minimum or (by default) the
        maximum.  Correct in parallel, including when a process owns no local dofs."""
        data = w.dat.data_ro
        if minimum:
            local = data.min() if len(data) > 0 else PETSc.INFINITY
            op = MPI.MIN
        else:
            local = data.max() if len(data) > 0 else PETSc.NINFINITY
            op = MPI.MAX
        return w.function_space().mesh().comm.allreduce(local, op=op)

    def _globalpnorm(self, w, p):
        """Compute the collective (allreduce) l^p norm of a generic scalar field's
        local dof values, that is, (sum_i |w_i|^p)^(1/p).  Correct in parallel,
        including when a process owns no local dofs, because the Vec holds only
        owned dofs.  Used for estimator terms whose global value accumulates as a
        sum over elements rather than as a maximum; compare _globalextreme()."""
        with w.dat.vec_ro as w_:
            local = np.sum(np.abs(np.asarray(w_.array_r)) ** p)
        total = w.function_space().mesh().comm.allreduce(local, op=MPI.SUM)
        return float(total) ** (1.0 / p)

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

    def meshsizes(self, mesh):
        """Compute number of vertices, number of elements, and range of
        mesh diameters."""
        CG1, DG0 = self.spaces(mesh, k=1)
        nvertices = CG1.dim()
        nelements = DG0.dim()
        hmin, hmax = self.scalarrange(mesh.cell_sizes)
        return nvertices, nelements, hmin, hmax

    def meshreport(self, mesh, indent=2):
        """Print standard mesh report."""
        nv, ne, hmin, hmax = self.meshsizes(mesh)
        indentstr = indent * " "
        PETSc.Sys.Print(
            f"{indentstr}current mesh: {nv} vertices, {ne} elements, h in [{hmin:.5f},{hmax:.5f}]"
        )
        return None

    def scalarrange(self, w):
        """Utility function to return the range of a generic scalar field.  Correct in parallel."""
        return self._globalextreme(w, minimum=True), self._globalextreme(
            w, minimum=False
        )

    def checkadmissible(self, uh, bounds, strict=False):
        """Utility function to check admissibility or strict admissibility of uh
        with respect to the given bounds = (lb, ub).  Either entry may be None,
        for a problem constrained on one side only; see nodalactive().  Returns
        True if lb <= uh <= ub.  The default test is at the nodes of uh's
        function space, while strict=True tests at the quadrature points.

        In debug mode, first checks that uh is a Function and that each
        non-None entry of bounds is a Function or Constant."""
        if self.debug:
            assert isinstance(uh, Function), "input uh must be of class Function"
            for bound in bounds:
                if bound is not None:
                    isbound = isinstance(bound, Function) or isinstance(bound, Constant)
                    assert isbound, "input bound must be of class Function or Constant"
        gap = self._boundsgap(uh, bounds)
        if strict:
            bad = assemble(conditional(gap < 0.0, 1.0, 0.0) * dx)
            return bad == 0.0
        else:
            delta = Function(uh.function_space()).interpolate(gap)
            return self._globalextreme(delta, minimum=True) >= 0.0

    def _boundsgap(self, uh, bounds, absolute=False):
        """Return UFL for the gap from uh to the nearer of the given bounds =
        (lb, ub), namely min(uh - lb, ub - uh), where a None entry drops its
        term.  With absolute=True the terms are |uh - lb| and |uh - ub|, which
        agree with the signed ones for admissible uh."""
        lb, ub = bounds
        assert not (lb is None and ub is None), "bounds must constrain a side"
        terms = []
        if lb is not None:
            terms.append(abs(uh - lb) if absolute else uh - lb)
        if ub is not None:
            terms.append(abs(uh - ub) if absolute else ub - uh)
        return terms[0] if len(terms) == 1 else min_value(*terms)

    def _checkparalleloverlap(self, mesh):
        """Raise ValueError if mesh is distributed across multiple processes
        without sufficient vertex overlap.  freeboundarygraph2D() walks DMPlex
        vertex stars across partition boundaries (via getTransitiveClosure()),
        which requires overlap_type=(DistributedMeshOverlapType.VERTEX, n) with
        n >= 1 for correct results.  Build the mesh with
        distribution_parameters=VIAMR.PARALLEL_OVERLAP to satisfy this."""
        if mesh.comm.size > 1:
            dp = mesh._distribution_parameters
            if dp["overlap_type"][0].name != "VERTEX" or dp["overlap_type"][1] < 1:
                raise ValueError(
                    "freeboundarygraph2D() in parallel requires mesh "
                    "distribution_parameters=VIAMR.PARALLEL_OVERLAP "
                    "on mesh initialization (or overlap_type=(VERTEX, n>=1))"
                )

    def nodalactive(self, uh, bounds):
        """Compute nodal active set indicator in same function space as uh, for an
        obstacle problem with the given bounds = (lb, ub), meaning lb <= uh <= ub.
        Either entry may be None, for a problem constrained on one side only, so
        (lb, None) is a lower obstacle alone and (None, ub) an upper one alone.
        A caller therefore never has to build an artificial infinite obstacle,
        which is what a PETSc VI solve does require.

        The nodal active set is the union of the two one-sided active sets,
          {x in N(V): |u(x) - lb(x)| < activetol or |u(x) - ub(x)| < activetol}
        where N(V) is the nodal set for V = uh.function_space().  Active nodes get value 1.0."""
        if self.debug:
            assert self.checkadmissible(uh, bounds)
        z = Function(uh.function_space(), name="Nodal Active")
        gap = self._boundsgap(uh, bounds, absolute=True)
        z.interpolate(conditional(gap < self.activetol, 1.0, 0.0))
        return z

    def elemactive(self, uh, bounds):
        """Compute an element active set indicator in DG0, for an obstacle
        problem with the given bounds = (lb, ub); see nodalactive().  Active
        elements get value 1.0.  Elements are marked active if the DG0 degree
        of freedom for that element is active against either bound, within
        activetol, so use with caution if z is not in CG1."""
        if self.debug:
            assert self.checkadmissible(uh, bounds)
        _, DG0 = self.spaces(uh.function_space().mesh())
        z = Function(DG0, name="Element Active")
        gap = self._boundsgap(uh, bounds, absolute=True)
        z.interpolate(conditional(gap < self.activetol, 1.0, 0.0))
        return z

    def eleminactive(self, uh, bounds, strong=False):
        """Compute an element inactive set indicator in DG0, for an obstacle
        problem with the given bounds = (lb, ub); see nodalactive().  Inactive
        elements get value 1.0.  By default, elements are marked inactive if
        their DG0 degree of freedom is inactive (by activetol).

        If strong=True then an element is only marked as inactive if all
        degrees of freedom of the gap function min(uh - lb, ub - uh) exceed
        activetol.  That is, a cell is "strongly" inactive if all of its
        original dofs are inactive.

        When both entries of bounds are given the inactive set is the
        *intersection* of the two one-sided inactive sets, i.e. the elements
        touching neither obstacle.  This is the set to which marking methods
        such as inactivemark() restrict an estimator.  One side alone will
        not do: the set inactive with respect to lb still contains the whole of
        ub's contact set, where uh is pinned to the other obstacle and the
        residual is large by construction, so an estimator restricted there
        would mark inside a contact set rather than outside both."""
        if self.debug:
            assert self.checkadmissible(uh, bounds)
        if strong:
            # note gap > 0 is equivalent to strictly inactive ... but we use activetol
            v = Function(uh.function_space()).interpolate(self._boundsgap(uh, bounds))
            # z is in DG0 and contains min of v over each cell's dofs
            z = self._elemextreme(v, minimum=True, defaultval=PETSc.INFINITY)
            z.interpolate(conditional(z > self.activetol, 1.0, 0.0))
            z.rename("Element Inactive")
        else:
            _, DG0 = self.spaces(uh.function_space().mesh())
            z = Function(DG0, name="Element Inactive")
            gap = self._boundsgap(uh, bounds, absolute=True)
            z.interpolate(conditional(gap < self.activetol, 0.0, 1.0))
        return z

    def thinelemactive(self, uh, bounds):
        """Compute element active set indicator into DG0, but "thinned", for an
        obstacle problem with the given bounds = (lb, ub); see nodalactive().

        In contrast to elemactive(), here a cell is marked as active only if it *and its neighboring cells* are active.  The test for active is based on testing at the DG0 degree of freedom, and according to activetol.  Returns a DG0 element-wise indicator, with thinned-active elements having value 1.

        The implementation is inspired by the UDO algorithm in VIAMR.fbmark():
        a thinned-active element is exactly one which is *not* within one ring of
        an inactive element, i.e. z = 1 - dilate1(inactive), computed via the same
        two constant-arity PyOP2 kernels _udofbmark() uses (cell->node max-scatter,
        then node->cell max-gather),
        with no DMPlex access.
        """
        mesh = uh.function_space().mesh()
        CG1, DG0 = self.spaces(mesh)
        inactive = self.eleminactive(uh, bounds)
        grown = self._elemextreme(
            self._elemtonodeextreme(inactive, CG1, minimum=False, defaultval=0.0),
            minimum=False,
            defaultval=0.0
        )
        return Function(DG0, name="Thin Element Active").interpolate(1.0 - grown)

    def _starwhollyactive(self, uh, bounds):
        """Compute a CG1 indicator of those nodes z whose entire star U_h(z) is
        active against the given bounds = (lb, ub); see nodalactive().  Such
        nodes get value 1.0.  That is, for bounds = (lb, None) this is the
        indicator of
            {z in N_h : U_h(z) subset {u_h = lb}},
        which is the condition attached to boundary nodes by the definition of
        the discrete residual sigma_h in section 2.1 of NSV03.

        The computation dilates the *inactive* nodal set by one element ring
        (node -> element max, then element -> node max) and negates, which is
        the same one-ring erosion that thinelemactive() does elementwise.  Thus
        thinelemactive() and this method are the element- and node-based forms
        of the same "whole star is active" test."""
        CG1, _ = self.spaces(uh.function_space().mesh())
        nodalinactive = Function(CG1).interpolate(
            1.0 - self.nodalactive(uh, bounds)
        )
        elemtouchesinactive = self._elemextreme(
            nodalinactive, minimum=False, defaultval=0.0
        )
        nodetouchesinactive = self._elemtonodeextreme(
            elemtouchesinactive, CG1, minimum=False, defaultval=0.0
        )
        return Function(CG1).interpolate(1.0 - nodetouchesinactive)

    def _elemborder(self, nodalactive):
        """From *nodal* active set indicator, computes bordering element indicator.  Uses the fact that the DG0 degree of freedom is strictly inside the element, so use with caution if z is not in CG1.  Returns 1.0 for elements with
          0 < nu_h(x_K) < 1
        for nodal active set indicator nu_h (in CG1), where x_K is the DG0 dof for element K.
        Actually used a tolerance: 0 + tol <= nu_h(x_K) <= 1 - tol.  (This tolerance is unitless,
        whereas self.activetol may be set by user according to units.)
        """
        if self.debug:
            if len(nodalactive.dat.data_ro) > 0:
                assert min(nodalactive.dat.data_ro) >= 0.0
                assert max(nodalactive.dat.data_ro) <= 1.0
        _, DG0 = self.spaces(nodalactive.function_space().mesh())
        z = Function(DG0, name="Element Border")
        bordertol = 1.0e-12
        z.interpolate(
            conditional(
                nodalactive >= 0.0 + bordertol,
                conditional(nodalactive <= 1.0 - bordertol, 1.0, 0.0),
                0.0,
            )
        )
        return z

    def _elemextreme(self, source, minimum=False, absolute=False, defaultval=None):
        """Compute element-wise extreme value of the source function, returning a DG0 field.  Either computes maximum or (optionally) minimum.  Optionally applies the absolute value.  User must set the default value.  Applies a PyOP2 parallel loop.  This should work in parallel for any nodal basis space, e.g. CG_k or DG_k for any k.  Note that this is *not* a reduction, which can be handled more simply, e.g. as in VIAMR.meshsizes()."""
        assert defaultval is not None
        V = source.function_space()
        DG0 = FunctionSpace(V.mesh(), "DG", 0)
        target = Function(DG0).assign(defaultval)
        kernel = op2.Kernel(
            """
        void elem_extreme(double *target, double const *source)
        {
        /* Evaluate extreme value over cell */
        double tmp = %(dval)s;
        for (int i = 0; i < %(ndofs)s; i++) {
            tmp = tmp %(compare)s %(src)s ? tmp : %(src)s;
        }

        /* Set as DG0 dof */
        target[0] = tmp;
        }"""
            % {
                "dval": float(defaultval),
                "ndofs": V.finat_element.space_dimension(),
                "compare": "<" if minimum else ">",
                "src": "fabs(source[i])" if absolute else "source[i]",
            },
            "elem_extreme",
        )
        op2.par_loop(
            kernel,
            V.mesh().cell_set,
            target.dat(op2.MIN if minimum else op2.MAX, target.cell_node_map()),
            source.dat(op2.READ, source.cell_node_map()),
        )
        return target

    def _elemmaxabs(self, source):
        return self._elemextreme(source, minimum=False, absolute=True, defaultval=0.0)

    def _elemtonodeextreme(self, elemfield, nodalspace, minimum=False, defaultval=None):
        """Scatter a DG0 element field into nodalspace (e.g. CG1), broadcasting each cell's value to all of its local nodes and taking the extreme value where multiple cells share a node.  Thus the node value is the extreme over the star U_h(z) of the element values.  Either computes the maximum or (optionally) the minimum; the user must set the default value, which initializes the reduction.  Applies a PyOP2 parallel loop; correct in parallel via Firedrake's own halo exchange, with no DMPlex access needed."""
        assert defaultval is not None
        target = Function(nodalspace).assign(defaultval)
        kernel = op2.Kernel(
            """
        void elem_to_node_extreme(double *target, double const *source)
        {
        for (int i = 0; i < %(ndofs)s; i++) {
            target[i] = source[0];
        }
        }"""
            % {"ndofs": nodalspace.finat_element.space_dimension()},
            "elem_to_node_extreme",
        )
        op2.par_loop(
            kernel,
            nodalspace.mesh().cell_set,
            target.dat(op2.MIN if minimum else op2.MAX, target.cell_node_map()),
            elemfield.dat(op2.READ, elemfield.cell_node_map()),
        )
        return target

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

    def countmark(self, mark):
        """Return count of number of elements marked."""
        mesh = mark.function_space().mesh()
        if self.debug:
            _, DG0 = self.spaces(mesh)
            assert mark.function_space().ufl_element() == DG0.ufl_element()
        j = np.count_nonzero(mark.dat.data_ro)
        return int(mesh.comm.allreduce(j, op=MPI.SUM))

    def unionmarks(self, mark1, mark2):
        """Computes the mark which is 1.0 where either mark1==1.0
        or mark2==1.0.  That is, computes the indicator set of the union."""
        if self.debug:
            _, DG0 = self.spaces(mark1.function_space().mesh())
            assert mark1.function_space().ufl_element() == DG0.ufl_element()
            assert mark2.function_space().ufl_element() == DG0.ufl_element()
        return Function(mark1.function_space(), name="mark (unionmarks)").interpolate(
            (mark1 + mark2) - (mark1 * mark2)
        )

    def lowerboundcelldiameter(self, mark, hmin):
        """For a DG0 cell marking mark, return a new DG0 marking with small elements unmarked, where "small" is CellDiameter() < hmin."""
        mesh = mark.function_space().mesh()
        _, DG0 = self.spaces(mesh)
        if self.debug:
            assert mark.function_space().ufl_element() == DG0.ufl_element()
        large = Function(DG0).interpolate(
            conditional(CellDiameter(mesh) >= hmin, 1.0, 0.0)
        )
        return Function(DG0).interpolate(mark * large)

    def _udofbmark(self, uh, bounds, n=1, restrict=None):
        """One-sided Unstructured Dilation Operator (UDO) marking of the vicinity
        of the free boundary; see fbmark() for the algorithm and the meaning of
        n and restrict.  Here bounds is one-sided, i.e. (lb, None) or (None, ub)."""

        # get mesh and border mark; added flag for restriction
        if restrict is not None:
            meshInit = uh.function_space().mesh()
            if restrict == "active":
                # restrict to active set plus border
                indicator = Function(FunctionSpace(meshInit, "DG", 0)).interpolate(
                    self.elemactive(uh, bounds)
                    + self._elemborder(self.nodalactive(uh, bounds))
                )
            elif restrict == "inactive":
                # restrict to inactive set, which contains border already
                indicator = self.eleminactive(uh, bounds)
            else:
                raise ValueError(
                    f"unknown restrict='{restrict}'; must be 'active', 'inactive', or None"
                )
            mesh = self._filtermesh(meshInit, indicator)
            CG1, DG0 = self.spaces(mesh)
            # Use nodal active set indicator to make an initial DG0 element border
            # indicator. This is now on a restricted domain so allow_missing_dofs=True
            border = Function(DG0).interpolate(
                self._elemborder(self.nodalactive(uh, bounds)),
                allow_missing_dofs=True,
            )
        else:
            mesh = uh.function_space().mesh()
            CG1, DG0 = self.spaces(mesh)
            # Use nodal active set indicator to make an initial DG0 element border
            # indicator.
            border = self._elemborder(self.nodalactive(uh, bounds))

        # main loop: expand element border out to n levels, via two constant-arity
        # PyOP2 kernels per level (cell->node max-scatter, then node->cell max-gather).
        # No DMPlex access, and thus no special mesh overlap requirement.
        for _ in range(n):
            border = self._elemextreme(
                self._elemtonodeextreme(border, CG1, minimum=False, defaultval=0.0),
                minimum=False,
                defaultval=0.0
            )

        return Function(DG0, name="mark (fbmark)").interpolate(
            border, allow_missing_dofs=True
        )

    def _vcdsmooth(self, uh, bounds, directsolver=False, solveriters=4):
        """Diffuse the nodal active set indicator, as the first stage of the
        Variable Coefficient Diffusion (VCD) algorithm; see fbmark().  Returns
        the smoothed indicator as a CG1 Function.  Also used by AMAMixin to build
        an isotropic free-boundary metric."""

        # Compute nodal active set indicator
        mesh = uh.function_space().mesh()
        CG1, _ = self.spaces(mesh)
        nu = self.nodalactive(uh, bounds)

        # Diffuse according to square of cell diameter, with diffusivity D = (1/2) h^2.
        # The nodal active indicator gives the initial field u0.  Solve one backward
        # Euler time-step using a linear solver.
        w = TrialFunction(CG1)
        v = TestFunction(CG1)
        h = CellDiameter(mesh)
        a = w * v * dx + 0.5 * h ** 2 * inner(grad(w), grad(v)) * dx
        L = nu * v * dx
        u = Function(CG1, name="Smoothed Nodal Active")

        if directsolver:
            sp = {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
        else:
            # optimal, approximate solver for linear problem
            # WARNING: can produce different results according to number of
            #          processes, because of ASM+ICC preconditioning
            sp = {
                "ksp_type": "cg",
                "ksp_max_it": solveriters,
                "ksp_convergence_test": "skip",
                "pc_type": "icc",
            }
            if mesh.comm.size > 1:
                sp.update({"pc_type": "asm", "pc_asm_overlap": 1, "sub_pc_type": "icc"})
        solve(a == L, u, solver_parameters=sp, options_prefix="viamr_vcd")
        return u

    def _vcdfbmark(
        self,
        uh,
        bounds,
        bracket=[0.2, 0.8],
        directsolver=False,
        solveriters=4,
    ):
        """One-sided Variable Coefficient Diffusion (VCD) marking of the vicinity
        of the free boundary; see fbmark() for the algorithm and the meaning of
        bracket, directsolver, and solveriters.  Here bounds is one-sided, i.e.
        (lb, None) or (None, ub)."""
        _, DG0 = self.spaces(uh.function_space().mesh())
        u = self._vcdsmooth(
            uh, bounds, directsolver=directsolver, solveriters=solveriters
        )
        # threshold the smoothed indicator, and interpolate into DG0
        middleUFL = conditional(u > bracket[0], conditional(u < bracket[1], 1, 0), 0)
        return Function(DG0, name="mark (fbmark)").interpolate(middleUFL)

    def fbmark(
        self,
        uh,
        bounds,
        algorithm="udo",
        udo_n=None,
        udo_restrict=None,
        vcd_bracket=None,
        vcd_directsolver=None,
        vcd_solveriters=None,
    ):
        """Mark the vicinity of the computed free boundary, for an obstacle
        problem with the given bounds = (lb, ub), meaning lb <= uh <= ub.  Either
        entry may be None for a problem constrained on one side only, so
        (lb, None) is a lower obstacle alone; see e.g. nodalactive().

        Each bound has its own free boundary, and it is marked separately.  We
        return the triple
            (mark, marklower, markupper)
        where mark is the union of the two markings, and where marklower or
        markupper is None if the corresponding bound is None.  Usually only
        mark is wanted:
            fbmark, _, _ = amr.fbmark(uh, (lb, ub))

        The algorithm is one of:

          algorithm="udo":  The Unstructured Dilation Operator (UDO) algorithm.
          It first computes an element-wise indicator for the free boundary.
          Then the elements which neighbor free-boundary elements are added, and
          so on iteratively through udo_n levels; note that udo_n=0 already marks
          the free boundary.  The default is udo_n=1.  Optionally the marking can
          be restricted to the active side of the initially-marked elements
          (udo_restrict="active"), or to the inactive side (="inactive").

          algorithm="vcd":  The Variable Coefficient Diffusion (VCD) algorithm.
          It computes a nodal active set indicator and then diffuses it, using a
          variable coefficient based on mesh geometry.  Diffusion is by solving a
          single backward Euler time step for the corresponding time-dependent
          diffusion equation.  The linear equations are solved by a fixed number
          of iterations of ICC-preconditioned CG, namely vcd_solveriters=4 by
          default, or by a direct solver if vcd_directsolver=True.  Thresholding
          to capture the middle values vcd_bracket=[a,b] of this field, [0.2,0.8]
          by default, then marks only those elements which are close to the free
          boundary.

        Parameters which do not apply to the chosen algorithm are not allowed.

        Tuning advice for vcd_bracket=[a,b]:
          * lower a from default 0.2 to mark more elements in/near *inactive* set
          * raise b from default 0.8 to mark more elements in/near *active* set"""

        lb, ub = bounds
        if lb is None and ub is None:
            raise ValueError("fbmark() requires at least one non-None bound")
        if algorithm == "udo":
            for pname, p in [
                ("vcd_bracket", vcd_bracket),
                ("vcd_directsolver", vcd_directsolver),
                ("vcd_solveriters", vcd_solveriters),
            ]:
                if p is not None:
                    raise ValueError(f"{pname} is not allowed unless algorithm='vcd'")
            udo_n = 1 if udo_n is None else udo_n
        elif algorithm == "vcd":
            for pname, p in [("udo_n", udo_n), ("udo_restrict", udo_restrict)]:
                if p is not None:
                    raise ValueError(f"{pname} is not allowed unless algorithm='udo'")
            vcd_bracket = [0.2, 0.8] if vcd_bracket is None else vcd_bracket
            vcd_directsolver = False if vcd_directsolver is None else vcd_directsolver
            vcd_solveriters = 4 if vcd_solveriters is None else vcd_solveriters
        else:
            raise ValueError(f"unknown algorithm='{algorithm}'; must be 'udo' or 'vcd'")

        # mark each side separately; note that diffusing, or dilating, the
        # nodal indicator of the *union* of the two active sets would not
        # resolve a free boundary where a lower and an upper active set are
        # close together
        def sidemark(onesided, side):
            if algorithm == "udo":
                m = self._udofbmark(uh, onesided, n=udo_n, restrict=udo_restrict)
            else:
                m = self._vcdfbmark(
                    uh,
                    onesided,
                    bracket=vcd_bracket,
                    directsolver=vcd_directsolver,
                    solveriters=vcd_solveriters,
                )
            m.rename(f"mark (fbmark {side})")
            return m

        marklower = None if lb is None else sidemark((lb, None), "lower")
        markupper = None if ub is None else sidemark((None, ub), "upper")
        if marklower is None or markupper is None:
            oneside = marklower if markupper is None else markupper
            mark = Function(oneside.function_space(), name="mark (fbmark)").assign(
                oneside
            )
        else:
            mark = Function(
                marklower.function_space(), name="mark (fbmark)"
            ).interpolate(self.unionmarks(marklower, markupper))
        return (mark, marklower, markupper)

    def fixedratemark(self, eta, theta, method):
        """Marks elements according to the values of estimator eta in DG0 and a threshold which depends on the scalar theta.

        Allowed values are method in {'max', 'total'}.  Both methods threshold eta at a value ethresh, and neither ever marks an element where eta = 0.

        The default 'max' strategy marks {eta > ethresh}, where
          ethresh = theta * max eta
        Here theta is a relative threshold, and the number of elements marked is a *decreasing function of theta*: theta near 1 marks only the worst elements, theta near 0 marks nearly all of them.  (See Verfuerth (2013). A Posteriori Error Estimation Techniques for Finite Element Methods, Oxford University Press, section 4.2.)  Note ethresh is a *fraction* of the maximum, so it is generally not a value which eta attains, and the comparison is therefore strict.  In particular eta = 0 everywhere gives ethresh = 0 and marks nothing.

        The 'total' strategy sorts all elements (globally, across processes) by decreasing eta value.  Then it marks {eta >= ethresh}, where the threshold
          ethresh = eta(index)
        is the largest eta value for which
          theta * (sum_i eta_i) <= sum_{eta_i >= ethresh} eta_i
        Here ethresh *is* a value which eta attains, namely the smallest one in the marked set, so the comparison must include equality for the marked set to reach the theta fraction at all.  A strict comparison would leave the set one element short of the criterion every time, and would mark nothing whatsoever when the eta values are all tied, as they are on a mesh symmetric enough to give bit-identical indicators.
        (I.e. theta gives the fraction of the total eta sum.)  This strategy is the refine-only version of the "fixed-rate" strategy, with X=theta and Y=0, described in section 4.2 of
          W. Bangerth & R. Rannacher (2003).  Adaptive Finite Element Methods for
          Differential Equations, Springer Basel.
        It is also the bulk/Doerfler marking criterion (W. Doerfler, 1996, SIAM J. Numer. Anal. 33(3)).  Here theta is a fraction of the total error, so the number of elements marked is an *increasing function of theta*: theta near 0 marks only the worst elements, theta near 1 marks nearly all of them.

        Both strategies give identical results in parallel.  The 'total' strategy allgathers eta onto every process, so it may not scale to very-large meshes and process counts.

        Returns (mark, ethresh)."""

        DG0 = eta.function_space()
        if self.debug:
            _, _DG0 = self.spaces(DG0.mesh())
            assert DG0.ufl_element() == _DG0.ufl_element()

        with eta.dat.vec_ro as eta_:
            if method == "max":
                ethresh = theta * eta_.max()[1]  # process independent
                atthresh = False  # ethresh is not an attained value; see doc string
            elif method == "total":
                comm = DG0.mesh().comm
                values = np.concatenate(comm.allgather(eta_.array_r))
                # drop the zeros, so that they cannot be swept in by the
                # inclusive comparison below when eta vanishes on much of the mesh
                values = values[values > 0.0]
                if values.size == 0:  # no elements globally, or eta vanishes
                    ethresh = PETSc.INFINITY
                else:
                    sorted_values = np.sort(values)[::-1]  # sort in descending order
                    cumsum = np.cumsum(sorted_values)  # array
                    target = np.sum(values) * theta  # scalar
                    idx = np.argmax(cumsum >= target)  # first index so that cumsum_i >= target
                    ethresh = sorted_values[idx]
                atthresh = True  # ethresh is an attained value; see doc string
            else:
                raise ValueError("unknown method for VIAMR.fixedratemark()")

        # the marked set must include ethresh itself exactly when ethresh is a
        # value eta attains, namely in the 'total' case
        emark = ge(eta, ethresh) if atthresh else gt(eta, ethresh)
        mark = Function(DG0, name="mark (fixedratemark)").interpolate(
            conditional(emark, 1.0, 0.0)
        )
        return mark, ethresh

    def _maskexclude(self, eta, mask):
        """Return eta zeroed outside of mask (a DG0 {0,1} indicator), or eta
        unchanged if mask is None.  Always apply this to eta *before*
        VIAMR.fixedratemark(), if elements must be kept out of
        consideration for marking."""
        if mask is None:
            return eta
        DG0 = eta.function_space()
        return Function(DG0, name=eta.name()).interpolate(eta * mask)

    def inactivemark(
        self, uh, bounds, estimator="br78", res=None, alpha=None, theta=0.5, method="max"
    ):
        """Return marking within the computed inactive set by using a classical (PDE)
        a posteriori error estimator, for an obstacle problem with the given
        bounds = (lb, ub), meaning lb <= uh <= ub.  Either entry may be None for
        a problem constrained on one side only, so (lb, None) is a lower obstacle
        alone; see e.g. eleminactive().

        The estimator eta is computed as a function in DG0, and then restricted
        to the inactive set.  We call VIAMR.fixedratemark() to mark using eta
        and a threshold theta.  Then we return the mark in DG0, eta in DG0,
        and a scalar estimate for the total error in energy norm.

        The estimator is one of:

          estimator="br78":  The Babuška-Rheinboldt (1978) residual estimator,
          which requires the residual res as a UFL expression.  See
            I. Babuvska & W. C. Rheinboldt (1978). Error estimates for adaptive
            finite element computations, SIAM Journal on Numerical Analysis 15 (4),
            736--754}, https://doi.org/10.1137/0715049
          and section 2.2 of
            M. Ainsworth & J. T. Oden (2000).  A Posteriori Error Estimation in
            Finite Element Analysis, John Wiley & Sons, Inc., New York.

          estimator="bv00":  The weighted version of "br78" by
            C. Bernardi & R. Verfürth (2000). Adaptive finite element methods
            for elliptic equations with non-smooth coefficients. Numerische
            Mathematik, 85(4), 579-608.
          Requires res.  Normally takes alpha but if alpha is None then it is
          replaced by 1.0 and the estimator is actually BR78; see below.

          estimator="gradientrecovery":  A gradient-recovery estimator, using
          CG1 recovery of the DG0 gradient; see Chapter 4 of Ainsworth & Oden
          (2000).  Takes neither res nor alpha.

        For "bv00", alpha is a scalar UFL expression for the local diffusion
        coefficient in a variable-coefficient operator
          - div(alpha grad(uh)) = f.
        In practice, alpha may depend on the solution, e.g.
          alpha = uh^{gamma-1}
        for the porous media equation.  We use equations (2.8), (2.12), and
        (2.13) from BV00.

        The residual estimators, "br78" and "bv00", are intended to approximate
        the error in the appropriate energy norm.  This is the H1 seminorm for
        "br78".  Note that "bv00" with alpha=1 recovers this case, but otherwise
        it is weighted.  BV00 justifies the weighting for linear operators and positive
        bounds on alpha.  Otherwise, in general nonlinear cases, use of "bv00"
        is heuristic, based on a frozen-coefficient extension, not a proven reliable or
        efficient estimator even in the PDE case.

        WARNING for "bv00": We divide by alpha everywhere, so it should be
        positive everywhere.

        Every estimator restricts eta to the strongly inactive set, i.e.
        eleminactive(..., strong=True), the elements whose dofs are all inactive.

        The diagonal solve implementation of the estimators came from slide 109 of
          https://github.com/pefarrell/icerm2024/blob/main/slides.pdf
        See also
          https://github.com/pefarrell/icerm2024/blob/main/02_netgen/01_l_shaped_adaptivity.py
        """
        if estimator == "gradientrecovery":
            if res is not None or alpha is not None:
                raise ValueError("estimator='gradientrecovery' takes neither res nor alpha")
            eta = self._gradientrecoveryeta(uh)
        elif estimator in ("br78", "bv00"):
            if res is None:
                raise ValueError(f"estimator='{estimator}' requires res")
            if estimator != "bv00" and alpha is not None:
                raise ValueError("alpha is not allowed unless estimator='bv00'")
            if estimator == "bv00" and alpha is None:
                alpha = Constant(1.0)
            eta = self._residualeta(uh, res, alpha=alpha)
        else:
            raise ValueError(
                f"unknown estimator='{estimator}'; must be 'br78', 'bv00', or 'gradientrecovery'"
            )
        # restrict eta to inactive set, before computing the threshold; strong=True
        # means all dofs must be inactive to get imark=1
        imark = self.eleminactive(uh, bounds, strong=True)
        ieta = self._maskexclude(eta, imark)
        mark, _ = self.fixedratemark(ieta, theta, method)
        total_error_est = self._globalpnorm(ieta, 2.0)
        return (mark, ieta, total_error_est)

    def _gradientrecoveryeta(self, uh):
        """Compute the DG0 gradient-recovery estimator for inactivemark()."""
        mesh = uh.function_space().mesh()
        v = CellVolume(mesh)
        # recover a CG1 gradient of uh by projection
        CG1vec = VectorFunctionSpace(mesh, "CG", 1)
        gradrecu = Function(CG1vec).project(grad(uh))
        # cell-wise error estimator
        _, DG0 = self.spaces(mesh)
        eta_sq = Function(DG0)
        w = TestFunction(DG0)
        G = (
            inner(eta_sq / v, w) * dx
            - inner(inner(gradrecu - grad(uh), gradrecu - grad(uh)), w) * dx
        )
        # each cell needs an independent 1x1 solve, so Jacobi is an exact preconditioner
        sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
        solve(G == 0, eta_sq, solver_parameters=sp)
        return Function(DG0, name="eta on inactive set").interpolate(sqrt(eta_sq))  # eta from eta^2

    def _residualeta(self, uh, res, alpha=None):
        """Compute the DG0 residual estimator for inactivemark(): BR78 if
        alpha is None, otherwise BV00 weighted by alpha."""
        # mesh quantities
        mesh = uh.function_space().mesh()
        h = CellDiameter(mesh)
        v = CellVolume(mesh)
        n = FacetNormal(mesh)
        # cell-wise error estimator
        _, DG0 = self.spaces(mesh)
        eta_sq = Function(DG0)
        w = TestFunction(DG0)
        G = inner(eta_sq / v, w) * dx
        if alpha is None:
            # original Babuska & Rheinboldt (1978) estimator; same as BV00 if alpha=1
            G -= (
                inner(h ** 2 * res ** 2, w) * dx(degree=3)
                + 0.5 * inner(h("+") * jump(grad(uh), n) ** 2, w("+")) * dS(degree=3)
                + 0.5 * inner(h("-") * jump(grad(uh), n) ** 2, w("-")) * dS(degree=3)
            )
        else:
            # Bernardi & Verfurth (2000) weighted estimator
            muK = h / alpha ** 0.5  # equation (2.12)
            # following is equation (2.13)
            alfe = conditional(alpha("+") >= alpha("-"), alpha("+"), alpha("-"))
            mue_p = h("+") / alfe
            mue_m = h("-") / alfe
            # following is equation (2.8)
            G -= (
                inner(muK ** 2 * res ** 2, w) * dx(degree=3)
                + 0.5
                * inner(mue_p * jump(alpha * grad(uh), n) ** 2, w("+"))
                * dS(degree=3)
                + 0.5
                * inner(mue_m * jump(alpha * grad(uh), n) ** 2, w("-"))
                * dS(degree=3)
            )

        # each cell needs an independent 1x1 solve, so Jacobi is an exact preconditioner
        sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
        solve(G == 0, eta_sq, solver_parameters=sp)
        return Function(DG0, name="eta on inactive set").interpolate(sqrt(eta_sq))  # eta from eta^2

    def _dmplextransform(self, mesh, transform_type, indicator=None):
        """Apply a PETSc DMPlexTransform of the given type to mesh's topology_dm,
        returning the resulting Firedrake mesh.  Shared by refinesbr2D()
        (transform_type "refine_sbr" or "refine_regular") and _filtermesh()
        (transform_type "transform_filter").

        If indicator (a DG0 Function) is given, its nonzero cells are copied onto
        a DMPlex label and that label is set as the transform's active label --
        this drives both "refine_sbr" (refine only marked cells) and
        "transform_filter" (extract the submesh of marked cells).
        
        If indicator is None then transform_type="refine_regular" is required.
        In that case, no label is created."""
        dm = mesh.topology_dm

        # (For now the only way to set the active label with petsc4py uses
        # PETSc.Options() because DMPlexTransformSetActive() has no binding.)
        # Save whatever was already in the (global) options database so
        # this call does not permanently leak state into it.
        opts = PETSc.Options()
        optkeys = ("dm_plex_transform_active", "dm_plex_transform_type")
        savedopts = {key: opts[key] for key in optkeys if key in opts}

        if indicator is not None:
            # section for DG0 indicator
            tdim = mesh.topological_dimension
            entity_dofs = np.zeros(tdim + 1, dtype=IntType)
            entity_dofs[-1] = 1
            indicatorSect, _ = dmcommon.create_section(mesh, entity_dofs)

            # create a DMPlex label to mark cells for the transform
            dm.createLabel("_viamr_dmplextransform")
            adaptLabel = dm.getLabel("_viamr_dmplextransform")
            adaptLabel.setDefaultValue(0)

            # dmcommon provides a python binding for this operation of setting
            # the label given an indicator function data array
            if self.debug:
                _, DG0 = self.spaces(mesh)
                assert indicator.function_space().ufl_element() == DG0.ufl_element()
            dmcommon.mark_points_with_function_array(
                dm, indicatorSect, 0, indicator.dat.data_with_halos, adaptLabel, 1
            )
            opts["dm_plex_transform_active"] = "_viamr_dmplextransform"

        opts["dm_plex_transform_type"] = transform_type

        # create a DMPlexTransform object to apply the transform
        dmTransform = PETSc.DMPlexTransform().create(comm=mesh.comm)
        dmTransform.setDM(dm)
        dmTransform.setFromOptions()
        dmTransform.setUp()
        dmAdapt = dmTransform.apply(dm)
        dmTransform.destroy()

        if indicator is not None:
            # label is no longer needed
            dmAdapt.removeLabel("_viamr_dmplextransform")
            dm.removeLabel("_viamr_dmplextransform")

        # remove other labels to stop further distribution in mesh()
        # (Koki's suggestion)
        dmAdapt.removeLabel("pyop2_core")
        dmAdapt.removeLabel("pyop2_owned")
        dmAdapt.removeLabel("pyop2_ghost")

        # create a new mesh from the adapted dm
        dp = mesh._distribution_parameters  # original parameters
        newmesh = Mesh(dmAdapt, distribution_parameters=dp, comm=mesh.comm)

        # restore options database to its state before this call
        for key in optkeys:
            if key in savedopts:
                opts[key] = savedopts[key]
            elif key in opts:
                del opts[key]

        return newmesh

    def refinesbr2D(self, mesh, indicator):
        """Call PETSc DMPlex routines to do skeleton-based refinement (SBR; Plaza & Carey, 2000).
        This version works in parallel, but only in 2D.

        Regarding 2D limitation, see TODO in
          https://petsc.org/release/src/dm/impls/plex/transform/impls/refine/sbr/plexrefsbr.c.html.
        Also see
          https://petsc.org/release/overview/plex_transform_table/
        and associated links.

        Compare this method to Netgen's refine_marked_elements() which also does SBR, in 2D or 3D,
        but which does not apply to Firedrake-native meshes.

        Performance note: wall time scales with the *output* mesh size, not with
        how few cells are marked -- most of the cost is in reconstructing a new
        Firedrake Mesh (spaces, sections, halos) from the adapted DMPlex, which
        happens regardless of marked fraction.  It also peaks at intermediate
        (~50%) marked fractions rather than at 100%, from the extra conformity
        handling needed at marked/unmarked boundaries.  So marking sparsely does
        not buy a proportionally cheap call.

        Parameters
        ----------
        mesh : firedrake.Mesh
            The mesh to refine.
        indicator : firedrake.Function or "uniform"
            A DG0 indicator function marking which cells to refine
            (nonzero means refine).  Pass the literal string "uniform"
            instead to refine every cell uniformly.

        Returns
        -------
        firedrake.Mesh
            The refined mesh.
        """
        if indicator == "uniform":
            return self._dmplextransform(mesh, "refine_regular")
        return self._dmplextransform(mesh, "refine_sbr", indicator=indicator)

    def _filtermesh(self, mesh, indicator):
        """Return the submesh containing only the cells where the DG0 indicator
        is nonzero, via PETSc's DMPlex "transform_filter" transform."""
        return self._dmplextransform(mesh, "transform_filter", indicator=indicator)
