import time
import warnings
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

from .avm import AVMMixin, haveanimate


class VIAMR(OptionsManager, AVMMixin):
    r"""A VIAMR object manages adaptive mesh refinement (AMR) for a Firedrake
    variational inequality (VI) solver, where the VI constraint set is defined by
    box bounds (lb <= u <= ub).

    Central notions behind this class:
      * Like a PDE AMR method, refinement in the inactive set is guided by
        an a posteriori estimator.
      * For some problems, refinement in the active set is worthwhile, but for some
        it is wasted effort.
      * Additional refinement near the free boundary is compatible with the goals of free
        boundary models.  For example, a purpose of solving glacier problems is to know which land
        is glaciated.  Refining near the free boundary matches user goals even if it does not
        reduce the norm error of the solution.  It does reduce the geometrical measures of set
        errors; see hausdorf() and jaccard() in this class.

    The public mark-and-refine API of the VIAMR class consists of:

      udomark():  marking method targeting refinement of the computed free boundary, based on a purely-discrete unstructured-dilation operation

      vcdmark():  marking method targeting refinement of the computed free boundary, based on diffusing the computed free boundary using mesh size in a variable coefficient

      gradreinactivemark():  classical a posterior error estimator, applied in the computed inactive set, using CG1 recovery of the DG0 gradient

      brinactivemark():  classical a posterior error estimator, applied in the computed inactive set, implementing either the method from Babushka & Rheinboldt (1978) or its weighted extension from Bernardi & Verfurth (2000)

      nsvmark():  mark using the "practical estimator" from Nochetto, Siebert, & Veeser (2003) = NSV03

      nsv05mark():  mark using the fully-localized, star-based estimator from Nochetto, Siebert, & Veeser (2005) = NSV05, the successor of NSV03

      fixedratemark():  general-purpose thresholding of an elementwise DG0 estimator field by a fixed-rate ('max' or 'total'/bulk/Doerfler) criterion; used internally by gradrecinactivemark(), brinactivemark(), nsvmark(), and nsv05mark(), but also usable directly

      unionmark():  a method for combining existing marks

      safeactiveunmark():  a method which detects active-set elements where higher-order inspection gives evidence that refinement is wasted effort; this needs exact data for the obstacle and source term

      refinesbr2D():  a method which calls PETSc for skeleton-based-refinement (SBR)

      eleminactive():  element markings for the computed inactive set

      elemactive(), thinelemactive():  two versions of element markings for computed active sets

      lowerboundcelldiameter():  unmark elements with cell diameters below a minimum cell diameter

    There are also diagnostic methods:

      jaccard(), jaccardUFL():  computation of the Jaccard similarity index for two active sets

      hausdorff2D():  compute the Hausdorff distance between two edge sets E1, E2 in a planar mesh

      freeboundarygraph2D():  for 2D obstacle problems, return the computed free boundary

    Some default calls to the major marking-and-refine methods are:

    .. code-block:: python3

      amr = VIAMR()
      fbmark = amr.udomark(uh, lb)                             # free-boundary targeted marking method
      fbmark = amr.vcdmark(uh, lb)                             # same, but based on diffusion
      imark, _, _ = amr.gradrecinactivemark(uh, lb)            # classical gradient recovery in inactive set
      imark, _, _ = amr.brinactivemark(uh, lb, res_ufl)        # classical BR78 estimator in inactive set
      imark, _, _ = amr.brinactivemark(uh, lb, res_ufl, Z=Z)   # weighted estimator (BV00) in inactive set
      mark, _, _, _, _ = amr.nsvmark(uh, lb, g, f_ufl, g_ufl)  # method from NSV03
      mark, _, _, _, _ = amr.nsv05mark(uh, lb, g, f_ufl, g_ufl)  # method from NSV05
      mark, ethresh, _ = amr.fixedratemark(eta, theta=0.5, method="total")  # threshold a DG0 estimator eta
      mark = amr.unionmarks(fbmark, imark)                     # mark if either is marked
      rmesh = amr.refinesbr2D(mesh, mark)                      # PETSc DMPlexTransform for skeleton-based refinement

    Regarding the arguments: uh is a computed VI solution, lb is a lower-bound obstacle, res_ufl is a UFL expression for the residual (applicable in the inactive set), Z is a weighting field (see examples), f_ufl is the source term in Poisson equation, and g_ufl are the boundary values.

    The methods above generalize to upper-bound obstacles by passing ub in the same position as lb and adding the boxside="upper" kwarg; see each method's own docstring.  Note that unionmarks() can be used to apply both lower and upper bounds.

    TODO: nsvmark() and safeactiveunmark() are not yet generalized this way.

    Regarding returned values: fbmark, imark, and mark are element markings in DG0 (Definition 4.2 in paper), rmesh is a refined mesh, and amesh is an adapted mesh.

    There is also a mesh adaptation API which needs the animate library:

    .. code-block:: python3

      import animate
      metric = amr.buildaveragedmetric(mesh, uh, lb)            # VIAMR builds the metric ...
      amesh = animate.adapt(mesh, metric)                       # ... caller adapts the mesh with it

    There are also some utility methods, including: spaces(), meshsizes(), meshreport(), scalarrange(), checkadmissible(), and countmark().  Other methods starting with an underscore are (roughly) intended to be private to the VIAMR class.

    Known limitations:
      * Functions which do not work in parallel: 1. jaccard(..., submesh=False).
      * Functions whose results depend on number of processes: 1. vcdmark(), 2. buildaveragedmetric() (via vcdmark()).
      * Functions which only work for 2D meshs: 1. freeboundarygraph2D(), 2. hausdorff2D(), 3. refinesbr2D()
      * Functions which only work for 2D triangular meshes: 1. refinesbr2D()

    Regarding the last limitation, see the doc string of refinesbr2D(), and compare to refine_marked_elements() from NetGen/ngspetsc.
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

    def checkadmissible(self, uh, bound, strict=False, boxside="lower"):
        """Utility function to check admissibility or strict admissibility of uh
        with respect to a single obstacle bound.  Returns True if uh >= bound
        (boxside="lower") or uh <= bound (boxside="upper")."""
        if self.debug:
            assert boxside in ("lower", "upper"), "boxside must be 'lower' or 'upper'"
        upper = boxside == "upper"
        if strict:
            if upper:
                bad = assemble(conditional(uh > bound, 1.0, 0.0) * dx)
            else:
                bad = assemble(conditional(uh < bound, 1.0, 0.0) * dx)
            return bad == 0.0
        else:
            V = uh.function_space()
            delta = Function(V).interpolate(bound - uh if upper else uh - bound)
            return self._globalextreme(delta, minimum=True) >= 0.0

    def _checkuhbound(self, uh, bound, boxside="lower"):
        """Debug-mode validation shared by the unilateral obstacle problem
        indicator methods: checks that uh is a Function, bound is a Function
        or Constant, and uh is admissible with respect to bound on the given
        boxside ("lower" or "upper").  No-op unless self.debug is True."""
        if self.debug:
            assert isinstance(uh, Function), "input uh must be of class Function"
            isbound = isinstance(bound, Function) or isinstance(bound, Constant)
            assert isbound, "input bound must be of class Function or Constant"
            assert self.checkadmissible(uh, bound, boxside=boxside)

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

    def _nodalactive(self, uh, bound, boxside="lower"):
        """Compute nodal active set indicator in same function space as uh, for a
        unilateral obstacle problem with the given bound.  boxside="lower" treats
        bound as a floor (uh >= bound); boxside="upper" treats it as a ceiling
        (uh <= bound).  The nodal active set is
          {x in N(V): |u(x) - bound(x)| < activetol}
        where N(V) is the nodal set for V = uh.function_space().  Active nodes get value 1.0."""
        self._checkuhbound(uh, bound, boxside=boxside)
        z = Function(uh.function_space(), name="Nodal Active")
        z.interpolate(conditional(abs(uh - bound) < self.activetol, 1.0, 0.0))
        return z

    def elemactive(self, uh, bound, boxside="lower"):
        """Compute an element active set indicator in DG0, for a unilateral
        obstacle problem with the given bound (boxside="lower" or "upper";
        see _nodalactive()).  Active elements get value 1.0.  Elements are
        marked active if the DG0 degree of freedom for that element is
        active, within activetol, so use with caution if z is not in CG1."""
        self._checkuhbound(uh, bound, boxside=boxside)
        _, DG0 = self.spaces(uh.function_space().mesh())
        z = Function(DG0, name="Element Active")
        z.interpolate(conditional(abs(uh - bound) < self.activetol, 1.0, 0.0))
        return z

    def eleminactive(self, uh, bound, boxside="lower", strong=False):
        """Compute an element inactive set indicator in DG0, for a unilateral
        obstacle problem with the given bound (boxside="lower" or "upper";
        see _nodalactive()).  Inactive elements get value 1.0.  By default,
        elements are marked inactive if their DG0 degree of freedom is
        inactive (by activetol).

        If strong=True then an element is only marked as inactive if all
        degrees of freedom of the gap function (uh-bound for boxside="lower",
        bound-uh for boxside="upper") exceed activetol.  That is, a cell is
        "strongly" inactive if all of its original dofs are inactive."""
        self._checkuhbound(uh, bound, boxside=boxside)
        if strong:
            # note gap > 0 is equivalent to strictly inactive ... but we use activetol
            gap_ufl = (bound - uh) if boxside == "upper" else (uh - bound)
            v = Function(uh.function_space()).interpolate(gap_ufl)
            # z is in DG0 and contains min of v over each cell's dofs
            z = self._elemextreme(v, minimum=True, defaultval=PETSc.INFINITY)
            z.interpolate(conditional(z > self.activetol, 1.0, 0.0))
            z.rename("Element Inactive")
        else:
            _, DG0 = self.spaces(uh.function_space().mesh())
            z = Function(DG0, name="Element Inactive")
            z.interpolate(conditional(abs(uh - bound) < self.activetol, 0.0, 1.0))
        return z

    def thinelemactive(self, uh, bound, boxside="lower"):
        """Compute element active set indicator into DG0, but "thinned", for a
        unilateral obstacle problem with the given bound (boxside="lower" or
        "upper"; see _nodalactive()).

        In contrast to elemactive(), here a cell is marked as active only if it *and its neighboring cells* are active.  The test for active is based on testing at the DG0 degree of freedom, and according to activetol.  Returns a DG0 element-wise indicator, with thinned-active elements having value 1.

        The implementation is inspired by VIAMR.udomark(): a thinned-active element
        is exactly one which is *not* within one ring of an inactive element, i.e.
        z = 1 - dilate1(inactive), computed via the same two constant-arity PyOP2
        kernels udomark() uses (cell->node max-scatter, then node->cell max-gather),
        with no DMPlex access.
        """
        mesh = uh.function_space().mesh()
        CG1, DG0 = self.spaces(mesh)
        inactive = self.eleminactive(uh, bound, boxside=boxside)
        grown = self._elemextreme(
            self._elemtonodemax(inactive, CG1), minimum=False, defaultval=0.0
        )
        return Function(DG0, name="Thin Element Active").interpolate(1.0 - grown)

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

    def _elemtonodemax(self, elemfield, nodalspace):
        """Scatter a DG0 element field into nodalspace (e.g. CG1), broadcasting
        each cell's value to all of its local nodes and taking the max where
        multiple cells share a node.  Applies a PyOP2 parallel loop; correct in
        parallel via Firedrake's own halo exchange, with no DMPlex access needed."""
        target = Function(nodalspace).assign(0.0)
        kernel = op2.Kernel(
            """
        void elem_to_node_max(double *target, double const *source)
        {
        for (int i = 0; i < %(ndofs)s; i++) {
            target[i] = source[0];
        }
        }"""
            % {"ndofs": nodalspace.finat_element.space_dimension()},
            "elem_to_node_max",
        )
        op2.par_loop(
            kernel,
            nodalspace.mesh().cell_set,
            target.dat(op2.MAX, target.cell_node_map()),
            elemfield.dat(op2.READ, elemfield.cell_node_map()),
        )
        return target

    def _elemtonodeextreme(self, elemfield, nodalspace, minimum=False, defaultval=None):
        """Scatter a DG0 element field into nodalspace (e.g. CG1), broadcasting each
        cell's value to all of its local nodes and taking the extreme value where
        multiple cells share a node.  Thus the node value is the extreme over the
        star U_h(z) of the element values.  Either computes the maximum or (optionally)
        the minimum; the user must set the default value, which initializes the
        reduction.  (In contrast, _elemtonodemax() always initializes to 0.0, so it
        is only correct for nonnegative fields.)  Applies a PyOP2 parallel loop;
        correct in parallel via Firedrake's own halo exchange, with no DMPlex access
        needed."""
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
        sign matters here, in contrast to nsvmark(), which only uses |J_h|.

        Facet values are recovered by dividing an assembled facet integral by the
        facet measure; this is exact because the trace space is elementwise
        constant, so its mass matrix is diagonal.  Exterior facets get the value
        zero, which is what the NSV05 theory wants: its facet set Gamma consists
        of interior facets only.

        The optional input mask is a DG0 {0,1} indicator.  When given, a facet
        gets a nonzero value only if *both* of its neighboring elements are in
        the mask.  This computes the restriction to Gamma_h^+ in NSV05, noting
        that Omega_h^+ is open, so a facet lying in the boundary of the
        full-contact set Omega_h^0 contributes nothing."""
        mesh = uh.function_space().mesh()
        T0 = FunctionSpace(mesh, "HDiv Trace", 0)
        w0 = TestFunction(T0)
        n = FacetNormal(mesh)
        Jh_ufl = -jump(grad(uh), n)
        if mask is not None:
            Jh_ufl = Jh_ufl * mask("+") * mask("-")
        # facet measure on *all* facets, so the division below never divides by zero
        areaS = assemble(w0("+") * dS + w0 * ds)
        Jh = Function(T0, name="J_h (facet jump)")
        Jh.dat.data[:] = (
            assemble(Jh_ufl * w0("+") * dS).dat.data_ro / areaS.dat.data_ro
        )
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
        is the opposite sign from nsvmark()'s sigma_h: s_z < 0 there corresponds to
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

    def udomark(self, uh, bound, boxside="lower", n=1, restrict=None):
        """Mark mesh using Unstructured Dilation Operator (UDO) algorithm, for a
        unilateral obstacle problem with the given bound (boxside="lower" or
        "upper"; see _nodalactive()).  The algorithm
        first computes an element-wise indicator for the free boundary.  Then the elements
        which neighbor free-boundary elements are added, and so on iteratively through n
        levels.  Note that n=0 already minimally marks the free boundary.  Optionally the
        marking can be restricted to the active side of the initially-marked elements
        (restrict="active"), or to the inactive side (="inactive").

        The output is an element-wise marking for those elements near the free boundary
        which should be refined."""

        # get mesh and border mark; added flag for restriction
        if restrict is not None:
            meshInit = uh.function_space().mesh()
            if restrict == "active":
                # restrict to active set plus border
                indicator = Function(FunctionSpace(meshInit, "DG", 0)).interpolate(
                    self.elemactive(uh, bound, boxside=boxside)
                    + self._elemborder(self._nodalactive(uh, bound, boxside=boxside))
                )
            elif restrict == "inactive":
                # restrict to inactive set, which contains border already
                indicator = self.eleminactive(uh, bound, boxside=boxside)
            else:
                raise ValueError(
                    f"unknown restrict='{restrict}'; must be 'active', 'inactive', or None"
                )
            mesh = self._filtermesh(meshInit, indicator)
            CG1, DG0 = self.spaces(mesh)
            # Use nodal active set indicator to make an initial DG0 element border
            # indicator. This is now on a restricted domain so allow_missing_dofs=True
            border = Function(DG0).interpolate(
                self._elemborder(self._nodalactive(uh, bound, boxside=boxside)),
                allow_missing_dofs=True,
            )
        else:
            mesh = uh.function_space().mesh()
            CG1, DG0 = self.spaces(mesh)
            # Use nodal active set indicator to make an initial DG0 element border
            # indicator.
            border = self._elemborder(self._nodalactive(uh, bound, boxside=boxside))

        # main loop: expand element border out to n levels, via two constant-arity
        # PyOP2 kernels per level (cell->node max-scatter, then node->cell max-gather).
        # No DMPlex access, and thus no special mesh overlap requirement.
        for _ in range(n):
            border = self._elemextreme(
                self._elemtonodemax(border, CG1), minimum=False, defaultval=0.0
            )

        return Function(DG0, name="mark (udomark)").interpolate(
            border, allow_missing_dofs=True
        )

    def vcdmark(
        self,
        uh,
        bound,
        boxside="lower",
        bracket=[0.2, 0.8],
        returnSmooth=False,
        directsolver=False,
        vcdsolveriters=4,
    ):
        """Mark mesh using Variable Coefficient Diffusion (VCD) algorithm, for a
        unilateral obstacle problem with the given bound (boxside="lower" or
        "upper"; see _nodalactive()).  The algorithm computes a nodal active set indicator and then diffuses it, using a variable coefficient based on mesh geometry.  Diffusion is by solving a single backward Euler time step for the corresponding time-dependent diffusion equation.  The linear equations are solved by a fixed number of iterations of ICC-preconditioned CG.  Thresholding to capture the middle values of this field then marks only those elements which are close to the free boundary.  The output is an element-wise marking for elements to refine near the free boundary.
        Tuning advice:  The bracket [a,b] should be adjusted as follows:
          * lower a from default 0.2 to mark more elements in/near *inactive* set
          * raise b from default 0.8 to mark more elements in/near *active* set"""

        # Compute nodal active set indicator
        mesh = uh.function_space().mesh()
        CG1, DG0 = self.spaces(mesh)
        nu = self._nodalactive(uh, bound, boxside=boxside)

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
                "ksp_max_it": vcdsolveriters,
                "ksp_convergence_test": "skip",
                "pc_type": "icc",
            }
            if mesh.comm.size > 1:
                sp.update({"pc_type": "asm", "pc_asm_overlap": 1, "sub_pc_type": "icc"})
        solve(a == L, u, solver_parameters=sp, options_prefix="viamr_vcd")

        # apply thresholding and interpolate into DG0
        if returnSmooth:
            return u
        middleUFL = conditional(u > bracket[0], conditional(u < bracket[1], 1, 0), 0)
        return Function(DG0, name="mark (vcdmark)").interpolate(middleUFL)

    def fixedratemark(self, eta, theta, method):
        """Marks elements according to the values of estimator eta in DG0 and a threshold which depends on the scalar theta.

        The default 'max' strategy marks all elements with eta greater than
          ethresh = theta * max eta
        Here theta is a relative threshold, and the number of elements marked is a *decreasing function of theta*: theta near 1 marks only the worst elements, theta near 0 marks nearly all of them.  (See Verfuerth (2013). A Posteriori Error Estimation Techniques for Finite Element Methods, Oxford University Press, section 4.2.)

        The 'total' strategy sorts all elements (globally, across processes) by decreasing eta value.  Then the threshold
          ethresh = eta(index)
        equals the eta value where theta times the total sum of eta is equal to the sum of the eta values above ethresh.  (I.e. theta gives the fraction of the total eta sum.)  The 'total' strategy is the refine-only version of the "fixed-rate" strategy, with X=theta and Y=0, described in section 4.2 of
          W. Bangerth & R. Rannacher (2003).  Adaptive Finite Element Methods for
          Differential Equations, Springer Basel.
        This is also the bulk/Doerfler marking criterion (W. Doerfler, 1996, SIAM J. Numer. Anal. 33(3)).  Here theta is a fraction of the total error, so the number of elements marked is an *increasing function of theta*.

        Both strategies give identical results in parallel.  The 'total' strategy allgathers eta onto every process, so it may not scale to very-large meshes and process counts.

        Returns (mark, ethresh, total_error_est).  The last is the l^2 norm of eta over the elements, which is the correct global estimator only when eta is an energy-norm indicator, as in gradrecinactivemark() and brinactivemark().  It is meaningless for a max-norm estimator such as nsvmark() or nsv05mark(), where the global quantity is a sup over elements; those methods therefore ignore it and return their own Eh."""

        with eta.dat.vec_ro as eta_:
            if method == "max":
                ethresh = theta * eta_.max()[1]  # process independent
            elif method == "total":
                comm = eta.function_space().mesh().comm
                values = np.concatenate(comm.allgather(eta_.array_r))
                if values.size == 0:  # global mesh has no elements
                    ethresh = PETSc.INFINITY
                else:
                    sorted_values = np.sort(values)[::-1]  # sort in descending order
                    cumsum = np.cumsum(sorted_values)
                    target = np.sum(values) * theta  # proportion of total error
                    idx = np.argmax(cumsum >= target)
                    ethresh = sorted_values[idx]
            else:
                raise ValueError("unknown method for VIAMR.fixedratemark()")
            total_error_est = sqrt(eta_.dot(eta_))  # l^2 norm of eta as Vec

        DG0 = eta.function_space()
        mark = Function(DG0, name="mark (fixedratemark)").interpolate(
            conditional(gt(eta, ethresh), 1.0, 0.0)
        )
        return mark, ethresh, total_error_est

    def _maskexclude(self, eta, mask):
        """Return eta zeroed outside of mask (a DG0 {0,1} indicator), or eta
        unchanged if mask is None.  Always apply this to eta *before*
        VIAMR.fixedratemark(), if elements must be kept out of
        consideration for marking (e.g. VIAMR.safeactiveunmark())."""
        if mask is None:
            return eta
        DG0 = eta.function_space()
        return Function(DG0, name=eta.name()).interpolate(eta * mask)

    def gradrecinactivemark(self, uh, bound, boxside="lower", theta=0.5, method="max", safe=None):
        """Return marking within the computed inactive set by using an
        a posteriori gradient-recovery error indicator, for a unilateral
        obstacle problem with the given bound (boxside="lower" or "upper";
        see _nodalactive()).  See Chapter 4 of
          M. Ainsworth & J. T. Oden (2000).  A Posteriori Error Estimation in
          Finite Element Analysis, John Wiley & Sons, Inc., New York.
        The optional input safe is a DG0 {0,1} indicator, e.g. the output of
        safeactiveunmark(), of elements to additionally exclude from
        consideration; see VIAMR._maskexclude()."""
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
        eta = Function(DG0, name="eta on inactive set").interpolate(sqrt(eta_sq))  # eta from eta^2
        # restrict grad recovery eta to inactive set, further excluding any
        # certified-safe elements, before computing the threshold
        imark = self.eleminactive(uh, bound, boxside=boxside)
        mask = imark if safe is None else Function(DG0).interpolate(imark * (1.0 - safe))
        ieta = self._maskexclude(eta, mask)
        # compute mark in inactive set
        mark, _, total_error_est = self.fixedratemark(ieta, theta, method)
        return (mark, ieta, total_error_est)

    def brinactivemark(
        self, uh, bound, res, boxside="lower", theta=0.5, method="max", alpha=None, safe=None
    ):
        """Return marking within the computed inactive set by using the
        a posteriori Babuška-Rheinboldt (1978) residual error indicator,
        or a weighted version of it, for a unilateral obstacle problem with
        the given bound (boxside="lower" or "upper"; see _nodalactive()).

        The primary inputs are the current solution uh, the obstacle bound (to
        restrict to the inactive set), and the residual res as a UFL expression.

        The optional input safe is a DG0 {0,1} indicator, e.g. the output of
        safeactiveunmark(), of elements to additionally exclude from
        consideration; see VIAMR._maskexclude().

        The output BR indicator eta is computed as a function in DG0.  We call
        VIAMR.fixedratemark() to mark using eta and a threshold theta.
        Then we return the marking mark, estimator eta, and a scalar estimate for
        the total error in energy norm.

        For the basic unweighted method see
          I. Babuvska & W. C. Rheinboldt (1978). Error estimates for adaptive
          finite element computations, SIAM Journal on Numerical Analysis 15 (4),
          736--754}, https://doi.org/10.1137/0715049
        and section 2.2 of
          M. Ainsworth & J. T. Oden (2000).  A Posteriori Error Estimation in
          Finite Element Analysis, John Wiley & Sons, Inc., New York.

        The optional input alpha is a scalar UFL expression for the local
        diffusion coefficient in a variable-coefficient operator
          - div(alpha grad(uh)) = f.
        In practice, alpha may depend on the solution, e.g.
          alpha = uh^{gamma-1}
        for the porous media equation.  When alpha is not None, eta is reweighted
        following the classical variable-coefficient residual estimator by
          C. Bernardi & R. Verfürth (2000). Adaptive finite element methods
          for elliptic equations with non-smooth coefficients. Numerische
          Mathematik, 85(4), 579-608.
        We use equations (2.8), (2.12), and (2.13) from this reference.

        The returned residual estimator eta is intended to approximate the error
        in the appropriate energy norm.  This is the H1 seminorm in the unweighted
        BR78 case, and setting alpha=1 recovers this case.  In the linear,
        uniformly-elliptic setting, with positive bounds on alpha, BV00 justifies
        the weighted formulation.  Otherwise, in general nonlinear cases, this
        method is a heuristic, frozen-coefficient extension, not a proven
        reliable or efficient estimator.

        WARNING when passing alpha: We divide by this coefficient everywhere,
        so it should be positive everywhere.

        The diagonal solve implementation of this function came from slide 109 of
          https://github.com/pefarrell/icerm2024/blob/main/slides.pdf
        See also
          https://github.com/pefarrell/icerm2024/blob/main/02_netgen/01_l_shaped_adaptivity.py
        """
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
        eta = Function(DG0, name="eta on inactive set").interpolate(sqrt(eta_sq))  # eta from eta^2
        # restrict BR eta to inactive set; strong=True means all dofs must be inactive to
        # get imark=1; further exclude any certified-safe elements, before computing the
        # threshold
        imark = self.eleminactive(uh, bound, boxside=boxside, strong=True)
        mask = imark if safe is None else Function(DG0).interpolate(imark * (1.0 - safe))
        ieta = self._maskexclude(eta, mask)
        mark, _, total_error_est = self.fixedratemark(ieta, theta, method)
        return (mark, ieta, total_error_est)

    def nsvmark(
        self,
        uh,
        lb,
        g,
        f_ufl,
        g_ufl,
        method="max",
        theta=0.5,
        dualtol=1.0e-10,
        C0=0.1,
        C1=0.01,
        fdegree=3,
        etadratio=1.0,
        safe=None,
    ):
        """For classical obstacle problems, with the Laplacian as the operator, compute marking on entire domain according to the local 'practical estimator' from NSV03:

            Nochetto, R. H., Siebert, K. G., & Veeser, A. (2003). Pointwise
            a posteriori error control for elliptic obstacle problems.
            Numerische Mathematik, 95(1), 163-195.

        The main formula (7.1) in NSV03 is
            eta_infty =
                  C_0 h_T^2 ||R_infty||_infty                      [term 1]
                + ||(chi - u_h)^+||_infty                          [term 2]
                + 1_{sigma_h > 0} * ||(u_h - chi)^+||_infty        [term 3]
                + ||g - I_h g||_{infty; partial Omega cap T}       [term 4]
        But there is a second quantity, the L^d "quadrature indicator" of section 7.1:
            eta_d = C_1 h_T^2 ||grad(sigma_h)||_{d; Lambda_h cap T}    [term eta_d]
        Both eta_.. are computed on each triangle T in the mesh.

        Meaning:
          term 1:  Estimates the residual relevant to the VI problem; see below for the R_infty formula, which uses the discrete residual sigma_h below.  C_0=0.1 is used by NSV03.

          term 2:  Assumed to be zero because we take chi=chi_h here and assert strict admissibility.  [<-- FIXME could be improved]

          term 3:  This "blocked gap" is gap = u_h - chi_h, but blocked according to the simplest discrete residual sigma_h, computed below.  (Note that we assert sigma_h > -dualtol below.)  The sup is over {sigma_h < 0} cap T, so an element contributes only if it lies in that set, i.e. only if *every* vertex has a strictly active multiplier.  Under the chi = chi_h assumption of term 2 this makes the term identically zero, since complementarity puts the gap at zero on exactly those nodes; see the comment at the code below.

          term 4:  Estimates the boundary interpolation error, and we use a formula which is correct if g is in CG4.  Being a sup norm over partial Omega cap T, it is computed as the elementwise sup of |g - I_h g| over the elements touching the boundary.

          term eta_d:  Controls the mass-lumping/quadrature error incurred when computing sigma_h (see sec. 6.3 and 7.1 in NSV03).  sigma_h is CG1, so grad(sigma_h) is elementwise constant, and its L^d(T) norm is |grad(sigma_h)|_T * |T|^{1/d}.  Localized to Lambda_h (the discrete contact set, approximated here by tactive below, the same "neighborhood active" indicator used for term 1's X) because sigma = 0 off the contact set (see (1.2) in NSV03), so grad(sigma_h) there is quadrature noise rather than signal.  C_1=0.01 is the practical value used by NSV03 in (7.1).

        Regarding the last term, NSV03 sec. 7.1 notes that eta_d "exhibits different accumulation" than eta_infty.  That is, eta_d, as a genuine L^d(Lambda_h) norm, aggregates over T by an L^d-type sum.  Mixing both into one scalar before marking would let eta_d's different scaling distort the max-based threshold.  Following NSV03, we mark in two separate passes.  First on eta_infty, then on eta_d restricted to Lambda_h, and take the union.  NSV03 further qualifies that the second pass only runs "provided quadrature dominates the estimator," so the second pass runs only if max(eta_d) > etadratio * max(eta_infty).  NSV03 does not give a precise numerical criterion for "dominates", so etadratio is exposed as a parameter.

        The optional input safe is a DG0 {0,1} indicator, e.g. the output of
        safeactiveunmark(), of active-set elements certified safe to leave
        unmarked.  When given, eta_infty and eta_d are masked to zero on
        those elements *before* VIAMR.fixedratemark() applies its threshold
        strategy.  (Thus the eta_... fields are not merely filtered from
        the resulting mark afterward.)

        Returns (mark, etainf, sigmah, Eh, etad).  Eh is the scalar estimator Etilde_h of (7.1) itself, so it is the quantity which bounds max(||u - u_h||_{0,inf;Omega}, ||sigma - sigmatilde_h||_{-2,inf;Omega}), and thus the right numerator for an effectivity index.  Each of its terms is accumulated in its own norm: the four sup-norm terms are maximized separately, since (7.1) adds the global norms rather than maximizing their elementwise sum eta_infty; and eta_d is accumulated as an L^d-type sum of d-th powers.  So Eh is *not* the l^2 norm of any single field, and in particular it is not what VIAMR.fixedratemark() returns, which is only meaningful for the energy-norm estimators of brinactivemark() and gradrecinactivemark().  The masking by safe= is a marking policy, so it is deliberately not applied when forming Eh.
        """
        # mesh quantities
        mesh = uh.function_space().mesh()
        CG1, DG0 = self.spaces(mesh)
        n = FacetNormal(mesh)
        hT = project(CellSize(mesh), DG0)  # versus mesh.cell_sizes(), which is in CG1

        # compute residual sigmah in CG1 following section 2.1 of NSV03, page 169,
        #   but use opposite sign convention so sigmah >= 0.   complementarity is
        #   uh >= lb,  sigmah >= 0,  (uh - lb) sigmah = 0
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
        #   "Whole star active" is computed by dilating the *inactive* nodal set by one
        #   element ring (node -> element max, then element -> node max) and negating,
        #   the same one-ring-erosion idea thinelemactive() uses elementwise.
        nodalinactive = Function(CG1).interpolate(1.0 - self._nodalactive(uh, lb))
        elemtouchesinactive = self._elemextreme(
            nodalinactive, minimum=False, defaultval=0.0
        )
        nodetouchesinactive = self._elemtonodemax(elemtouchesinactive, CG1)
        starwhollyactive = Function(CG1).interpolate(1.0 - nodetouchesinactive)
        bdryval = Function(CG1).interpolate(
            starwhollyactive * conditional(sigmah > 0.0, sigmah, 0.0)
        )
        DirichletBC(CG1, bdryval, "on_boundary").apply(sigmah)

        # check dual admissiblity (up to tolerance)
        assert self._globalextreme(sigmah, minimum=True) >= -dualtol

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
        # and then maximized over each cell's own facets.  (A previous version
        # divided an assembled facet integral by the cell volume instead, which
        # scales like jump*h^(d-1)/h^d, i.e. a factor h_T too large.  That made
        # C0*h_T^2*R_inf tend to C0*|jump| instead of to zero, so an element with a
        # kink in chi -- where the jump does not vanish under refinement -- kept an
        # O(1) indicator forever and permanently dominated the marking threshold.)
        # _facetjump() gives exterior facets the value zero, which is exactly the
        # "\setminus \partial \Omega" restriction wanted here.
        jumpu = self._elemmaxabs(self._facetjump(uh))
        tactive = self.thinelemactive(uh, lb)
        X_ufl = abs(f_ufl + tactive * sigmah)
        # note pages 188-189 in NSV03 regarding use of DG7, to deal with the fact
        # that f_ufl is generally not in CG1:
        #     "For terms involving non-polynomial data, the maximum norm is
        #      approximated by evaluating element point-values at the Lagrange
        #      nodes for 7th order polynomials.""
        # BUT using DG7 this way is really slow because it is so big, so we drop
        # to DG3 by default; DG3.dim() = 10*DG0.dim(), while DG7.dim() ~= 40*DG0.dim()
        DGf = FunctionSpace(mesh, "DG", fdegree)
        Rinf = Function(DGf).interpolate((jumpu / hT) + X_ufl)
        Rinf = self._elemmaxabs(Rinf)

        # admissibility check; FIXME removes term 2 "(\chi - u_h)_+" from
        #   the estimator *only when* \chi=lb is representable in u_h's space
        gaph = Function(CG1).interpolate(uh - lb)
        assert self._globalextreme(gaph, minimum=True) >= 0.0

        # term 3
        # The sup is over {sigma_h < 0} \cap T, so an element contributes only if it
        # lies *in* that set, which is tested by requiring every vertex to have a
        # strictly active multiplier.  (A previous version tested sigma_h at the DG0
        # degree of freedom, i.e. at the centroid, where a P1 field takes the average
        # of its vertex values.  A free-boundary border element with one contact
        # vertex and two inactive ones then passed the test, and contributed the max
        # of the gap over the *whole* element -- a value attained at an inactive
        # vertex, which is not in {sigma_h < 0} at all.  On the pyramid example that
        # pinned this term at g - chi = 1/5 on the boundary, at every level.)
        # Note that under the chi = chi_h assumption asserted above, this term is now
        # identically zero: complementarity makes the gap vanish at exactly the nodes
        # where the multiplier is active.  That is the honest value here.  NSV03's
        # Remark 5.8, where this term drives the initial refinement, relies on the
        # *continuous* chi in (u_h - chi)^+, so the term regains its content only
        # once the FIXME above is addressed.  Compare nsv05mark(), whose Lambda_h of
        # (2.20) is a union of whole stars, so there the analogous term does not
        # degenerate.
        elemincontact = self._elemextreme(
            sigmah, minimum=True, defaultval=PETSc.INFINITY
        )
        blockgap_ufl = conditional(
            elemincontact > dualtol, self._elemmaxabs(gaph), 0.0
        )
        blockgap = Function(DG0).interpolate(blockgap_ufl)

        # term 4
        # This is a sup norm over \partial \Omega \cap T, so it is computed as the
        # elementwise sup of |g - I_h g| on the elements which touch the boundary.
        # That overestimates the sup over the boundary facets themselves, but it is
        # an upper bound and so keeps the estimator reliable.  (A previous version
        # divided an assembled boundary integral by the cell volume, which has the
        # units of |g - I_h g| / h rather than of a sup norm; compare term 1.)
        CG4 = FunctionSpace(mesh, "CG", 4)  # CG4.dim() ~ 9*DG0.dim()
        adg = self._elemmaxabs(Function(CG4).interpolate(g_ufl - g))
        touchesbdry = Function(DG0)
        touchesbdry.dat.data[:] = assemble(v0 * ds).dat.data_ro
        bdryerr = Function(DG0).interpolate(
            conditional(touchesbdry > 0.0, 1.0, 0.0) * adg
        )

        # finally compute eta_inf; see doc string above for formula
        residterm = Function(DG0).interpolate(C0 * hT ** 2 * Rinf)
        etainf_ufl = residterm + blockgap + bdryerr
        etainf = Function(DG0, name="eta_inf").interpolate(etainf_ufl)

        # mask out any certified-safe elements *before* thresholding, so they
        # cannot set (or dilute) the fixedratemark() threshold; see VIAMR._maskexclude()
        notsafe = None if safe is None else Function(DG0).interpolate(1.0 - safe)
        etainf_eff = self._maskexclude(etainf, notsafe)

        # first marking pass: eta_infty over the whole domain
        mark, _, _ = self.fixedratemark(etainf_eff, theta, method)

        # term eta_d
        d = mesh.cell_dimension()
        gradsigmanorm = Function(DG0).interpolate(
            sqrt(inner(grad(sigmah), grad(sigmah)))
        )
        etad_ufl = (
            C1 * hT ** 2 * gradsigmanorm * CellVolume(mesh) ** (1.0 / d) * tactive
        )
        etad = Function(DG0, name="eta_d").interpolate(etad_ufl)
        etad_eff = self._maskexclude(etad, notsafe)

        # second marking pass: eta_d, but only when "quadrature dominates the
        # estimator" (NSV03 section 7.1)
        etainf_max = self._globalextreme(etainf_eff, minimum=False)
        etad_max = self._globalextreme(etad_eff, minimum=False)
        if etad_max > etadratio * etainf_max:
            markd, _, _ = self.fixedratemark(etad_eff, theta, method)
            mark = self.unionmarks(mark, markd)

        # the estimator Etilde_h of (7.1), accumulating each term in its own norm;
        # see doc string above.  The four sup-norm terms are maximized separately,
        # because (7.1) adds the global norms rather than maximizing their
        # elementwise sum eta_inf; and eta_d, being an L^d(Lambda_h) norm, is
        # accumulated as a sum of d-th powers.  Note the masking by safe= is a
        # marking policy and does not belong in a reliability bound, so the
        # unmasked fields are used here.
        Eh = (
            self._globalextreme(residterm, minimum=False)
            + self._globalextreme(blockgap, minimum=False)
            + self._globalextreme(bdryerr, minimum=False)
            + self._globalpnorm(etad, d)
        )
        return (mark, etainf, sigmah, Eh, etad)

    def nsv05mark(
        self,
        uh,
        lb,
        g,
        f_ufl,
        g_ufl,
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

        This is the successor of the NSV03 estimator implemented by nsvmark(); see that method for the earlier version, and see the comparison at the end of this doc string.

        The estimator E_h of Theorem 2.7 in NSV05 is

            E_h =   C_* |log h_min|^2 max_{z in N_h} eta_z    localized residual
                  + ||(chi - u_h)^+||_{inf; Omega}            localized obstacle approx.
                  + ||(u_h - chi)^+||_{inf; Lambda_h}
                  + ||g - I_h g||_{inf; partial Omega}        boundary datum approx.

        where the star-based residual indicator (2.15) is

            eta_z = h_z^2 ||(f - fhat_z) phi_z||_{inf; omega_z^+}
                    + h_z ||J_h phi_z||_{inf; gamma_z^+}.

        Here phi_z is the hat function at node z, omega_z = supp(phi_z) is the star, and gamma_z = Gamma \cap int(omega_z) is the set of interior facets lying in the *interior* of the star, which for simplices is exactly the set of interior facets which contain z.  (The other facets of the star lie in its boundary, and phi_z vanishes identically on them.)  Following section 3 of NSV05, the unknown constant and the logarithmic factor are replaced by the single practical constant C_* = 0.02 = C0.

        Meaning of the pieces:

          Full-contact nodes and set:  The set of full-contact nodes is

            C_h = {z in N_h : u_h = chi_h at z, and f <= 0 in omega_z, and J_h <= 0 on gamma_z},

          and the discrete full-contact set Omega_h^0 is the union of the elements all of whose vertices lie in C_h, with Omega_h^+ = Omega \setminus Omega_h^0 its (open) complement.  The residual indicator is restricted to omega_z^+ = omega_z cap Omega_h^+ and gamma_z^+ = gamma_z cap Omega_h^+, so it *vanishes identically* on Omega_h^0.  This is the "full localization" which distinguishes NSV05 from NSV03; it is why the mesh can stay coarse inside the contact set.  Note Omega_h^+ is open, so a facet lying in the boundary of Omega_h^0 is not in Gamma_h^+, i.e. a facet counts only if *both* its elements are outside the full-contact set.

          Element residual:  Only the *oscillation* f - fhat_z of the load enters, not f itself, where by (2.9)

            fhat_z = (1/2)(min_{omega_z^+} f + max_{omega_z^+} f)   if rho_z = 0,
                     0                                              otherwise,

          with rho_z = int_{Omega_h^+} f phi_z + int_{Gamma_h^+} J_h phi_z the restriction of the nodal multiplier s_z to Omega_h^+.  By (2.2), rho_z = s_z = 0 at every node which is not a full-contact node, so the test only bites inside the contact set.  Since it is an exact equality in the theory but a floating-point comparison here, rho_z is tested against rhotol times the local scale int |f| phi_z.

          Weight phi_z:  On the facet term the hat function is handled exactly; see _tracetonodemaxabs().  On the element term the bound phi_z <= 1 is used instead, which overestimates and so preserves reliability while giving the clean closed form

            ||f - fhat_z||_{inf; omega_z^+} = (1/2)(max_{omega_z^+} f - min_{omega_z^+} f)   if rho_z = 0,
                                              max_{omega_z^+} |f|                           otherwise,

          i.e. exactly half the oscillation of f over the star, computable from elementwise extremes of f.

          Obstacle approximation:  ||(chi - u_h)^+|| is assumed to be zero because we take chi = chi_h here and assert admissibility.  [<-- FIXME same limitation as nsvmark()]

          The "blocked gap" ||(u_h - chi)^+|| is restricted to the set Lambda_h of (2.20), which is the union of the stars omega_z over nodes z with s_z < 0, where z is an interior node or a full-contact boundary node.  Compare nsvmark(), which restricts the same quantity to {sigma_h < 0} elementwise, without dilating to stars.

          Boundary datum:  ||g - I_h g||_{inf; partial Omega} is computed with a formula which is correct if g is in CG4.  It is localized here as the elementwise sup over boundary-touching elements, which overestimates the sup over the boundary facets themselves.  (nsvmark() instead divides an assembled boundary integral by the cell volume, which does not have the units of a sup norm.)

        Marking.  E_h is a single number: a max over nodes plus global sup norms.  Localizing it to elements for marking purposes is not spelled out in NSV05, so we follow what section 7.1 of NSV03 does for its own estimator, and use

            eta_T = C0 max_{z a vertex of T} eta_z + ||(u_h - chi)^+||_{inf; Lambda_h cap T} + ||g - I_h g||_{inf; partial Omega cap T}.

        Under the default 'max' strategy, taking the max over the vertices of T is exactly equivalent to marking the whole star omega_z of every marked node z.  Note there is only *one* marking pass, in contrast to nsvmark(): NSV05's multiplier sigma_h is a functional defined by (2.3), built without mass lumping, so the quadrature estimator ||h^2 grad(sigma_h)||_{d; Lambda_h} of NSV03 and its separate marking loop have no counterpart here.

        There is no safe= argument, in contrast to nsvmark().  The two sign conditions defining C_h are a discrete version of safeactiveunmark()'s certificate sigma_psi = L(psi) - f > 0, so full localization already switches the estimator off where that method would certify it, while masking anywhere else would break the reliability of Theorem 2.7.

        Returns (mark, eta, sz, fullcontact, Eh), where eta is the DG0 elementwise estimator, sz is the CG1 nodal multiplier of (2.5), and fullcontact is the DG0 indicator of Omega_h^0.  Eh is the scalar estimator E_h of Theorem 2.7 itself, so it is the quantity which bounds ||u - u_h||_{0,inf;Omega}, and thus the right numerator for an effectivity index.  Its three terms are separate global sup norms, so each is maximized on its own; this makes Eh >= max_T eta_T, with equality only if the three maxima happen to fall on one element.  Note eta exists for marking, and is *not* a field whose l^2 norm means anything here.

        Summary of the differences from nsvmark() (= NSV03):
          * the indicator is star-based (per node z), not element-based;
          * the residual is switched off entirely on Omega_h^0, instead of having sigma_h subtracted from f on the discrete contact set;
          * the load enters only through its oscillation f - fhat_z;
          * there is no quadrature estimator and no second marking pass;
          * the blocked gap is restricted to a union of stars, not to elements;
          * one constant C0 = 0.02, versus C0 = 0.1 and C1 = 0.01.
        """
        # mesh quantities
        mesh = uh.function_space().mesh()
        CG1, DG0 = self.spaces(mesh)
        hT = project(CellSize(mesh), DG0)  # versus mesh.cell_sizes(), which is in CG1
        phi = TestFunction(CG1)
        v0 = TestFunction(DG0)

        # sample the (generally non-polynomial) load, then get its elementwise
        # extremes; note pages 188-189 in NSV03 regarding the use of DG7, and
        # see nsvmark() for why we drop to DG3 by default
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
            self._nodalactive(uh, lb)
            * conditional(fmaxstar <= ftol, 1.0, 0.0)
            * conditional(Jmaxgamma <= jtol, 1.0, 0.0)
        )

        # discrete full-contact set Omega_h^0 = union of elements whose vertices
        # are all in C_h, and its complement Omega_h^+
        fullcontact = self._elemextreme(Ch, minimum=True, defaultval=1.0)
        fullcontact.rename("Omega_h^0 (full contact)")
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
        # closed form of the oscillation term, and _tracetonodemaxabs() for the
        # exact treatment of phi_z on the facets
        hz = self._elemtonodeextreme(hT, CG1, minimum=False, defaultval=0.0)
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

        # admissibility check; FIXME removes the "(chi - u_h)^+" term from the
        #   estimator *only when* chi=lb is representable in u_h's space
        gaph = Function(CG1).interpolate(uh - lb)
        assert self._globalextreme(gaph, minimum=True) >= 0.0

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
            self._elemextreme(lamz, minimum=False, defaultval=0.0)
            * self._elemmaxabs(gaph)
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
        residterm = Function(DG0).interpolate(C0 * etaR)
        eta = Function(DG0, name="eta (NSV05)").interpolate(
            residterm + blockgap + bdryerr
        )

        # the estimator E_h of Theorem 2.7, whose three terms are separate global
        # sup norms, so each is maximized on its own rather than maximizing their
        # elementwise sum eta.  This is the quantity the reliability theorem bounds
        # ||u - u_h||_{0,inf;Omega} by, hence the right numerator for an
        # effectivity index; it is >= max_T eta_T.
        Eh = (
            self._globalextreme(residterm, minimum=False)
            + self._globalextreme(blockgap, minimum=False)
            + self._globalextreme(bdryerr, minimum=False)
        )

        mark, _, _ = self.fixedratemark(eta, theta, method)
        return (mark, eta, sz, fullcontact, Eh)

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

    def jaccard(self, active1, active2, submesh=False):
        """Compute the Jaccard metric from two element-wise DG0 active set indicators.  By definition, the Jaccard metric of two sets is
            J(S,T) = |S cap T| / |S cup T|,
        where |.| is area (measure) of the set.  Thus J(S,T) the ratio of the area (measure) of the intersection divided by that of the union.  The inputs are the indicator functions of the sets as DG0 functions.  In serial they can be on different meshes.  (In that case project()
        method is used to put them on active1's mesh.)  If submesh==True then active2 is assumed to live on a submesh of active1, so interpolate onto the active1 mesh will work correctly.  *Note that with submesh==True this function works in parallel.*"""
        # FIXME how to check that, when submesh==True, active2 is actually on a submesh of active1?
        a1DG0 = active1.function_space()
        a2DG0 = active2.function_space()
        mesh1 = a1DG0.mesh()
        mesh2 = a2DG0.mesh()
        if self.debug:
            _, DG01 = self.spaces(mesh1)
            assert a1DG0.ufl_element() == DG01.ufl_element()
            _, DG02 = self.spaces(mesh2)
            assert a2DG0.ufl_element() == DG02.ufl_element()
        if submesh == False and (mesh1.comm.size > 1 or mesh2.comm.size > 1):
            raise ValueError("jaccard(.., submesh=False) is not valid in parallel")
        if self.debug:
            for a in [active1, active2]:
                if len(a.dat.data_ro) > 0:
                    assert min(a.dat.data_ro) >= 0.0
                    assert max(a.dat.data_ro) <= 1.0
        if submesh:
            new2 = Function(a1DG0).interpolate(active2)
        else:
            new2 = Function(a1DG0).project(active2)
        AreaIntersection = assemble(new2 * active1 * dx(mesh1))
        AreaUnion = assemble((new2 + active1 - (new2 * active1)) * dx(mesh1))
        if AreaUnion <= 0.0:
            warnings.warn(
                "VIAMR.jaccard() called with two empty sets (AreaUnion <= 0.0); "
                "returning -1.0"
            )
            return -1.0
        return AreaIntersection / AreaUnion

    def jaccardUFL(self, active1, active2, qdegree=6):
        """Version of jaccard() for when active1 is a UFL expression.
        Uses high-degree quadrature.  Always valid in parallel."""
        a2DG0 = active2.function_space()
        mesh2 = a2DG0.mesh()
        if self.debug:
            _, DG02 = self.spaces(mesh2)
            assert a2DG0.ufl_element() == DG02.ufl_element()
        if self.debug:
            if len(active2.dat.data_ro) > 0:
                assert min(active2.dat.data_ro) >= 0.0
                assert max(active2.dat.data_ro) <= 1.0
        AreaIntersection = assemble(active1 * active2 * dx(mesh2, degree=qdegree))
        AreaUnion = assemble(
            (active2 + active1 - (active2 * active1)) * dx(mesh2, degree=qdegree)
        )
        if AreaUnion <= 0.0:
            warnings.warn(
                "VIAMR.jaccardUFL() called with two empty sets (AreaUnion <= 0.0); "
                "returning -1.0"
            )
            return -1.0
        return AreaIntersection / AreaUnion

    def hausdorff2D(self, E1, E2, densify=0.99):
        """Compute the (densified, approximate) Hausdorff distance between two planar
        edge-coordinate sets E1, E2, e.g. as returned by freeboundarygraph2D().
        densify is the shapely densify fraction in (0,1]: each segment is
        subdivided into 1/densify pieces before comparison, which turns the
        (fast but only locally-accurate) vertex-based Hausdorff distance into a
        global approximation.  Smaller values are more accurate but slower;
        see shapely.hausdorff_distance()."""
        if len(E1) == 0 or len(E2) == 0:
            warnings.warn(
                "VIAMR.hausdorff2D() called with an empty free-boundary edge set; "
                "returning None"
            )
            return None
        try:
            import shapely
        except ImportError:
            raise ImportError(
                "VIAMR.hausdorff2D() requires shapely; install it with 'pip install shapely'"
            )
        return shapely.hausdorff_distance(
            shapely.MultiLineString(E1), shapely.MultiLineString(E2), densify
        )

    def freeboundarygraph2D(self, uh, bound, boxside="lower"):
        """Compute the graph (vertices and edges) of the computed free boundary
        of a 2D unilateral obstacle problem with the given bound (boxside="lower"
        or "upper"; see _nodalactive()), as (x,y) coordinates.  Works for
        meshes with triangular or quadrilateral cells.  The free boundary
        vertices are those incident to both a bordering (partially-active)
        element and a fully-active element; see _elemborder() and
        elemactive().  The free boundary edges are the edges of bordering
        elements which connect two such vertices.

        Returns (coordsV, coordsE): coordsV is a list of [x,y] vertex
        coordinates, and coordsE is a list of [[x1,y1],[x2,y2]] edge
        coordinate pairs.  The latter is the format hausdorff2D() expects
        for its "edge sets".

        Only implemented for 2D meshes (raises ValueError otherwise).

        Correct in parallel provided the mesh was built with
        distribution_parameters=VIAMR.PARALLEL_OVERLAP (or overlap_type=
        (DistributedMeshOverlapType.VERTEX, n>=1)); see
        _checkparalleloverlap().

        If the free boundary is empty (e.g. uh==bound identically, or uh
        strictly on the inactive side everywhere) a warning is issued and
        empty lists are returned."""

        mesh = uh.function_space().mesh()
        if mesh.topological_dimension != 2:
            raise ValueError("freeboundarygraph2D() only supports 2D meshes")
        self._checkparalleloverlap(mesh)

        # basic mesh topology information
        nv = mesh.ufl_cell().num_vertices  # =3 for triangles, =4 for quadrilaterals
        # note CellVertexMap is DG0.dim() x (2*nv + 1) array; each row is cell closure
        CellVertexMap = mesh.topology.cell_closure
        plexelementlist = CellVertexMap[:, -1]  # DMPlex point (index) of each cell

        # Get lists of indices for active and border elements.  Include halo
        # (ghost) cells so free-boundary vertices/edges lying on a process boundaries
        # are visible to every rank.
        elemactive = self.elemactive(uh, bound, boxside=boxside)
        elemborder = self._elemborder(self._nodalactive(uh, bound, boxside=boxside))
        ActiveSetElementsIndices = np.where(elemactive.dat.data_ro_with_halos)[0]
        BorderElementsIndices = np.where(elemborder.dat.data_ro_with_halos)[0]

        # Vertices incident to a bordering / active-set cell.  The first nv columns
        # of CellVertexMap are always a cell's nv vertices, for any 2D cell type.
        BorderVertices = set()
        for cellIdx in BorderElementsIndices:
            BorderVertices.update(CellVertexMap[cellIdx][:nv])
        ActiveVertices = set()
        for cellIdx in ActiveSetElementsIndices:
            ActiveVertices.update(CellVertexMap[cellIdx][:nv])

        # free boundary = (active) \cap (border), as local DMPlex point
        # numbers.  In parallel a vertex/edge on a process boundary is
        # found independently by every rank that borders it; this is
        # resolved below by an allgather-and-deduplicate step below.
        FreeBoundaryVertices = BorderVertices.intersection(ActiveVertices)

        # Create an edge *set* for the FreeBoundaryVertices.  Use each bordering
        # cell's actual boundary edges, its DMPlex cone.  (For a triangle this equals
        # all pairs of vertices, but not for a quadrilateral.)
        dm = mesh.topology_dm
        EdgeSet = set()
        for j in BorderElementsIndices:
            k = plexelementlist[j]
            for edge in dm.getCone(k):
                v1, v2 = dm.getCone(edge)
                if v1 in FreeBoundaryVertices and v2 in FreeBoundaryVertices:
                    EdgeSet.add((min(v1, v2), max(v1, v2)))

        # Convert local DMPlex point numbers to physical coordinates.
        # NOTE: _vertex_numbering is a private Firedrake attribute (no
        # public equivalent as of this writing); a future Firedrake release
        # could rename or remove it without warning.
        coords = mesh.coordinates.dat.data_ro_with_halos
        vnum = mesh.topology._vertex_numbering
        coordsV = [tuple(coords[vnum.getOffset(v)]) for v in FreeBoundaryVertices]
        coordsE = [
            tuple(
                sorted(
                    (
                        tuple(coords[vnum.getOffset(v1)]),
                        tuple(coords[vnum.getOffset(v2)]),
                    )
                )
            )
            for v1, v2 in EdgeSet
        ]

        # Deduplicate across ranks using allgather().  Halo coordinate values are
        # exact copies of the owning rank's data (no arithmetic), so a shared
        # vertex/edge is bit-identical.  A plain set correctly merges
        # the per-rank (possibly-overlapping) contributions into a single global
        # graph, identical on every rank.
        if mesh.comm.size > 1:
            coordsV = set().union(*mesh.comm.allgather(coordsV))
            coordsE = set().union(*mesh.comm.allgather(coordsE))
        else:
            coordsV = set(coordsV)
            coordsE = set(coordsE)

        # return plain lists-of-lists
        if not coordsV:
            warnings.warn(
                "VIAMR.freeboundarygraph2D() found an empty free boundary; "
                "returning an empty graph"
            )
        return [list(v) for v in coordsV], [[list(e[0]), list(e[1])] for e in coordsE]

    def _filtermesh(self, mesh, indicator):
        """Return the submesh containing only the cells where the DG0 indicator
        is nonzero, via PETSc's DMPlex "transform_filter" transform."""
        return self._dmplextransform(mesh, "transform_filter", indicator=indicator)

    def safeactiveunmark(
        self,
        uh0,
        lb0,
        F_strong_fcn,
        psi_ufl,
        f_ufl,
        psi_mode="analytic",
        f_mode="analytic",
        pdegree=2,
        stricttol=1.0e-10,
    ):
        """Return a marking (DG0 field with {0,1} values) where 1 indicates a part
        of the current active set, defined by uh0 and lb0, in which it is safe
        *not* to mark.  This allows solver efficiency in problems where the
        operator is such that the interiors of active sets can avoid
        wasted-effort refinement.

        The strong (NCP) form of the unilateral obstacle problem is
          u >= psi,   L(u) - f >= 0,   (u - psi)(L(u) - f) = 0.
        In general we could write F(u,f)=L(u)-f; see below.  We assume here
        that L(u) has some kind of positive-definiteness or coercivity; this
        is not checked but see examples.

        On the (true) active set, u = psi, so on this set it reduces to
        requiring
          sigma_psi := L(psi) - f >= 0.
        This is a condition on the given data (psi,f) alone, independent of
        the discrete iterate uh0.

        The safety check in this method evaluates sigma_psi at higher degree
        than the current functions, but on the same mesh.  The idea is that
        if sigma_psi stays at or above a strictly-positive tolerance on a
        mesh cell then it is safe to leave unmarked.  Refining it cannot
        reveal a hidden inactive region, because the check already used
        higher-resolution data than the current mesh provides.  We require a
        strict positive margin
          sigma_psi > stricttol,
        because this is a *safety* certificate.  As approximation error is
        already inherent in psi_mode,f_mode="analytic" (see below),
        declaring an element safe only when it clears zero by a margin is
        the conservative choice.

        The sigma_psi check is only applied on VIAMR.thinelemactive()'s
        thinned active set, i.e. elements that are active *and* every
        neighbor is active too, which excludes the one-element-thick border
        adjacent to the discrete free boundary from ever being marked safe.
        This is the same "neighborhood of the active set" idea NSV03 uses
        (see nsvmark()'s tactive), applied here in the complementary
        direction.

        This method is O(N) work if N represents current mesh complexity,
        e.g. element count, but with a decent constant because of the use of
        higher-order elements.

        Inputs uh0, lb0 are the current discrete solution and obstacle.
        They are only used to determine the current thinned active set
        (self.thinelemactive(uh0, lb0)).

        F_strong_fcn is a function returning a UFL expression for the
        strong-form residual:
          res = F_strong_fcn(u, f)     # UFL for L(u) - f
        so that sigma_psi = F_strong_fcn(psi_ufl, f_ufl).  psi_ufl, f_ufl
        are the user's *exact* obstacle and source data (not the
        generally-coarser lb0), as UFL expressions.

        psi_mode="analytic" assumes psi_ufl is an exact analytic UFL
        expression, evaluable (including derivatives) to arbitrary
        precision.  The higher resolution used for the safety check is then
        obtained by p-refinement on the *current* mesh, using Bernstein
        polynomials.  Similarly f_mode="analytic" assumes f_ufl is exact
        analytic UFL.

        psi_mode and f_mode are independent, because in practice psi and f
        often come from unrelated sources.  (E.g. for glacier problems, psi
        is bed topography from a DEM while f is surface mass balance from a
        climate model, on a different grid.)

        FIXME A future psi_mode="data" or f_mode="data" would instead
        support psi and/or f given as (e.g. gridded/observational) data
        rather than analytic UFL.  This would need edge-jump-estimator
        machinery such as in brinactivemark(), rather than mere pointwise
        UFL differentiation, since second derivatives of piecewise-linear
        discrete data are weakly zero.  Not yet implemented.
        """
        if psi_mode != "analytic" or f_mode != "analytic":
            raise NotImplementedError(
                f"safeactiveunmark() with psi_mode='{psi_mode}', f_mode='{f_mode}' "
                "is not implemented; only psi_mode='analytic', f_mode='analytic' "
                "(exact UFL psi_ufl, f_ufl) is currently supported"
            )
        mesh = uh0.function_space().mesh()
        _, DG0 = self.spaces(mesh)
        # thinned active set: only elements that are active *and* whose
        # neighbors are all active too, i.e. elements not adjacent to the
        # current discrete free boundary; same NSV03 "neighborhood of the
        # active set" concept nsvmark() itself uses (there called tactive).
        thinactive0 = self.thinelemactive(uh0, lb0)

        # p-refined estimate of sigma_psi = L(psi) - f, sampled at higher resolution
        # than the current mesh; represented in the Bernstein basis so that its
        # elementwise coefficient extremes rigorously bound the interpolant's
        # range over each cell (convex hull property), not just its value at the
        # Lagrange nodes
        CGp = FunctionSpace(mesh, "CG", pdegree)
        psip = Function(CGp).interpolate(psi_ufl)
        sigma_ufl = F_strong_fcn(psip, f_ufl)
        Bp = FunctionSpace(mesh, "Bernstein", pdegree)
        sigma = Function(Bp).interpolate(sigma_ufl)

        # a thinned-active element is safe to unmark if sigma_psi stays
        # strictly above stricttol everywhere within it
        sigmamin = self._elemextreme(sigma, minimum=True, defaultval=PETSc.INFINITY)
        safe_ufl = thinactive0 * conditional(sigmamin > stricttol, 1.0, 0.0)
        return Function(DG0, name="mark (safeactiveunmark)").interpolate(safe_ufl)
