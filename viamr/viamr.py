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

haveanimate = True
try:
    import animate
except ImportError:
    haveanimate = False


class VIAMR(OptionsManager):
    r"""A VIAMR object manages adaptive mesh refinement (AMR) for a Firedrake variational inequality (VI) solver.

    Central notions behind this class:
      * Like a PDE AMR method, refinement in the inactive set should be guided by
        an a posteriori estimator.
      * Refinement near the free boundary is compatible with the goals of free
        boundary models, even if it does not directly reduce the norm error of the solution.
        E.g. a purpose of solving glacier problems is to know which land is glaciated.
      * For some problems, refinement in the active set is worthwhile, but for some
        it is wasted effort.

    The prominent public API of the VIAMR class consists of:

      udomark(), vcdmark():  2 marking methods which target the computed free boundary

      gradreinactivemark(), brinactivemark():  2 classical a posterior error indicator marking methods applied in the computed inactive set.  the latter implements methods from
      Babushka & Rheinboldt (1978) and the weighted extension from Bernardi & Verfurth (2000)

      nsvmark():  mark using the "practical estimator" from Nochetto, Siebert, & Veeser (2003) = NSV03

      refinemarkedelements():  a method which calls PETSc for skeleton-based-refinement (SBR)

      adaptaveragedmetric():  a method which does metric-based mesh adaptation by combining an anisotropic metric with a free-boundary targeted isotropic metric

      eleminactive():  element markings for the computed inactive set

      elemactive(), thinelemactive():  two versions of element markings for computed active sets

      unionmark():  a method for combining existing marks

      lowerboundcelldiameter():  unmark elements with cell diameters below a minimum cell diameter

      jaccard(), jaccardUFL():  computation of the Jaccard similarity index for two active sets

    Some default calls to the major marking methods and refinement methods are:

    .. code-block:: python3

      amr = VIAMR()
      mark = amr.udomark(uh, lb)                     # free-boundary targeted marking method
      mark = amr.vcdmark(uh, lb)                     # same, but based on diffusion
      imark = amr.gradrecinactivemark(uh, lb)        # classical gradient recovery in inactive set
      imark = amr.brinactivemark(uh, lb, res_ufl)    # classical BR78 in inactive set
      imark = amr.brinactivemark(uh, lb, res_ufl, Z=Z)   # weighted (BV00) in inactive set
      mark = amr.unionmarks(mark, imark)             # mark according to two methods above
      mark, _, _, _ = amr.nsvmark(uh, lb, g, f_ufl, g_ufl)  # method from NSV03
      rmesh = amr.refinemarkelements(mesh, mark)     # calls PETSc DMPlexTransform for SBR
      rmesh = amr.adaptaveragedmetric(mesh, uh, lb)  # use animate for metric-based adaptation

    Regarding the arguments: uh is a computed VI solution, lb=psi is the lower bound (obstacle), res_ufl is a UFL expression for the residual (applicable in the inactive set), mark is an element marking in DG0 (Definition 4.2 in paper), and rmesh is a refined or adapted mesh.

    Regarding the refinemarkedelements(), compare refine_marked_elements() from NetGen/ngspetsc.

    There are also some utility methods: spaces(), meshsizes(), meshreport(), checkadmissible(), and countmark().  Other methods starting with an underscore are (roughly) intended to be private to the VIAMR class.

    Limitations:
      * Functions which do not work in parallel: 1. jaccard() with submesh=False, 2. hausdorff(), and 3. freeboundarygraph2D().
      * Functions which give different results depending on the number of processes: 1. vcdmark() and 2. adaptaveragedmetric().
      * Functions which only work for 2D meshs: 1. freeboundarygraph2D().
    """

    PARALLEL_OVERLAP = {
        "partition": True,
        "overlap_type": (DistributedMeshOverlapType.VERTEX, 1),
    }
    """distribution_parameters value needed for udomark()/thinelemactive() (and
    therefore nsvmark()) to give correct results in parallel; see
    _checkparalleloverlap().  Usage:
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

    def scalarrange(self, w):
        """Utility function to return the range of a generic scalar field.  Correct in parallel."""
        return self._globalextreme(w, minimum=True), self._globalextreme(w, minimum=False)

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

    def checkadmissible(self, uh, bound, strict=False, upper=False):
        """Utility function to check admissibility or strict admissibility of uh.  Returns True if uh >= bound (upper=False) or uh <= bound (upper=True)."""
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

    def _checkuhlb(self, uh, lb):
        """Debug-mode validation shared by the unilateral (lower bound) obstacle
        problem indicator methods: checks that uh is a Function, lb is a Function
        or Constant, and uh is admissible with respect to lb.  No-op unless
        self.debug is True."""
        if self.debug:
            assert isinstance(uh, Function), "input uh must be of class Function"
            islb = isinstance(lb, Function) or isinstance(lb, Constant)
            assert islb, "input lb must be of class Function or Constant"
            assert self.checkadmissible(uh, lb)

    def _checkparalleloverlap(self, mesh):
        """Raise ValueError if mesh is distributed across multiple processes
        without sufficient vertex overlap.  udomark() and thinelemactive() walk
        DMPlex vertex stars across partition boundaries (via getTransitiveClosure()),
        which requires overlap_type=(DistributedMeshOverlapType.VERTEX, n) with
        n >= 1 for correct results.  Build the mesh with
        distribution_parameters=VIAMR.PARALLEL_OVERLAP to satisfy this; see
        tests/test_parallel.py::test_parallel_udo."""
        if mesh.comm.size > 1:
            dp = mesh._distribution_parameters
            if dp["overlap_type"][0].name != "VERTEX" or dp["overlap_type"][1] < 1:
                raise ValueError(
                    "udomark()/thinelemactive() in parallel require mesh "
                    "distribution_parameters=VIAMR.PARALLEL_OVERLAP "
                    "on mesh initialization (or overlap_type=(VERTEX, n>=1))"
                )

    def _nodalactive(self, uh, lb):
        """Compute nodal active set indicator in same function space as uh.  Only implemented for unilateral (lower bound) obstacle problems.  The nodal active set is
          {x in N(V): |u(x) - lb(x)| < activetol}
        where N(V) is the nodal set for V = uh.function_space().  Active nodes get value 1.0."""
        self._checkuhlb(uh, lb)
        z = Function(uh.function_space(), name="Nodal Active")
        z.interpolate(conditional(abs(uh - lb) < self.activetol, 1.0, 0.0))
        return z

    def elemactive(self, uh, lb):
        """Compute an element active set indicator in DG0.  Active elements get value 1.0.  Only implemented for unilateral (lower bound) obstacle problems.  Elements are marked active if the DG0 degree of freedom for that element is active, within activetol, so use with caution if z is not in CG1."""
        self._checkuhlb(uh, lb)
        _, DG0 = self.spaces(uh.function_space().mesh())
        z = Function(DG0, name="Element Active")
        z.interpolate(conditional(abs(uh - lb) < self.activetol, 1.0, 0.0))
        return z

    def eleminactive(self, uh, lb, strong=False):
        """Compute an element inactive set indicator in DG0.  Inactive elements get value 1.0.  Only implemented for unilateral (lower bound) obstacle problems.  By default, elements are marked inactive if their DG0 degree of freedom is inactive (by activetol).

        If strong=True then an element is only marked as inactive if all degrees of freedom of the function uh-lb exceed activetol.  That is, a cell is "strongly" inactive if all of its original dofs are inactive."""
        self._checkuhlb(uh, lb)
        if strong:
            # note uh > lb is equivalent to v > 0 ... actually we use activetol
            v = Function(uh.function_space()).interpolate(uh - lb)
            # z is in DG0 and contains min of v over each cell's dofs
            z = self._elemextreme(v, minimum=True, defaultval=PETSc.INFINITY)
            z.interpolate(conditional(z > self.activetol, 1.0, 0.0))
            z.rename("Element Inactive")
        else:
            _, DG0 = self.spaces(uh.function_space().mesh())
            z = Function(DG0, name="Element Inactive")
            z.interpolate(conditional(abs(uh - lb) < self.activetol, 0.0, 1.0))
        return z

    def thinelemactive(self, uh, lb):
        """Compute element active set indicator into DG0, but "thinned".  In contrast to elemactive(), a cell is marked as active only if it *and its neighboring cells* are active.  The test for active is based on testing at the DG0 degree of freedom, and according to activetol. Returns a DG0 element-wise indicator, with active elements having value 1.
        The implementation is inspired by VIAMR.udomark().  The neighbor elements of *inactive* elements are found, and they are effectively removed from the active element indicator.
        The note about constant arity at https://op2.github.io/PyOP2/concepts.html suggests that this operation, and presumably VIAMR.udomark() also, cannot be done with PyOP2.
        """
        inactive = self.eleminactive(uh, lb)
        mesh = uh.function_space().mesh()
        self._checkparalleloverlap(mesh)
        _, DG0 = self.spaces(mesh)
        dm = mesh.topology_dm
        # map from firedrake mesh indices to DMPlex element indices (-1 = 2 = elements):
        plexelementlist = mesh.cell_closure[:, -1]
        # map back:
        # (Is there a better way to do this in dmcommon?)
        dm2fd = np.argsort(plexelementlist)
        # get DMPlex element indices of inactive cells using firedrake indices
        inactivecells = [
            plexelementlist[k]
            for k, value in enumerate(inactive.dat.data_ro_with_halos)
            if value == 1.0
        ]
        # vertex closure: indices of vertices which are incident to an inactive
        #   element, then flatten and remove duplicates
        d = mesh.cell_dimension()
        incvertices = [dm.getTransitiveClosure(j)[0][-d - 1 :] for j in inactivecells]
        incvertices = np.unique(np.ravel(incvertices))
        # star: indices of all elements which are incident to the incvertices
        #   note that getTransitiveClosure() with useCone=False gives the star
        #   note that the number of elements incident to a vertex is not predictable
        #   then flatten and remove duplicates
        kmin, kmax = dm.getHeightStratum(0)[:2]  # range for element indices
        neighborindices = []
        for j in incvertices:
            star = dm.getTransitiveClosure(j, useCone=False)[0]
            # FIXME is np.where needed?
            mark = np.where((star >= kmin) & (star < kmax))
            neighborindices.extend(star[mark])
        neighborindices = np.unique(np.ravel(neighborindices))
        # generate DG0 thin element active indicator by zeroing-out all neighbors
        # of inactive cells
        z = Function(DG0).interpolate(Constant(1.0))  # mark *all* cells 1.0
        for j in neighborindices:
            # parallel communication *here*:
            z.dat.data_wo_with_halos[dm2fd[j]] = 0.0  # remove inactive etc.
        return z

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

    def udomark(self, uh, lb, n=2, restrict=None):
        """Mark mesh using Unstructured Dilation Operator (UDO) algorithm.  The algorithm
        first computes an element-wise indicator for the free boundary.  Then the elements
        which neighbor free-boundary elements are added, and so on iteratively.
        The input n gives the number of levels to expand the initial element border.  The
        output is an element-wise marking for those elements near the free boundary which
        should be refined.
        Tuning advice:  Increase n to mark more elements near the free boundary, but
        on simple examples even n=1 may suffice."""

        # get mesh and border mark; added flag for restriction
        if restrict is not None:
            meshInit = uh.function_space().mesh()
            dInit = meshInit.cell_dimension()
            dmInit = meshInit.topology_dm
            if restrict == "active":  # restrict to active set plus border
                indicator = Function(FunctionSpace(meshInit, "DG", 0)).interpolate(
                    self.elemactive(uh, lb)
                    + self._elemborder(self._nodalactive(uh, lb))
                )
            elif restrict == "inactive":
                # restrict to inactive set, which contains border already
                indicator = self.eleminactive(uh, lb)
            else:
                raise ValueError(
                    f"unknown restrict='{restrict}'; must be 'active', 'inactive', or None"
                )
            mesh = self._filtermesh(meshInit, indicator)
            _, DG0 = self.spaces(mesh)
            # Use nodal active set indicator to make an initial DG0 element border
            # indicator. This is now on a restricted domain so allow_missing_dofs=True
            border = Function(DG0).interpolate(
                self._elemborder(self._nodalactive(uh, lb)), allow_missing_dofs=True
            )
        else:
            mesh = uh.function_space().mesh()
            _, DG0 = self.spaces(mesh)
            # Use nodal active set indicator to make an initial DG0 element border
            # indicator.
            border = self._elemborder(self._nodalactive(uh, lb))

        # get DMPlex
        self._checkparalleloverlap(mesh)
        d = mesh.cell_dimension()
        dm = mesh.topology_dm
        # FIXME Experiment implementation in cython, DMLabel to mark accumulation, dmplex with only vertex and cell connectivity to save memory

        # Find range of indices for element stratum
        kmin, kmax = dm.getHeightStratum(0)[:2]

        # need map from DMPlex to firedrake indices
        # (Is there a better way to do this in dmcommon?)
        plexelementlist = mesh.cell_closure[:, -1]
        dm2fd = np.argsort(plexelementlist)

        # main loop: expand element border out to n levels, using only DMPlex indices
        #   (index convention:  i for levels, j for nodes/vertices, k for elements)
        for i in range(n):
            # Pull DMPlex border element indices using dmplex cell indices
            borderindices = [
                plexelementlist[k]
                for k, value in enumerate(border.dat.data_ro_with_halos)
                if value != 0
            ]

            # closure: Pull indices of all vertices which are incident
            # to some border element, then flatten and remove duplicates.
            incidentVertices = [
                dm.getTransitiveClosure(k)[0][-d - 1 :] for k in borderindices
            ]
            incidentVertices = np.unique(np.ravel(incidentVertices))

            # star: Pull indices of all elements which are incident to the
            # incidentVertices.  Note that getTransitiveClosure() with useCone=False
            # gives the star, and that the number of elements incident to a vertex
            # is not predictable.  Then flatten and remove duplicates.
            neighborindices = []
            for j in incidentVertices:
                star = dm.getTransitiveClosure(j, useCone=False)[0]
                mark = np.where((star >= kmin) & (star < kmax))
                neighborindices.extend(star[mark])
            neighborindices = np.unique(np.ravel(neighborindices))

            # re-generate DG0 element border indicator by adding neighbors
            border = Function(DG0).interpolate(Constant(0.0))
            for k in neighborindices:
                # parallel communication *here*:
                border.dat.data_wo_with_halos[dm2fd[k]] = 1

        return Function(DG0, name="mark (udomark)").interpolate(
            border, allow_missing_dofs=True
        )

    def vcdmark(
        self,
        uh,
        lb,
        bracket=[0.2, 0.8],
        coefficient=0.5,
        returnSmooth=False,
        directsolver=False,
        vcdsolveriters=4,
        printsolvertime=False,
    ):
        """Mark mesh using Variable Coefficient Diffusion (VCD) algorithm.
        The algorithm computes a nodal active set indicator and then
        diffuses it, using a variable mesh-sized based coefficient.  Diffusion
        is by solving a single backward Euler time step for the corresponding
        time-dependent diffusion equation.  Thresholding for the middle
        values of this field marks only those elements which are close to the
        free boundary.  The output is an element-wise marking for which elements,
        near the free boundary, should be refined.
        Tuning advice:  Generally the bracket [a,b] should be adjusted as follows:
          * lower a from default 0.2 to mark more elements in/near *inactive* set
          * raise b from default 0.8 to mark more elements in/near *active* set"""

        # Compute nodal active set indicator within some tolerance
        mesh = uh.function_space().mesh()
        CG1, DG0 = self.spaces(mesh)
        nu = self._nodalactive(uh, lb)

        # Diffuse according to square of cell diameter: D = C h^2.  The nodal
        # active indicator gives the initial field u0.  Solve one backward
        # Euler time-step using a linear solver.
        w = TrialFunction(CG1)
        v = TestFunction(CG1)
        h = CellDiameter(mesh)
        a = w * v * dx + coefficient * h ** 2 * inner(grad(w), grad(v)) * dx
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
        if printsolvertime:
            start = time.perf_counter()
        solve(a == L, u, solver_parameters=sp, options_prefix="viamr_vcd")
        if printsolvertime:
            end = time.perf_counter()
            PETSc.Sys.Print(
                f"VIAMR INFO  vcdmark() solver time = {end - start:.6f} seconds"
            )

        if returnSmooth:
            return u

        # apply thresholding and interpolate into DG0
        mark = Function(DG0, name="mark (vcdmark)")
        mark.interpolate(
            conditional(u > bracket[0], conditional(u < bracket[1], 1, 0), 0)
        )
        return mark

    def _fixedrate(self, eta, theta, method):
        """Marks elements according to the values of estimator eta in DG0 and a threshold which depends on the scalar theta.  The number of elements marked is an increasing function of theta.  The default 'max' strategy marks all elements with eta greater than
          ethresh = theta * max eta
        The 'total' strategy sorts all elements (globally, across processes) by decreasing eta value.  Then the threshold
          ethresh = eta(index)
        equals the eta value where theta times the total sum of eta is equal to the sum of the eta values above ethresh.  (I.e. theta gives the fraction of the total eta sum.)  The 'total' strategy is the refine-only version of the "fixed-rate" strategy, with X=theta and Y=0, described in section 4.2 of
          W. Bangerth & R. Rannacher (2003).  Adaptive Finite Element Methods for
          Differential Equations, Springer Basel.
        Both strategies give identical results regardless of the number of processes.  Note the 'total' strategy allgathers eta onto every process, so it does not scale to very large meshes on very large process counts."""

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
                raise ValueError("unknown method for VIAMR._fixedrate()")
            total_error_est = sqrt(eta_.dot(eta_))  # l^2 norm of eta as Vec

        DG0 = eta.function_space()
        mark = Function(DG0, name="mark (_fixedrate)").interpolate(
            conditional(gt(eta, ethresh), 1.0, 0.0)
        )
        return mark, ethresh, total_error_est

    def gradrecinactivemark(self, uh, lb, theta=0.5, method="max"):
        """Return marking within the computed inactive set by using an
        a posteriori gradient-recovery error indicator.  See Chapter 4 of
          M. Ainsworth & J. T. Oden (2000).  A Posteriori Error Estimation in
          Finite Element Analysis, John Wiley & Sons, Inc., New York."""
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
        eta = Function(DG0).interpolate(sqrt(eta_sq))  # eta from eta^2
        # restrict grad recovery eta to inactive set
        imark = self.eleminactive(uh, lb)
        ieta = Function(DG0, name="eta on inactive set").interpolate(eta * imark)
        # compute mark in inactive set
        mark, _, total_error_est = self._fixedrate(ieta, theta, method)
        return (mark, ieta, total_error_est)

    def brinactivemark(self, uh, lb, res, theta=0.5, method="max", alpha=None):
        """Return marking within the computed inactive set by using the
        a posteriori Babuška-Rheinboldt (1978) residual error indicator,
        or a weighted version of it.

        The primary inputs are the current solution uh, the obstacle lb (to restrict
        to the inactive set), and the residual res as a UFL expression.

        The output BR indicator eta is computed as a function in DG0.  We call
        VIAMR._fixedrate() to mark using eta and a threshold theta.
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
                + 0.5 * inner(mue_p * jump(alpha * grad(uh), n) ** 2, w("+")) * dS(degree=3)
                + 0.5 * inner(mue_m * jump(alpha * grad(uh), n) ** 2, w("-")) * dS(degree=3)
            )

        # each cell needs an independent 1x1 solve, so Jacobi is an exact preconditioner
        sp = {"mat_type": "matfree", "ksp_type": "richardson", "pc_type": "jacobi"}
        solve(G == 0, eta_sq, solver_parameters=sp)
        eta = Function(DG0).interpolate(sqrt(eta_sq))  # eta from eta^2
        # restrict BR eta to inactive set; strong=True means all dofs must be inactive to get imark=1
        imark = self.eleminactive(uh, lb, strong=True)
        ieta = Function(DG0, name="eta on inactive set").interpolate(eta * imark)
        mark, _, total_error_est = self._fixedrate(ieta, theta, method)
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
        Cfb=1.0,
        fdegree=3,
    ):
        """Compute marking on entire domain according to the local 'practical estimator' from NSV03:
            Nochetto, R. H., Siebert, K. G., & Veeser, A. (2003). Pointwise a posteriori error control for elliptic obstacle problems. Numerische Mathematik, 95(1), 163-195.
        The main formula (7.1) in NSV03 is the following quantity computed on each triangle T in the mesh:
            eta_infty =
                  C_0 h_T^2 ||R_infty||_infty                      [term 1]
                + ||(chi - u_h)^+||_infty                          [term 2]
                + C_fb 1_{sigma_h > 0} * ||(u_h - chi)^+||_infty   [term 3]
                + ||g - I_h g||_{infty; partial Omega cap T}       [term 4]
        Meaning:
          term 1:  Estimates the residual relevant to the VI problem; see below for the R_infty formula, which uses the discrete residual sigma_h below.  C_0=0.1 is used by NSV03.
          term 2:  Assumed to be zero because we take chi=chi_h here and assert strict admissibility.  [<-- FIXME could be improved]
          term 3:  This "blocked gap" is gap = u_h - chi_h, but blocked according to the simplest discrete residual sigma_h, computed below.  (Note that we assert sigma_h > -dualtol below.)  Here we add coefficient C_fb, which is 1.0 in NSV03.  Increasing C_fb will generate refinement near the free boundary.
          * Term 4 estimates the boundary interpolation error, and we use a formula which is correct if g is in CG4.
        FIXME also eta_d?
        """
        # mesh quantities
        mesh = uh.function_space().mesh()
        CG1, DG0 = self.spaces(mesh)
        n = FacetNormal(mesh)
        hT = project(CellSize(mesh), DG0)  # versus mesh.cell_sizes(), which is in CG1

        # compute residual sigmah in CG1 following section 2.1 of NSV03, page 169,
        #   but use opposite sign convention so sigmah >= 0.   complementarity is
        #   uh >= 0,  sigmah >= 0,  uh sigmah = 0  because psih=0
        # step 1: residual as a cofunction
        phi = TestFunction(CG1)
        res = assemble((inner(grad(uh), grad(phi)) - f_ufl * phi) * dx)  # cofunction
        # step 2: create cofunction with values  s_i = int_Omega phi_i dx  for *all*
        #   nodes i; we *do not* want riesz_representation() here
        scale = assemble(phi * dx)
        # step 3: apply scale, that is, divide by s_i
        sigmah = Function(CG1, name="sigma_h (residual)")
        sigmah.dat.data[:] = res.dat.data_ro / scale.dat.data_ro  # divide numpy arrays
        # FIXME use general g below
        # step 4: zero out boundary values because all boundary nodes are inactive *in this
        #    example*, but page 169 addresses cases where boundary nodes are active
        #    plan to use?:  inner(grad(uh), n) * omegah * ds
        DirichletBC(CG1, Constant(0.0), "on_boundary").apply(sigmah)

        # check dual admissiblity (up to tolerance)
        assert self._globalextreme(sigmah, minimum=True) >= -dualtol

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
        # jumpu is in DG0
        with PETSc.Log.Event(
            "nsvmark()_calls_assemble3"
        ):  # FIXME riesz_rep..() expensive because solve()
            jumpu = assemble(jump(grad(uh), n) * v0("-") * dS).riesz_representation()
        tactive = self.thinelemactive(uh, lb)
        X_ufl = abs(f_ufl + tactive * sigmah)
        # note pages 188-189 in NSV03 regarding use of DG7, to deal with the fact
        # that f_ufl is generally not in CG1:
        #     "For terms involving non-polynomial data, the maximum norm is
        #      approximated by evaluating element point-values at the Lagrange
        #      nodes for 7th order polynomials.""
        # BUT using DG7 this way is really slow because it is so big, so we drop
        # to DG3 by default; see .dim() printing below
        DGf = FunctionSpace(mesh, "DG", fdegree)
        Rinf = Function(DGf).interpolate((abs(jumpu) / hT) + X_ufl)
        Rinf = self._elemmaxabs(Rinf)

        # admissibility check; FIXME removes term 2 "(\chi - u_h)_+" from
        #   the estimator *only when* \chi=lb is representable in u_h's space
        gaph = Function(CG1).interpolate(uh - lb)
        assert self._globalextreme(gaph, minimum=True) >= 0.0

        # finally compute eta_inf; see doc string above for formula
        # note that blockgap is nonzero in same cells as UDO n=0 fmark
        blockgap_ufl = conditional(sigmah > dualtol, self._elemmaxabs(gaph), 0.0)
        blockgap = Function(DG0).interpolate(blockgap_ufl)
        CG4 = FunctionSpace(mesh, "CG", 4)
        # adg = abs delta g; in DG0 and defined over all of Omega
        adg = self._elemmaxabs(Function(CG4).interpolate(g_ufl - g))
        # bdryerr is a DG0 function, but only nonzero along boundary
        # note restriction is implicit when using ds (versus dS)
        with PETSc.Log.Event(
            "nsvmark()_calls_assemble4"
        ):  # FIXME riesz_rep..() expensive because solve()
            bdryerr = assemble(adg * v0 * ds).riesz_representation()
        etainf_ufl = C0 * hT ** 2 * Rinf + Cfb * blockgap + bdryerr
        etainf = Function(DG0, name="eta_inf").interpolate(etainf_ufl)

        # DIAGNOSTIC. note DG3.dim()=10*DG0.dim() exactly, while DG7 is
        # about 40 times, and CG4.dim() ~ 9*DG0.dim()
        # PETSc.Sys.Print(f"VIAMR INFO for nsvmark():  DG0.dim()={DG0.dim()}, DG{fdegree}.dim()={DGf.dim()}, CG4.dim()={CG4.dim()}")

        # compute mark in whole domain
        mark, _, total_error_est = self._fixedrate(etainf, theta, method)
        return (mark, etainf, sigmah, total_error_est)

    def refinemarkedelements(self, mesh, indicator, isUniform=False):
        """Call PETSc DMPlex routines to do skeleton-based refinement
        (SBR; Plaza & Carey, 2000).  This version works in parallel,
        but only in 2D; see TODO in
          https://petsc.org/release/src/dm/impls/plex/transform/impls/refine/sbr/plexrefsbr.c.html.
        See https://petsc.org/release/overview/plex_transform_table/
        and associated links.  Compare this method to Netgen's
        refine_marked_elements() which also does SBR, in 2D or 3D.
        Optionally does uniform refinement."""

        # section for DG0 indicator
        tdim = mesh.topological_dimension
        entity_dofs = np.zeros(tdim + 1, dtype=IntType)
        entity_dofs[-1] = 1
        indicatorSect, _ = dmcommon.create_section(mesh, entity_dofs)

        # create a DMPlex adaptation label to mark cells for refinement
        dm = mesh.topology_dm
        dm.createLabel("markviamr")
        adaptLabel = dm.getLabel("markviamr")
        adaptLabel.setDefaultValue(0)

        # dmcommon provides a python binding for this operation of setting
        # the label given an indicator function data array
        if self.debug:
            _, DG0 = self.spaces(mesh)
            assert indicator.function_space().ufl_element() == DG0.ufl_element()
        dmcommon.mark_points_with_function_array(
            dm, indicatorSect, 0, indicator.dat.data_with_halos, adaptLabel, 1
        )

        # augment/override options database to indicate type of refinement
        # (For now the only way to set the active label with petsc4py uses
        # PETSc.Options() because DMPlexTransformSetActive() has no binding.)
        opts = PETSc.Options()
        if isUniform:
            opts["dm_plex_transform_type"] = "refine_regular"
        else:
            opts["dm_plex_transform_active"] = "markviamr"
            opts["dm_plex_transform_type"] = "refine_sbr"

        # create a DMPlexTransform object to apply the refinement
        dmTransform = PETSc.DMPlexTransform().create(comm=mesh.comm)
        dmTransform.setDM(dm)
        dmTransform.setFromOptions()
        dmTransform.setUp()
        dmAdapt = dmTransform.apply(dm)

        # labels are no longer needed
        dmAdapt.removeLabel("markviamr")
        dm.removeLabel("markviamr")
        dmTransform.destroy()  # do we need this?

        # remove other labels to stop further distribution in mesh()
        # (Koki's suggestion)
        dmAdapt.removeLabel("pyop2_core")
        dmAdapt.removeLabel("pyop2_owned")
        dmAdapt.removeLabel("pyop2_ghost")

        # create a new mesh from the adapted dm
        dp = mesh._distribution_parameters  # original parameters
        refinedmesh = Mesh(dmAdapt, distribution_parameters=dp, comm=mesh.comm)
        opts["dm_plex_transform_type"] = "refine_regular"  # reset
        return refinedmesh

    def setmetricparameters(self, **kwargs):
        """Set self.metricparameters, used by adaptaveragedmetric(), from keyword
        arguments target_complexity (target number of mesh nodes; default 3000.0),
        h_min (minimum allowed edge length; default 0.0), and h_max (maximum
        allowed edge length; default no limit).  Must be called before
        adaptaveragedmetric()."""
        tc = kwargs.pop("target_complexity", 3000.0)
        hmin = kwargs.pop("h_min", 0.0)
        hmax = kwargs.pop("h_max", PETSc.INFINITY)
        if self.debug:
            assert np.isreal(tc) and tc > 0
            assert np.isreal(hmin) and hmin >= 0
            assert np.isreal(hmax) and hmax > 0
        mp = {
            "target_complexity": tc,  # target number of nodes
            "p": 2.0,  # normalisation order
            "h_min": hmin,  # minimum allowed edge length
            "h_max": hmax,  # maximum allowed edge length
        }
        self.metricparameters = {"dm_plex_metric": mp}
        return None

    def _isotropicfbmetric(self, mesh, uh, lb, CG1, P1tensor):
        """Construct a normalized free-boundary isotropic metric from abs(grad(s)),
        where s is the (smooth) output of vcdmark().  Compare "L2" option in
        animate.compute_isotropic_metric(); here we already have a P1 indicator.)"""
        assert haveanimate, "animate import failed, method unavailable"
        s = self.vcdmark(uh, lb, returnSmooth=True)
        maggrads = Function(CG1).interpolate(sqrt(dot(grad(s), grad(s))))
        VIMetric = animate.RiemannianMetric(P1tensor)
        VIMetric.set_parameters(self.metricparameters)
        VIMetric.interpolate(maggrads * Identity(mesh.topological_dimension))
        VIMetric.normalise()  # normalize *before* averaging
        return VIMetric

    def _hessianmetric(self, mesh, uh, P1tensor):
        """Construct a normalized metric from the Hessian of uh.  This is motivated
        by the interpolation error formula."""
        assert haveanimate, "animate import failed, method unavailable"
        hessmetric = animate.RiemannianMetric(P1tensor)
        hessmetric.set_parameters(self.metricparameters)
        # re method: default "mixed_L2" is more expensive
        hessmetric.compute_hessian(uh, method="L2")
        hessmetric.normalise()  # normalize *before* averaging
        return hessmetric

    def adaptaveragedmetric(
        self, mesh, uh, lb, gamma=0.50, intersect=False, metric=False
    ):
        """From the solution uh, of an obstacle problem with obstacle lb, constructs both
        an anisotropic Hessian-based metric and an isotropic metric computed from the
        magnitude of the gradient of the smoothed VCD indicator.  These metrics are averaged
        (linearly-combined) using gamma:
          M(x) = gamma (isotropic) + (1-gamma) (anisotropic)
        The result M(x) is an anisotropic metric which is free-boundary aware.  The mesh
        is adapted, according to the metric parameters, by calling the Animate
        mesh-adaptation library, which applies the Pragmatic mesher by default.
        If intersect=True then does Animate intersect (instead of gamma average).
        If metric=True then returns the metric itself, not the mesh."""

        assert haveanimate, "animate import failed, method unavailable"
        assert (
            self.metricparameters is not None
        ), "call setmetricparameters() before calling adaptaveragedmetric()"

        # Get the function spaces
        CG1, _ = self.spaces(mesh)
        P1tensor = TensorFunctionSpace(mesh, "CG", 1)

        # Branch on gamma to avoid unecesarry computation of both metrics
        if gamma == 1:  # isotropic metric only case
            VIMetric = self._isotropicfbmetric(mesh, uh, lb, CG1, P1tensor)
            if metric:
                return VIMetric
            return animate.adapt(mesh, VIMetric)

        elif gamma == 0:  # hessian metric only case
            hessmetric = self._hessianmetric(mesh, uh, P1tensor)
            if metric:
                return hessmetric
            return animate.adapt(mesh, hessmetric)

        else:
            # Default case where both metrics are computed
            VIMetric = self._isotropicfbmetric(mesh, uh, lb, CG1, P1tensor)
            hessmetric = self._hessianmetric(mesh, uh, P1tensor)
            # average or intersect
            if intersect:
                VIMetric.intersect(hessmetric)
            else:
                VIMetric.average(hessmetric, weights=[gamma, 1.0 - gamma])
            if metric:
                return VIMetric
            return animate.adapt(mesh, VIMetric)

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

    def hausdorff(self, E1, E2, densify=0.99):
        """Compute the (densified, approximate) Hausdorff distance between two
        edge sets E1, E2, e.g. as returned by freeboundarygraph2D(type="coords").
        densify is the shapely densify fraction in (0,1]: each segment is
        subdivided into 1/densify pieces before comparison, which turns the
        (fast but only locally-accurate) vertex-based Hausdorff distance into a
        global approximation.  Smaller values are more accurate but slower;
        see shapely.hausdorff_distance()."""
        if len(E1) == 0 or len(E2) == 0:
            warnings.warn(
                "VIAMR.hausdorff() called with an empty free-boundary edge set; "
                "returning None"
            )
            return None
        try:
            import shapely
        except ImportError:
            raise ImportError(
                "VIAMR.hausdorff() requires shapely; install it with 'pip install shapely'"
            )
        return shapely.hausdorff_distance(
            shapely.MultiLineString(E1), shapely.MultiLineString(E2), densify
        )

    def freeboundarygraph2D(self, uh, lb, type="coords"):
        """Compute the graph (vertices and edges) of the computed free boundary
        of a 2D obstacle problem.  Works for meshes with triangular or
        quadrilateral cells.  The free boundary vertices are those incident to
        both a bordering (partially-active) element and a fully-active element;
        see _elemborder() and elemactive().  The free boundary edges are the
        edges of bordering elements which connect two such vertices.

        The type argument selects the output format:
          type="dm":     (FreeBoundaryVertices, EdgeSet), as sets of DMPlex
                          point numbers for vertices, and (v1,v2) tuples of the
                          same for edges
          type="fd":     (fdV, fdE), the same information using Firedrake's
                          global vertex numbering
          type="coords": (coordsV, coordsE) (the default), the same
                          information as physical (x,y) coordinate pairs; this
                          is the format hausdorff() expects

        Only implemented for 2D meshes (raises ValueError otherwise), and only
        valid in serial (raises ValueError if called in parallel).  If the
        free boundary is empty (e.g. uh == lb identically, or uh is strictly
        above lb everywhere) a warning is issued and empty structures, of the
        type requested, are returned."""
        if type not in ("dm", "fd", "coords"):
            raise ValueError(
                f"unknown type '{type}'; must be 'dm', 'fd', or 'coords'"
            )
        mesh = uh.function_space().mesh()
        if mesh.comm.size > 1:
            raise ValueError("freeboundarygraph2D() is not valid in parallel")
        if mesh.topological_dimension != 2:
            raise ValueError("freeboundarygraph2D() only supports 2D meshes")

        nv = mesh.ufl_cell().num_vertices  # =3 for triangles, =4 for quadrilaterals
        CellVertexMap = mesh.topology.cell_closure  # DG0.dim() x (2*nv + 1) array; each row is cell closure
        plexelementlist = CellVertexMap[:, -1]  # DMPlex point number (index) of each cell
        dm = mesh.topology_dm

        # Get lists of indices for active and border elements
        elemactive = self.elemactive(uh, lb)
        elemborder = self._elemborder(self._nodalactive(uh, lb))
        ActiveSetElementsIndices = np.where(elemactive.dat.data_ro)[0]
        BorderElementsIndices = np.where(elemborder.dat.data_ro)[0]

        # Vertices incident to a bordering / active-set cell.  The first nv
        # columns of CellVertexMap are always a cell's nv vertices, for any 2D cell type
        BorderVertices = set()
        for cellIdx in BorderElementsIndices:
            BorderVertices.update(CellVertexMap[cellIdx][:nv])
        ActiveVertices = set()
        for cellIdx in ActiveSetElementsIndices:
            ActiveVertices.update(CellVertexMap[cellIdx][:nv])

        # free boundary = (active) \cap (border)
        FreeBoundaryVertices = BorderVertices.intersection(ActiveVertices)
        if not FreeBoundaryVertices:
            warnings.warn(
                "VIAMR.freeboundarygraph2D() found an empty free boundary; "
                "returning an empty graph"
            )
            return (set(), set()) if type == "dm" else ([], [])

        # Create an edge set for the FreeBoundaryVertices.  Use each bordering
        # cell's actual boundary edges, its DMPlex cone.  (For a triangle this equals
        # all pairs of vertices, but not for a quadrilateral.)
        EdgeSet = set()
        for j in BorderElementsIndices:
            k = plexelementlist[j]
            for edge in dm.getCone(k):
                v1, v2 = dm.getCone(edge)
                if v1 in FreeBoundaryVertices and v2 in FreeBoundaryVertices:
                    EdgeSet.add((min(v1, v2), max(v1, v2)))

        if type == "dm":
            return FreeBoundaryVertices, EdgeSet
        else:
            # NOTE: _vertex_numbering is a private Firedrake attribute (no
            # public equivalent as of this writing); a future Firedrake
            # release could rename or remove it without warning.
            fdV = [
                mesh.topology._vertex_numbering.getOffset(vertex)
                for vertex in FreeBoundaryVertices
            ]
            fdE = [
                [
                    mesh.topology._vertex_numbering.getOffset(edge[0]),
                    mesh.topology._vertex_numbering.getOffset(edge[1]),
                ]
                for edge in EdgeSet
            ]
            if type == "fd":
                return fdV, fdE
            else:  # returning coordinates of vertices and edges for the free boundary
                coords = mesh.coordinates.dat.data_ro_with_halos
                coordsV = [coords[vertex] for vertex in fdV]
                coordsE = [
                    [
                        [coords[edge[0]][0], coords[edge[0]][1]],
                        [coords[edge[1]][0], coords[edge[1]][1]],
                    ]
                    for edge in fdE
                ]
                return coordsV, coordsE

    def _filtermesh(self, mesh, indicator):

        # Create Section for DG0 indicator
        tdim = mesh.topological_dimension
        entity_dofs = np.zeros(tdim + 1, dtype=IntType)
        entity_dofs[:] = 0
        entity_dofs[-1] = 1
        indicatorSect, _ = dmcommon.create_section(mesh, entity_dofs)

        # Pull Plex from mesh
        dm = mesh.topology_dm

        # Create a filter label
        dm.createLabel("filter")
        adaptLabel = dm.getLabel("filter")
        adaptLabel.setDefaultValue(0)

        # Set label values with function array
        dmcommon.mark_points_with_function_array(
            dm, indicatorSect, 0, indicator.dat.data_with_halos, adaptLabel, 1
        )

        # Create a DMPlexTransform object to apply the filter
        opts = PETSc.Options()

        opts["dm_plex_transform_active"] = "filter"
        opts["dm_plex_transform_type"] = "transform_filter"
        dmTransform = PETSc.DMPlexTransform().create(comm=mesh.comm)
        dmTransform.setDM(dm)

        # For now the only way to set the active label with petsc4py is with PETSc.Options() (DMPlexTransformSetActive() has no binding)
        dmTransform.setFromOptions()
        dmTransform.setUp()
        dmAdapt = dmTransform.apply(dm)

        # Labels are no longer needed we need to call destroy on them.
        dmAdapt.removeLabel("filter")
        dm.removeLabel("filter")
        dmTransform.destroy()

        # Remove labels to stop further distribution in mesh()
        # dm.distributeSetDefault(False) <- Matt's suggestion
        dmAdapt.removeLabel("pyop2_core")
        dmAdapt.removeLabel("pyop2_owned")
        dmAdapt.removeLabel("pyop2_ghost")
        # ^ Koki's suggestion

        # Pull distribution parameters from original dm
        distParams = mesh._distribution_parameters

        # Create a new mesh from the adapted dm
        refinedmesh = Mesh(dmAdapt, distribution_parameters=distParams, comm=mesh.comm)

        # Set transform type back to regular refinemenet
        opts["dm_plex_transform_type"] = "refine_regular"

        return refinedmesh
