import warnings
import numpy as np
from firedrake import *


class SetDiagnosticsMixin:
    r"""Mixed into class VIAMR (see viamr.py): diagnostic and measurement methods
    for computed sets, namely jaccard(), hausdorff2D(), and
    freeboundarygraph2D().

    SetDiagnosticsMixin is not usable separately from VIAMR.  Specifically,
    freeboundarygraph2D() calls VIAMR.nodalactive(), VIAMR.elemactive(),
    VIAMR._elemborder(), and VIAMR._checkparalleloverlap(), and jaccard()
    uses self.debug and VIAMR.spaces().
    """

    def _checkDG0indicator(self, active):
        """Assert that active is a DG0 Function with values in [0,1]."""
        aDG0 = active.function_space()
        _, DG0 = self.spaces(aDG0.mesh())
        assert aDG0.ufl_element() == DG0.ufl_element()
        if len(active.dat.data_ro) > 0:
            assert min(active.dat.data_ro) >= 0.0
            assert max(active.dat.data_ro) <= 1.0

    def _checksubmeshmeasure(self, active, activeinterp, rtol=1.0e-8):
        """Check that interpolating a DG0 indicator onto a refinement of its own mesh
        preserves measures.  The argument activeinterp is the interpolation of the DG0
        indicator active onto a mesh mesh1.  Here mesh1 is assumed to refine mesh2, the
        mesh of active, meaning that each cell of mesh1 lies within a single cell of
        mesh2.  In that case activeinterp equals active almost everywhere, and therefore
        the two meshes have the same measure, and the set indicated by active also has the
        same measure as the set for activeinterp.  Both of these equalities are checked to
        relative tolerance rtol, and a ValueError is raised if either fails.  The converse
        does not hold; passing both checks does not prove that mesh1 refines mesh2.
        Applies to polygonal meshes.  The jaccard() method calls this when submesh==True,
        when it interpolates its second argument onto the mesh of its first."""
        mesh1 = activeinterp.function_space().mesh()
        mesh2 = active.function_space().mesh()
        area1 = assemble(Constant(1.0) * dx(mesh1))
        area2 = assemble(Constant(1.0) * dx(mesh2))
        if abs(area1 - area2) > rtol * max(area1, area2):
            raise ValueError(
                f"_checksubmeshmeasure(): the two meshes have different measures {area1} and {area2}"
            )
        set1 = assemble(activeinterp * dx(mesh1))
        set2 = assemble(active * dx(mesh2))
        if abs(set1 - set2) > rtol * area2:
            raise ValueError(
                f"_checksubmeshmeasure(): the active sets have different measures {set2} and {set1}"
            )

    def jaccard(self, active1, active2, submesh=False, qdegree=6, mesh=None):
        """Compute the Jaccard metric of two sets, for example two active sets.  By definition, the Jaccard metric J(S,T) of two sets is the ratio of the area (measure) of the intersection divided by that of the union:
            J(S,T) = |S cap T| / |S cup T|.
        Note that J(S,T) = J(T,S); it is a symmetric function.

        The sets S,T are the input variables active1, active2, each of which is an indicator function for the set, given either as a DG0 Function or as a UFL expression, in any combination.  If both arguments are UFL expressions then a mesh= keyword argument is required; otherwise mesh= is not allowed.  A UFL expression argument is integrated using quadrature of degree qdegree, so qdegree has no effect if both arguments are DG0 Functions.

        If both arguments are DG0 Functions then, in serial, they can be on different meshes.  In that case the project() method is used to put active2 on active1's mesh.

        If submesh==True then the mesh of active1 is assumed to be a refinement of the mesh of active2, meaning that each cell of the active1 mesh lies within a single cell of the active2 mesh.  Under that assumption interpolate() puts active2 onto the active1 mesh exactly, and it does so in parallel.  Regarding checking the submesh relation, no mesh object records such a relation between two separately-constructed meshes, so submesh==True is a promise by the caller.  However, _checksubmeshmeasure() verifies a necessary consequence of it.

        This method works in parallel if either submesh==True, or if either indicator argument is a UFL expression."""
        isfem1 = isinstance(active1, Function)
        isfem2 = isinstance(active2, Function)
        if isfem1 or isfem2:
            if mesh is not None:
                raise ValueError(
                    "jaccard(.., mesh=..) is only valid if both sets are UFL expressions"
                )
            imesh = (active1 if isfem1 else active2).function_space().mesh()
        else:
            if mesh is None:
                raise ValueError(
                    "jaccard() with two UFL expressions requires a mesh= argument"
                )
            imesh = mesh
        if self.debug:
            for a, isfem in [(active1, isfem1), (active2, isfem2)]:
                if isfem:
                    self._checkDG0indicator(a)
        if isfem1 and isfem2:
            mesh2 = active2.function_space().mesh()
            if submesh == False and (imesh.comm.size > 1 or mesh2.comm.size > 1):
                raise ValueError("jaccard(.., submesh=False) is not valid in parallel")
            a1DG0 = active1.function_space()
            if submesh:
                active2interp = Function(a1DG0).interpolate(active2)
                self._checksubmeshmeasure(active2, active2interp)
                active2 = active2interp
            else:
                active2 = Function(a1DG0).project(active2)
            dV = dx(imesh)
        else:
            if submesh:
                raise ValueError(
                    "jaccard(.., submesh=True) is only valid if both sets are DG0 Functions"
                )
            dV = dx(imesh, degree=qdegree)
        AreaIntersection = assemble(active1 * active2 * dV)
        AreaUnion = assemble((active1 + active2 - (active1 * active2)) * dV)
        if AreaUnion <= 0.0:
            warnings.warn(
                "VIAMR.jaccard() called with two empty sets (AreaUnion <= 0.0); "
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
        of a 2D unilateral obstacle problem with the given bound, which is a
        floor (uh >= bound) if boxside="lower" or a ceiling (uh <= bound) if
        boxside="upper", as (x,y) coordinates.  Works for
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
        bounds = (None, bound) if boxside == "upper" else (bound, None)
        elemactive = self.elemactive(uh, bounds)
        elemborder = self._elemborder(self.nodalactive(uh, bounds))
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
