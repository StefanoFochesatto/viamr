import pytest
from firedrake import *
from viamr import VIAMR

from test_basic import (
    _get_netgen_mesh,
    _get_ball_obstacle,
    _freeboundarygraph2D_circle_case,
)
from test_refine import (
    _fixedrate_total_case,
    _nsvmark_nontrivial,
    _udomark_nontrivial,
    _udomark_nontrivial_lb,
)


class VIAMRRegression(VIAMR):
    def __init__(self):
        super().__init__()

    def _bfsneighbors(self, mesh, border, levels):
        """Element-wise multi-neighbor lookup using breadth-first search."""
        from collections import deque

        # build dictionary which maps each vertex in the mesh
        # to the cells that are incident to it
        vertex_to_cells = {}
        closure = mesh.topology.cell_closure  # cell to vertex connectivity
        # loop over all cells to populate the dictionary
        for i in range(mesh.num_cells()):
            # first three entries correspond to the vertices
            for vertex in closure[i][:3]:
                if vertex not in vertex_to_cells:
                    vertex_to_cells[vertex] = []
                vertex_to_cells[vertex].append(i)

        # loop over all cells to mark neighbors, and store the result in DG0
        result = Function(border.function_space(), name="nNeighbors")
        for i in range(mesh.num_cells()):
            if border.dat.data[i] == 1.0:
                # use BFS to find all cells within the specified number of levels
                queue = deque([(i, 0)])
                visited = set()
                while queue:
                    cell, level = queue.popleft()
                    if cell not in visited and level <= levels:
                        visited.add(cell)
                        result.dat.data[cell] = 1
                        for vertex in closure[cell][:3]:
                            for neighbor in vertex_to_cells[vertex]:
                                queue.append((neighbor, level + 1))
        return result

    def udomarkOLD(self, uh, lb, n=2):
        """Mark mesh using Unstructured Dilation Operator (UDO) algorithm."""
        mesh = uh.function_space().mesh()
        if mesh.comm.size > 1:
            raise ValueError("udomark() is not valid in parallel")
        # generate element-wise indicator for border set
        elemborder = self._elemborder(self._nodalactive(uh, lb))
        # _bfs_neighbors() constructs N^n(B) indicator
        return self._bfsneighbors(mesh, elemborder, n)


def test_refine_udo_parallelUDO():
    mesh1 = _get_netgen_mesh(TriHeight=0.1)
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh1)
    (x, y) = SpatialCoordinate(mesh1)
    psi = Function(CG1).interpolate(_get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi, 0.0))
    unorm0 = norm(u)
    # VTKFile(f"result_refine_0.pvd").write(u)
    mark1 = amr.udomark(u, psi)
    rmesh1 = mesh1.refine_marked_elements(mark1)  # netgen's refine method
    mesh2 = _get_netgen_mesh(TriHeight=0.1)
    CG1, _ = amr.spaces(mesh2)
    (x, y) = SpatialCoordinate(mesh1)
    psi = Function(CG1).interpolate(_get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi, 0.0))
    unorm0 = norm(u)
    # VTKFile("result_refine_0.pvd").write(u)
    mark2 = amr.udomark(u, psi)
    rmesh2 = mesh2.refine_marked_elements(mark2)  # netgen's refine method
    assert abs(amr.jaccard(mark1, mark2, submesh=True) - 1.0) < 1.0e-10
    r1CG1, _ = amr.spaces(rmesh1)
    r2CG1, _ = amr.spaces(rmesh2)
    assert r1CG1.dim() == r2CG1.dim()


@pytest.mark.parallel(nprocs=3)
def test_udomark_nontrivial_parallel():
    _udomark_nontrivial(VIAMR())


def test_udo_regression():
    # This test utilizes the the old implementation of UDO which builds the neighborhood of the free boundary using breadth first search,
    # as a regression test for the dmplex based implementation.
    amr = VIAMRRegression()
    u, lb = _udomark_nontrivial_lb(amr)
    markold = amr.udomarkOLD(u, lb, n=2)
    marknew = amr.udomark(u, lb, n=2)
    assert amr.jaccard(markold, marknew) == 1.0


@pytest.mark.parallel(nprocs=3)
def test_fixedrate_total_parallel():
    # 8 cells split unevenly (e.g. 2/3/3) across processes; confirms
    # tests/test_refine.py::_fixedrate_total_case() gives the same result
    # regardless of process count.
    _fixedrate_total_case(VIAMR(debug=True))


@pytest.mark.parallel(nprocs=3)
def test_nsvmark_nontrivial_parallel():
    # Confirms tests/test_refine.py::_nsvmark_nontrivial() -- including
    # the eta_d dominance gate -- gives the same result regardless of process
    # count.
    _nsvmark_nontrivial(VIAMR(debug=True))


@pytest.mark.parallel(nprocs=3)
def test_freeboundarygraph2D_circle_parallel():
    # Confirms tests/test_basic.py::_freeboundarygraph2D_circle_case() gives
    # the same free boundary graph (vertex/edge count) regardless of process
    # count, i.e. that the allgather-and-deduplicate step in
    # freeboundarygraph2D() correctly merges the per-rank contributions
    # along partition boundaries.
    coordsV, coordsE = _freeboundarygraph2D_circle_case(VIAMR(debug=True))
    assert len(coordsV) == 36
    assert len(coordsE) == 36


if __name__ == "__main__":
    test_refine_udo_parallelUDO()
    test_udo_regression()
    test_udomark_nontrivial_parallel()
    test_fixedrate_total_parallel()
    test_nsvmark_nontrivial_parallel()
    test_freeboundarygraph2D_circle_parallel()
