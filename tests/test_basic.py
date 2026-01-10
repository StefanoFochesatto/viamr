import numpy as np
from firedrake import *
from viamr import VIAMR


def _get_netgen_mesh(TriHeight=0.4, width=2):
    import netgen
    from netgen.geom2d import SplineGeometry

    geo = SplineGeometry()
    geo.AddRectangle(
        p1=(-1 * width, -1 * width), p2=(1 * width, 1 * width), bc="rectangle"
    )
    ngmsh = None
    ngmsh = geo.GenerateMesh(maxh=TriHeight)
    return Mesh(
        ngmsh,
        distribution_parameters={
            "partition": True,
            "overlap_type": (DistributedMeshOverlapType.VERTEX, 1),
        },
    )


def _get_ball_obstacle(x, y):
    r = sqrt(x * x + y * y)
    r0 = 0.9
    psi0 = np.sqrt(1.0 - r0 * r0)
    dpsi0 = -r0 / psi0
    return conditional(le(r, r0), sqrt(1.0 - r * r), psi0 + dpsi0 * (r - r0))


def test_spaces_sizes():
    mesh = _get_netgen_mesh(TriHeight=1.2)
    assert mesh.num_cells() == 24
    amr = VIAMR(debug=True)
    CG1, DG0 = amr.spaces(mesh)
    nv, ne, hmin, hmax = amr.meshsizes(mesh)
    assert nv == CG1.dim() == 19
    assert ne == DG0.dim() == 24
    assert 1.0 < hmin < 1.2
    assert 1.9 < hmax < 2.0


def test_mark_none():
    mesh = _get_netgen_mesh(TriHeight=1.2)
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh)
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(_get_ball_obstacle(x, y))
    mark = amr.udomark(psi, psi)  # all active
    assert norm(mark, "L1") == 0.0
    mark = amr.vcdmark(psi, psi)  # all active
    assert norm(mark, "L1") == 0.0
    lift = Function(CG1).interpolate(psi + 1.0)
    mark = amr.udomark(lift, psi)  # all inactive
    assert norm(mark, "L1") == 0.0
    mark = amr.vcdmark(lift, psi)  # all inactive
    assert norm(mark, "L1") == 0.0


def test_unionmarks():
    mesh = RectangleMesh(
        5, 5, 2.0, 2.0, originX=-2.0, originY=-2.0
    )  # Firedrake utility mesh, not netgen
    amr = VIAMR(debug=True)
    _, DG0 = amr.spaces(mesh)
    (x, y) = SpatialCoordinate(mesh)
    rightmark = Function(DG0).interpolate(conditional(x > 0.0, 1.0, 0.0))
    discmark = Function(DG0).interpolate(_get_ball_obstacle(x, y) > 0.0)
    mark = amr.unionmarks(rightmark, discmark)
    assert abs(assemble(mark * dx) - 9.92) < 1.0e-10  # union of marked area


def test_overlapping_jaccard():
    mesh = _get_netgen_mesh(TriHeight=1.2)
    amr = VIAMR(debug=True)
    _, DG0 = amr.spaces(mesh)
    x, _ = SpatialCoordinate(mesh)
    right = conditional(x > 0, 1, 0)
    active1 = Function(DG0).interpolate(right)  # right half active
    active2 = Function(DG0).interpolate(right)  # same; full overlap
    assert abs(amr.jaccard(active1, active2) - 1.0) < 1.0e-10


def test_nonoverlapping_jaccard():
    mesh = _get_netgen_mesh()
    amr = VIAMR(debug=True)
    _, DG0 = amr.spaces(mesh)
    x, _ = SpatialCoordinate(mesh)
    right = conditional(x > 0, 1, 0)
    farleft = conditional(x < -0.5, 1, 0)
    active1 = Function(DG0).interpolate(right)
    active2 = Function(DG0).interpolate(farleft)  # no overlap
    assert abs(amr.jaccard(active1, active2) - 0.0) < 1.0e-10


def test_symmetry_jaccard():
    mesh = _get_netgen_mesh()
    amr = VIAMR(debug=True)
    _, DG0 = amr.spaces(mesh)
    x, _ = SpatialCoordinate(mesh)
    right = conditional(x > 0, 1, 0)
    more = conditional(x < 1, 1, 0)
    active1 = Function(DG0).interpolate(right)
    active2 = Function(DG0).interpolate(more)
    assert abs(amr.jaccard(active1, active2) - amr.jaccard(active2, active1)) < 1.0e-10


def test_jaccard_submesh_uniform():
    mesh = UnitSquareMesh(5, 4)
    amr = VIAMR(debug=True)
    _, DG0 = amr.spaces(mesh)
    x, _ = SpatialCoordinate(mesh)
    mark = Function(DG0).interpolate(conditional(x < 0.5, 1.0, 0.0))
    rmesh = amr.refinemarkedelements(mesh, mark, isUniform=True)
    rx, _ = SpatialCoordinate(mesh)
    rmark = Function(DG0).interpolate(conditional(x < 0.7, 1.0, 0.0))
    assert amr.jaccard(mark, rmark, submesh=True) == amr.jaccard(mark, rmark)


def test_third_jaccard_ufl():
    mesh = UnitSquareMesh(4, 4)
    mesh = RectangleMesh(4, 4, Lx=1.0, Ly=1.0, originX=-1.0, originY=-1.0)
    amr = VIAMR(debug=True)
    _, DG0 = amr.spaces(mesh)
    x, y = SpatialCoordinate(mesh)
    active1 = conditional(y > 0, 1, 0)  # top half as UFL
    right = conditional(x > 0, 1, 0)
    active2 = Function(DG0).interpolate(right)  # right half as DG0
    assert abs(amr.jaccardUFL(active1, active2) - 1.0 / 3.0) < 1.0e-10


def test_overlapping_and_nonoverlapping_hausdorff():
    # to have free boundaries line up with conditional statements
    mesh = RectangleMesh(10, 10, 1, 1)
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh)
    x, y = SpatialCoordinate(mesh)
    sol1 = Function(CG1).interpolate(Constant(1.0))
    lb = Function(CG1).interpolate(conditional(x <= 0.2, 1, 0))
    _, E1 = amr.freeboundarygraph(sol1, lb)
    assert amr.hausdorff(E1, E1) == 0
    lb2 = Function(CG1).interpolate(conditional(x <= 0.4, 1, 0))
    _, E2 = amr.freeboundarygraph(sol1, lb2)
    assert amr.hausdorff(E1, E2) == 0.2


def test_elemmaxabs():
    mesh = UnitSquareMesh(2, 1)
    x, y = SpatialCoordinate(mesh)
    CG1 = FunctionSpace(mesh, "CG", 1)
    f = Function(CG1).interpolate(x * y)
    eamax = VIAMR(debug=True)._elemmaxabs(f)
    #VTKFile("foo.pvd").write(f, amax)
    assert eamax.function_space().ufl_element() == FiniteElement("DG", triangle, 0)
    diff = eamax.dat.data_ro - np.array([1.0, 0.5, 0.5, 0.0])
    assert np.linalg.norm(diff) == 0.0


def test_elemmin():
    mesh = UnitCubeMesh(1, 1, 2)
    x, y, z = SpatialCoordinate(mesh)
    CG1 = FunctionSpace(mesh, "CG", 1)
    f = Function(CG1).interpolate(x + y + z - 1.0)
    amr = VIAMR(debug=True)
    emin = amr._elemextreme(f, minimum=True, absolute=False, defaultval=1000.0)
    vals = np.array([-1 for j in range(6)] + [-0.5 for j in range(6)])
    assert emin.function_space().ufl_element() == FiniteElement("DG", tetrahedron, 0)
    assert np.linalg.norm(emin.dat.data_ro - vals) == 0.0


if __name__ == "__main__":
    test_spaces_sizes()
    test_mark_none()
    test_unionmarks()
    test_overlapping_jaccard()
    test_nonoverlapping_jaccard()
    test_symmetry_jaccard()
    test_jaccard_submesh_uniform()
    test_third_jaccard_ufl()
    test_overlapping_and_nonoverlapping_hausdorff()
    test_elemmaxabs()
    test_elemmin()
