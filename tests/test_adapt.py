from firedrake import *
from viamr import VIAMR, haveanimate
from test_basic import _get_ball_obstacle
import pytest

needsanimate = pytest.mark.skipif(
    not haveanimate, reason="animate import failed; adaptaveragedmetric() unavailable"
)


@needsanimate
def test_adapt_avm():
    mesh = RectangleMesh(6, 6, 2.0, 2.0, originX=-2.0, originY=-2.0)
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh)
    assert CG1.dim() == 49
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(_get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi, 0.0))
    amr.setmetricparameters(target_complexity=100, h_min=1.0e-4, h_max=1.0)
    rmesh = amr.adaptaveragedmetric(mesh, u, psi)
    rCG1, _ = amr.spaces(rmesh)
    assert rCG1.dim() > 80


@needsanimate
def test_adapt_avm_separated():
    mesh = RectangleMesh(5, 5, 2.0, 2.0, originX=-2.0, originY=-2.0)
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh)
    assert CG1.dim() == 36
    psi = Function(CG1).interpolate(Constant(0.0))
    (x, y) = SpatialCoordinate(mesh)
    r = sqrt(x ** 2 + y ** 2)
    u_ufl = conditional(r < 1, 1.0 + cos(pi * r), 0.0)
    uh = Function(CG1).interpolate(u_ufl)
    amr.setmetricparameters(target_complexity=100, h_min=1.0e-4, h_max=1.0)
    # only isotropic free-boundary metric
    fbmesh = amr.adaptaveragedmetric(mesh, uh, psi, gamma=1.0)
    fbCG1, _ = amr.spaces(fbmesh)
    assert fbCG1.dim() > 80
    # only hessian metric
    hmesh = amr.adaptaveragedmetric(mesh, uh, psi, gamma=0.0)
    hCG1, _ = amr.spaces(hmesh)
    assert hCG1.dim() > 80


@needsanimate
def test_adapt_avm_metric_only():
    # Exercises adaptaveragedmetric(..., metric=True), which returns the
    # RiemannianMetric itself instead of adapting the mesh, in all three
    # gamma branches (isotropic-only, hessian-only, averaged).
    from animate import RiemannianMetric

    mesh = RectangleMesh(5, 5, 2.0, 2.0, originX=-2.0, originY=-2.0)
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh)
    psi = Function(CG1).interpolate(Constant(0.0))
    (x, y) = SpatialCoordinate(mesh)
    r = sqrt(x**2 + y**2)
    uh = Function(CG1).interpolate(conditional(r < 1, 1.0 + cos(pi * r), 0.0))
    amr.setmetricparameters(target_complexity=100, h_min=1.0e-4, h_max=1.0)

    fbmetric = amr.adaptaveragedmetric(mesh, uh, psi, gamma=1.0, metric=True)
    assert isinstance(fbmetric, RiemannianMetric)

    hmetric = amr.adaptaveragedmetric(mesh, uh, psi, gamma=0.0, metric=True)
    assert isinstance(hmetric, RiemannianMetric)

    avgmetric = amr.adaptaveragedmetric(mesh, uh, psi, metric=True)  # gamma=0.5
    assert isinstance(avgmetric, RiemannianMetric)


@needsanimate
def test_adapt_avm_intersect():
    mesh = RectangleMesh(4, 4, 2.0, 2.0, originX=-2.0, originY=-2.0, diagonal="crossed")
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh)
    assert CG1.dim() == 41
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(_get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi, 0.0))
    amr.setmetricparameters(target_complexity=100, h_min=1.0e-4, h_max=1.0)
    rmesh = amr.adaptaveragedmetric(mesh, u, psi, intersect=True)
    rCG1, _ = amr.spaces(rmesh)
    assert rCG1.dim() > 80


if __name__ == "__main__":
    test_adapt_avm()
    test_adapt_avm_separated()
    test_adapt_avm_metric_only()
    test_adapt_avm_intersect()
