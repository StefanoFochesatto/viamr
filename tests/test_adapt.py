from firedrake import *
from viamr import VIAMR, haveanimate
from test_basic import _get_ball_obstacle
import pytest

needsanimate = pytest.mark.skipif(
    not haveanimate, reason="animate import failed; buildaveragedmetric() unavailable"
)


@needsanimate
def test_adapt_ama():
    import animate

    mesh = RectangleMesh(6, 6, 2.0, 2.0, originX=-2.0, originY=-2.0)
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh)
    assert CG1.dim() == 49
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(_get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi, 0.0))
    amr.setmetricparameters(target_complexity=100, h_min=1.0e-4, h_max=1.0)
    rmesh = animate.adapt(mesh, amr.buildaveragedmetric(mesh, u, (psi, None)))
    rCG1, _ = amr.spaces(rmesh)
    assert rCG1.dim() > 80


@needsanimate
def test_adapt_ama_separated():
    import animate

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
    fbmetric = amr.buildaveragedmetric(mesh, uh, (psi, None), gamma=1.0)
    fbmesh = animate.adapt(mesh, fbmetric)
    fbCG1, _ = amr.spaces(fbmesh)
    assert fbCG1.dim() > 80
    # only hessian metric
    hmetric = amr.buildaveragedmetric(mesh, uh, (psi, None), gamma=0.0)
    hmesh = animate.adapt(mesh, hmetric)
    hCG1, _ = amr.spaces(hmesh)
    assert hCG1.dim() > 80


@needsanimate
def test_buildaveragedmetric_gammas():
    # Exercises buildaveragedmetric(), which returns the RiemannianMetric
    # itself (never calling animate.adapt()), in all three gamma branches
    # (isotropic-only, hessian-only, averaged).
    from animate import RiemannianMetric

    mesh = RectangleMesh(5, 5, 2.0, 2.0, originX=-2.0, originY=-2.0)
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh)
    psi = Function(CG1).interpolate(Constant(0.0))
    (x, y) = SpatialCoordinate(mesh)
    r = sqrt(x**2 + y**2)
    uh = Function(CG1).interpolate(conditional(r < 1, 1.0 + cos(pi * r), 0.0))
    amr.setmetricparameters(target_complexity=100, h_min=1.0e-4, h_max=1.0)

    fbmetric = amr.buildaveragedmetric(mesh, uh, (psi, None), gamma=1.0)
    assert isinstance(fbmetric, RiemannianMetric)

    hmetric = amr.buildaveragedmetric(mesh, uh, (psi, None), gamma=0.0)
    assert isinstance(hmetric, RiemannianMetric)

    avgmetric = amr.buildaveragedmetric(mesh, uh, (psi, None))  # gamma=0.5
    assert isinstance(avgmetric, RiemannianMetric)


@needsanimate
def test_buildaveragedmetric_bilateral_strip():
    # uh jumps from lb=0 to ub=1 across a one-cell strip, so every node is
    # active against one bound or the other; smoothing the union indicator
    # would give s == 1 and no metric refinement at the strip
    mesh = UnitSquareMesh(10, 10)
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh)
    (x, y) = SpatialCoordinate(mesh)
    lb = Function(CG1).interpolate(Constant(0.0))
    ub = Function(CG1).interpolate(Constant(1.0))
    uh = Function(CG1).interpolate(conditional(x < 0.55, 0.0, 1.0))
    assert min(amr.nodalactive(uh, (lb, ub)).dat.data_ro) == 1.0  # all nodes active
    amr.setmetricparameters(target_complexity=100, h_min=1.0e-4, h_max=1.0)
    fbmetric = amr.buildaveragedmetric(mesh, uh, (lb, ub), gamma=1.0)
    trace = Function(CG1).interpolate(tr(fbmetric)).dat.data_ro
    xnode = Function(CG1).interpolate(x).dat.data_ro
    trstrip = max(trace[abs(xnode - 0.55) < 0.1])  # nodes at x=0.5, 0.6
    trfar = max(trace[xnode < 0.2])
    assert trstrip > 10.0 * trfar


@needsanimate
def test_adapt_ama_intersect():
    import animate

    mesh = RectangleMesh(4, 4, 2.0, 2.0, originX=-2.0, originY=-2.0, diagonal="crossed")
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh)
    assert CG1.dim() == 41
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(_get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi, 0.0))
    amr.setmetricparameters(target_complexity=100, h_min=1.0e-4, h_max=1.0)
    rmesh = animate.adapt(mesh, amr.buildaveragedmetric(mesh, u, (psi, None), intersect=True))
    rCG1, _ = amr.spaces(rmesh)
    assert rCG1.dim() > 80


if __name__ == "__main__":
    test_adapt_ama()
    test_adapt_ama_separated()
    test_buildaveragedmetric_gammas()
    test_buildaveragedmetric_bilateral_strip()
    test_adapt_ama_intersect()
