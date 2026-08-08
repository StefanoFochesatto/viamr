from firedrake import *
from viamr import VIAMR

from test_basic import _get_netgen_mesh, _get_ball_obstacle


def test_overrefine_udo():
    mesh = _get_netgen_mesh(TriHeight=1.2)
    amr = VIAMR(debug=True)
    CG1, DG0 = amr.spaces(mesh)
    assert CG1.dim() == 19
    assert DG0.dim() == 24
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(_get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi, 0.0))
    unorm0 = norm(u)
    mark = amr.udomark(u, psi)
    assert amr.countmark(mark) == DG0.dim()  # everything gets marked
    # VTKFile("result_overrefine_udo.pvd").write(u, psi, mark)
    rmesh = mesh.refine_marked_elements(mark)  # netgen's refine method
    rCG1, _ = amr.spaces(rmesh)
    assert rCG1.dim() == 61
    rV = FunctionSpace(rmesh, "CG", 1)
    ru = Function(rV).interpolate(u)  # cross-mesh interpolation
    assert abs(norm(ru) - unorm0) < 1.0e-10  # ... should be conservative


def test_finer_udo():
    mesh = _get_netgen_mesh(TriHeight=0.5)  # note smaller triangles
    amr = VIAMR(debug=True)
    CG1, DG0 = amr.spaces(mesh)
    assert CG1.dim() == 86
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(_get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi, 0.0))
    unorm0 = norm(u)
    mark = amr.udomark(u, psi, n=1)  # note n=1 (default is n=2)
    assert amr.countmark(mark) == 90
    assert amr.countmark(mark) < DG0.dim()
    # VTKFile("result_finer_udo.pvd").write(u, psi, mark)
    rmesh = mesh.refine_marked_elements(mark)  # netgen's refine method
    rCG1, _ = amr.spaces(rmesh)
    assert rCG1.dim() == 256
    rV = FunctionSpace(rmesh, "CG", 1)
    ru = Function(rV).interpolate(u)  # cross-mesh interpolation
    # VTKFile("result_rfiner_udo.pvd").write(ru)
    assert abs(norm(ru) - unorm0) < 1.0e-10  # ... should be conservative


def test_refine_vcd():
    mesh = _get_netgen_mesh(TriHeight=1.2)
    amr = VIAMR(debug=True)
    CG1, DG0 = amr.spaces(mesh)
    assert CG1.dim() == 19
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(_get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi, 0.0))
    unorm0 = norm(u)
    mark = amr.vcdmark(u, psi)
    assert amr.countmark(mark) == 13
    assert amr.countmark(mark) < DG0.dim()  # not everything gets marked
    # VTKFile("result_refine_vcd.pvd").write(u, psi, mark)
    rmesh = mesh.refine_marked_elements(mark)  # netgen's refine method
    rCG1, _ = amr.spaces(rmesh)
    assert rCG1.dim() == 49
    rV = FunctionSpace(rmesh, "CG", 1)
    ru = Function(rV).interpolate(u)  # cross-mesh interpolation
    assert abs(norm(ru) - unorm0) < 1.0e-10  # ... should be conservative


def test_refine_vcd_petscsbr():
    mesh = _get_netgen_mesh(TriHeight=1.2)
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh)
    assert CG1.dim() == 19
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(_get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi, 0.0))
    unorm0 = norm(u)
    mark = amr.vcdmark(u, psi)
    rmesh = amr.refinemarkedelements(mesh, mark)  # PETSc's skeleton-based refine method
    rCG1, _ = amr.spaces(rmesh)
    assert rCG1.dim() == 49
    rV = FunctionSpace(rmesh, "CG", 1)
    ru = Function(rV).interpolate(u)  # cross-mesh interpolation
    assert abs(norm(ru) - unorm0) < 1.0e-10  # ... should be conservative


def test_refine_vcd_firedrake_petscsbr():
    mesh = RectangleMesh(
        5, 5, 2.0, 2.0, originX=-2.0, originY=-2.0
    )  # Firedrake utility mesh, not netgen
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh)
    assert CG1.dim() == 36
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(_get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi, 0.0))
    unorm0 = norm(u)
    mark = amr.vcdmark(u, psi)
    rmesh = amr.refinemarkedelements(mesh, mark)  # PETSc's skeleton-based refine method
    rCG1, _ = amr.spaces(rmesh)
    # check that direct solver gets same result
    markDS = amr.vcdmark(u, psi, directsolver=True)
    rmeshDS = amr.refinemarkedelements(mesh, markDS)
    rCG1DS, _ = amr.spaces(rmeshDS)
    assert rCG1DS.dim() == rCG1.dim() == 73
    assert errornorm(mark, markDS) < 1.0e-14
    # check conservative cross-mesh interpolation
    rV = FunctionSpace(rmesh, "CG", 1)
    ru = Function(rV).interpolate(u)
    assert abs(norm(ru) - unorm0) < 1.0e-10


def test_refine_gr():
    mesh = RectangleMesh(6, 6, 2.0, 2.0, originX=-2.0, originY=-2.0, diagonal="crossed")
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh)
    assert CG1.dim() == 85
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(_get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi + 1.0, psi))
    imark, _, _ = amr.gradrecinactivemark(u, psi, theta=0.5)
    rmesh = amr.refinemarkedelements(mesh, imark)
    rCG1, _ = amr.spaces(rmesh)
    assert rCG1.dim() == 165


def test_refine_br():
    mesh = RectangleMesh(8, 8, 2.0, 2.0, originX=-2.0, originY=-2.0)
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh)
    assert CG1.dim() == 81
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(_get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi + 1.0, psi))
    residual = -div(grad(u))  # largest near circle psi==0
    (imark, _, _) = amr.brinactivemark(u, psi, residual, theta=0.5)
    rmesh = amr.refinemarkedelements(mesh, imark)
    # VTKFile(f"result_brinactivemark_refined.pvd").write(rmesh)
    rCG1, _ = amr.spaces(rmesh)
    assert rCG1.dim() == 109


def test_refine_br_total():
    mesh = RectangleMesh(8, 8, 2.0, 2.0, originX=-2.0, originY=-2.0)
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh)
    assert CG1.dim() == 81
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(_get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi + 1.0, psi))
    residual = -div(grad(u))  # largest near circle psi==0
    (imark, _, _) = amr.brinactivemark(u, psi, residual, theta=0.8, method="total")
    rmesh = amr.refinemarkedelements(mesh, imark)
    rCG1, _ = amr.spaces(rmesh)
    assert rCG1.dim() == 99


def test_refine_br_weighted():
    mesh = RectangleMesh(8, 8, 2.0, 2.0, originX=-2.0, originY=-2.0)
    amr = VIAMR(debug=True)
    CG1, _ = amr.spaces(mesh)
    assert CG1.dim() == 81
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(_get_ball_obstacle(x, y))
    u = Function(CG1).interpolate(conditional(psi > 0.0, psi + 1.0, psi))
    residual = -div(grad(u))  # largest near circle psi==0
    # non-constant, positive-everywhere coefficient
    alpha = x * x + y * y + 1.0
    (imark, eta, tot_eta) = amr.brinactivemark(u, psi, residual, theta=0.5, alpha=alpha)
    # weighting by a non-constant alpha should differ from alpha=None estimator
    (imark0, eta0, tot_eta0) = amr.brinactivemark(u, psi, residual, theta=0.5)
    assert errornorm(eta, eta0) > 1.0e-10
    assert abs(tot_eta - tot_eta0) > 1.0e-10
    rmesh = amr.refinemarkedelements(mesh, imark)
    rCG1, _ = amr.spaces(rmesh)
    assert rCG1.dim() == 109


def test_nsvmark_allinactive():
    # classic obstacle-touching-zero problem:  -Delta u = 8,  u >= 0,  u = 0 on boundary
    mesh = UnitSquareMesh(8, 8)
    amr = VIAMR(debug=True)
    CG1, DG0 = amr.spaces(mesh)
    lb = Function(CG1).interpolate(Constant(0.0))
    f = Constant(-8.0)
    g = Constant(0.0)

    u = Function(CG1)
    v = TestFunction(CG1)
    F = (inner(grad(u), grad(v)) - f * v) * dx
    bcs = DirichletBC(CG1, g, "on_boundary")
    problem = NonlinearVariationalProblem(F, u, bcs=bcs)
    sp = {
        "snes_type": "vinewtonrsls",
        "snes_vi_zero_tolerance": 1.0e-10,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp, options_prefix="s")
    ub = Function(CG1).interpolate(Constant(1.0e10))
    solver.solve(bounds=(lb, ub))
    assert amr.checkadmissible(u, lb)

    mark, etainf, sigmah, total_err = amr.nsvmark(u, lb, g, f, g)
    assert mark.function_space().ufl_element() == DG0.ufl_element()
    assert 0 <= amr.countmark(mark) <= DG0.dim()
    assert total_err > 0.0

    # also exercises _fixedrate()'s "total" branch
    tmark, ieta, tot2 = amr.gradrecinactivemark(u, lb, theta=0.5, method="total")
    assert tmark.function_space().ufl_element() == DG0.ufl_element()
    assert tot2 >= 0.0


def test_fixedrate_total():
    # Check of _fixedrate()'s "total" strategy.  eta is a function of x,y
    # (via SpatialCoordinate) so that its value on a given physical cell is
    # the same regardless of which process owns that cell; this is what lets
    # tests/test_parallel.py::test_fixedrate_total_parallel reuse the same
    # construction and assert it reproduces these same serial numbers.
    mesh = UnitSquareMesh(4, 1)
    amr = VIAMR(debug=True)
    _, DG0 = amr.spaces(mesh)
    assert DG0.dim() == 8
    x, y = SpatialCoordinate(mesh)
    eta = Function(DG0).interpolate(x + 2 * y)
    mark, ethresh, total_error_est = amr._fixedrate(eta, theta=0.5, method="total")
    assert abs(ethresh - 1.75) < 1.0e-10
    assert amr.countmark(mark) == 2
    assert abs(total_error_est - 4.444097208657794) < 1.0e-10


if __name__ == "__main__":
    test_overrefine_udo()
    test_finer_udo()
    test_refine_vcd()
    test_refine_vcd_petscsbr()
    test_refine_vcd_firedrake_petscsbr()
    test_refine_gr()
    test_refine_br()
    test_refine_br_total()
    test_nsvmark_allinactive()
    test_fixedrate_total()
