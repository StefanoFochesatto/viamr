import pytest
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
    rmesh = amr.refinesbr2D(mesh, mark)  # PETSc's skeleton-based refine method
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
    rmesh = amr.refinesbr2D(mesh, mark)  # PETSc's skeleton-based refine method
    rCG1, _ = amr.spaces(rmesh)
    # check that direct solver gets same result
    markDS = amr.vcdmark(u, psi, directsolver=True)
    rmeshDS = amr.refinesbr2D(mesh, markDS)
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
    rmesh = amr.refinesbr2D(mesh, imark)
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
    rmesh = amr.refinesbr2D(mesh, imark)
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
    rmesh = amr.refinesbr2D(mesh, imark)
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
    rmesh = amr.refinesbr2D(mesh, imark)
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

    mark, etainf, etad, sigmah, total_err = amr.nsvmark(u, lb, g, f, g)
    assert mark.function_space().ufl_element() == DG0.ufl_element()
    assert etad.function_space().ufl_element() == DG0.ufl_element()
    assert 0 <= amr.countmark(mark) <= DG0.dim()
    assert total_err > 0.0

    # also exercises fixedratemark()'s "total" branch
    tmark, ieta, tot2 = amr.gradrecinactivemark(u, lb, theta=0.5, method="total")
    assert tmark.function_space().ufl_element() == DG0.ufl_element()
    assert tot2 >= 0.0


def _nsvmark_nontrivial_soln(amr):
    """Solve NSV03's own "Example 7.2" (their sec. 7.2): a constant obstacle
    chi=0 on (-1,1)^2 with radius r=0.7, whose exact solution is
    u(x) = (max(|x|^2-r^2,0))^2.  Unlike test_nsvmark_allinactive() above, this
    gives nsvmark() a genuine interior contact set, free boundary, and inactive
    boundary to work with.  Shared by _nsvmark_nontrivial() here and
    tests/test_parallel.py::test_nsvmark_nontrivial_parallel()."""
    r = 0.7
    mesh = RectangleMesh(
        16,
        16,
        1.0,
        1.0,
        originX=-1.0,
        originY=-1.0,
        diagonal="crossed",
        distribution_parameters=VIAMR.PARALLEL_OVERLAP,
    )
    CG1, _ = amr.spaces(mesh)
    x, y = SpatialCoordinate(mesh)
    x2 = x ** 2 + y ** 2
    circle = x2 - r ** 2
    f_ufl = conditional(
        x2 <= r ** 2, -8.0 * r ** 2 * (1.0 - circle), -4.0 * (2.0 * x2 + 2.0 * circle)
    )
    g_ufl = circle ** 2

    uh = Function(CG1, name="uh")
    vh = TestFunction(CG1)
    F = inner(grad(uh), grad(vh)) * dx - f_ufl * vh * dx
    g = Function(CG1).interpolate(g_ufl)
    bcs = DirichletBC(CG1, g, "on_boundary")
    problem = NonlinearVariationalProblem(F, uh, bcs)
    lb = Function(CG1).interpolate(Constant(0.0))
    ub = Function(CG1).interpolate(Constant(1.0e10))
    sp = {
        "snes_type": "vinewtonrsls",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp, options_prefix="s")
    solver.solve(bounds=(lb, ub))
    return mesh, uh, lb, f_ufl, g, g_ufl


def _nsvmark_nontrivial(amr):
    # Shared by test_nsvmark_nontrivial() here and
    # tests/test_parallel.py::test_nsvmark_nontrivial_parallel(), so the
    # two assert identical counts from one definition.
    mesh, uh, lb, f_ufl, g, g_ufl = _nsvmark_nontrivial_soln(amr)
    CG1, DG0 = amr.spaces(mesh)
    assert amr.checkadmissible(uh, lb)

    mark, etainf, etad, sigmah, total_err = amr.nsvmark(
        uh, lb, g, f_ufl, g_ufl, dualtol=1.0e-8
    )
    assert mark.function_space().ufl_element() == DG0.ufl_element()
    assert sigmah.function_space().ufl_element() == CG1.ufl_element()
    assert etad.function_space().ufl_element() == DG0.ufl_element()
    assert 0 < amr.countmark(mark) < DG0.dim()  # marks something, not everything
    assert total_err > 0.0
    assert amr.scalarrange(etad)[0] >= 0.0  # eta_d is a norm, so non-negative

    # regression for the eta_d dominance gate (see msg.txt): uh is exactly
    # 0 = lb deep inside the contact set, so refinement driven by eta_d there
    # would be wasted; with the default gate (etadratio=1.0) it must not mark
    # any such elements
    x, y = SpatialCoordinate(mesh)
    deepinterior = Function(DG0).interpolate(
        conditional(x ** 2 + y ** 2 < (0.4 * 0.7) ** 2, 1.0, 0.0)
    )
    markedinterior = Function(DG0).interpolate(mark * deepinterior)
    assert amr.countmark(markedinterior) == 0

    # forcing the second (eta_d) pass to always run (etadratio=0.0) must mark
    # at least as many elements as the gated default does
    mark2, _, _, _, _ = amr.nsvmark(
        uh, lb, g, f_ufl, g_ufl, dualtol=1.0e-8, etadratio=0.0
    )
    assert amr.countmark(mark2) >= amr.countmark(mark)


def test_nsvmark_nontrivial():
    _nsvmark_nontrivial(VIAMR(debug=True))


def _nsvmark_kink_soln(amr, m):
    """Solve an obstacle problem whose obstacle has kinks *inside* the contact set:
    the pyramid chi = dist(x, boundary of Omega) - 1/5 on Omega = (-1,1)^2, with
    f = -5 and g = 0.  This is the example of section 3.2 of NSV05.  The load
    pushes the solution down onto the pyramid over a genuine two-dimensional
    contact region, and chi is concave and piecewise affine with ridges along the
    diagonals |x| = |y|.  A "crossed" RectangleMesh puts element edges exactly on
    those diagonals, so I_h chi = chi and the jump [[d_n u_h]] along a ridge is the
    fixed kink in grad(chi), which does *not* vanish under refinement.  Returns
    (max_T etainf, Eh)."""
    mesh = RectangleMesh(
        m,
        m,
        1.0,
        1.0,
        originX=-1.0,
        originY=-1.0,
        diagonal="crossed",
        distribution_parameters=VIAMR.PARALLEL_OVERLAP,
    )
    CG1, _ = amr.spaces(mesh)
    x, y = SpatialCoordinate(mesh)
    f_ufl = Constant(-5.0)
    g_ufl = Constant(0.0)
    lb = Function(CG1).interpolate(min_value(1.0 - abs(x), 1.0 - abs(y)) - 0.2)

    uh = Function(CG1, name="uh").interpolate(max_value(Constant(0.0), lb))
    vh = TestFunction(CG1)
    F = inner(grad(uh), grad(vh)) * dx - f_ufl * vh * dx
    g = Function(CG1).interpolate(g_ufl)
    problem = NonlinearVariationalProblem(F, uh, DirichletBC(CG1, g, "on_boundary"))
    ub = Function(CG1).interpolate(Constant(1.0e10))
    sp = {
        "snes_type": "vinewtonrsls",
        "snes_atol": 1.0e-12,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    NonlinearVariationalSolver(
        problem, solver_parameters=sp, options_prefix="s"
    ).solve(bounds=(lb, ub))
    assert amr.checkadmissible(uh, lb)
    # the contact set must be nonempty, or the kink is not exercised at all
    assert amr.countmark(amr.elemactive(uh, lb)) > 0

    _, etainf, _, _, Eh = amr.nsvmark(uh, lb, g, f_ufl, g_ufl)
    return amr.scalarrange(etainf)[1], Eh


def _nsvmark_kink_decay(amr):
    """Regression for the scaling of the jump term in nsvmark()'s R_infty.

    NSV03 (3.7) with p=infty gives R_infty|_T = h_T^{-1} ||[[d_n u_h]]||_{0,inf;dT}
    plus the interior residual, so the estimator term C_0 h_T^2 R_infty is O(h_T)
    on an element carrying a kink, and must decay under refinement.  A previous
    version recovered the jump by dividing an assembled facet integral by the
    *cell volume*, which scales like jump*h^(d-1)/h^d, i.e. one factor of h_T too
    large.  That left C_0 h_T^2 R_infty tending to C_0 |jump|, a nonzero constant.
    The estimator then stalled at an O(1) value no matter how fine the mesh, and
    elements carrying a kink in chi permanently dominated the marking threshold.

    Halving h must therefore cut both max_T etainf and Eh substantially.  The bug
    gave a ratio of essentially 1; the threshold 0.7 below sits far from both that
    and the correct behavior.  Shared by test_nsvmark_kink_decay() here and
    tests/test_parallel.py::test_nsvmark_kink_decay_par()."""
    coarse_eta, coarse_Eh = _nsvmark_kink_soln(amr, 8)
    fine_eta, fine_Eh = _nsvmark_kink_soln(amr, 16)
    assert coarse_eta > 0.0 and coarse_Eh > 0.0
    assert fine_eta < 0.7 * coarse_eta
    assert fine_Eh < 0.7 * coarse_Eh


def test_nsvmark_kink_decay():
    _nsvmark_kink_decay(VIAMR(debug=True))


def _nsvmark_active_boundary_soln(amr):
    """Solve examples/aol.py's obstacle problem: -Delta u = -1, u >= 0, on the
    rectangle [0,0.5]x[0,1], with Dirichlet data equal to the exact solution
    u(x,y) = max(0.25 r - 0.5 - 0.5 ln(0.5 r), 0), r = (x+1)^2+y^2.  Unlike
    _nsvmark_nontrivial_soln() above, whose Dirichlet data keeps the whole
    boundary inactive, part of this rectangle's boundary genuinely touches
    the obstacle (r >= 2 there), exercising nsvmark()'s NSV03-page-169
    boundary formula for sigma_h rather than the all-zero fallback."""
    mesh = RectangleMesh(
        6, 12, 0.5, 1.0, distribution_parameters=VIAMR.PARALLEL_OVERLAP
    )
    CG1, _ = amr.spaces(mesh)
    x, y = SpatialCoordinate(mesh)
    r = (x + 1.0) ** 2 + y ** 2
    g_ufl = conditional(r < 2.0, 0.25 * r - 0.5 - 0.5 * ln(0.5 * r), 0.0)
    f_ufl = Constant(-1.0)

    uh = Function(CG1, name="uh")
    vh = TestFunction(CG1)
    F = inner(grad(uh), grad(vh)) * dx - f_ufl * vh * dx
    g = Function(CG1).interpolate(g_ufl)
    bcs = DirichletBC(CG1, g, "on_boundary")
    problem = NonlinearVariationalProblem(F, uh, bcs)
    lb = Function(CG1).interpolate(Constant(0.0))
    ub = Function(CG1).interpolate(Constant(1.0e10))
    sp = {
        "snes_type": "vinewtonrsls",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp, options_prefix="s")
    solver.solve(bounds=(lb, ub))
    return mesh, uh, lb, f_ufl, g, g_ufl


def test_nsvmark_active_boundary():
    amr = VIAMR(debug=True)
    mesh, uh, lb, f_ufl, g, g_ufl = _nsvmark_active_boundary_soln(amr)
    CG1, DG0 = amr.spaces(mesh)
    assert amr.checkadmissible(uh, lb)

    # boundary indicator, parallel-safe (same DirichletBC-apply idiom nsvmark() uses)
    isbdry = Function(CG1).assign(0.0)
    DirichletBC(CG1, Constant(1.0), "on_boundary").apply(isbdry)

    # confirm this problem genuinely has both active and inactive boundary
    # nodes, i.e. it actually exercises the case nsvmark() previously ignored
    gap = Function(CG1).interpolate(uh - lb)
    gap_boundary_only = Function(CG1).interpolate(
        conditional(isbdry > 0.5, gap, 1.0e10)
    )
    assert amr.scalarrange(gap_boundary_only)[0] < amr.activetol  # some node active
    assert amr.scalarrange(Function(CG1).interpolate(gap * isbdry))[1] > amr.activetol
    # ... and some node inactive

    mark, etainf, etad, sigmah, total_err = amr.nsvmark(uh, lb, g, f_ufl, g_ufl)
    assert sigmah.function_space().ufl_element() == CG1.ufl_element()

    # the point of the fix: sigma_h must respect the sign convention (also
    # asserted internally by nsvmark()) and now be genuinely nonzero at some
    # boundary dof; before the fix, boundary values of sigma_h were
    # unconditionally zeroed by nsvmark(), regardless of activity there
    sigmah_boundary = Function(CG1).interpolate(sigmah * isbdry)
    assert amr.scalarrange(sigmah_boundary)[0] >= -1.0e-10
    assert amr.scalarrange(sigmah_boundary)[1] > amr.activetol


def _fixedrate_total_case(amr):
    # Shared by test_fixedrate_total() here and
    # tests/test_parallel.py::test_fixedrate_total_parallel(), so the two
    # assert the identical numbers from one definition.  eta is a function of
    # x,y (via SpatialCoordinate) so that its value on a given physical cell
    # is the same regardless of which process owns that cell.
    mesh = UnitSquareMesh(4, 1)
    _, DG0 = amr.spaces(mesh)
    assert DG0.dim() == 8
    x, y = SpatialCoordinate(mesh)
    eta = Function(DG0).interpolate(x + 2 * y)
    mark, ethresh = amr.fixedratemark(eta, theta=0.5, method="total")
    total_error_est = amr._globalpnorm(eta, 2.0)
    assert abs(ethresh - 1.75) < 1.0e-10
    assert amr.countmark(mark) == 2
    assert abs(total_error_est - 4.444097208657794) < 1.0e-10


def test_fixedrate_total():
    _fixedrate_total_case(VIAMR(debug=True))


def _udomark_nontrivial_lb(amr):
    # Shared "somewhat interesting obstacle configuration" used by
    # test_udomark_nontrivial() here, and by
    # tests/test_parallel.py::test_udomark_nontrivial_parallel() and
    # test_udo_regression().
    mesh = RectangleMesh(20, 20, 1, 1, distribution_parameters=VIAMR.PARALLEL_OVERLAP)
    CG1, _ = amr.spaces(mesh)
    u = Function(CG1).interpolate(1.0)
    x, y = SpatialCoordinate(mesh)
    lb = Function(CG1).interpolate(
        conditional(
            And(And(x > 0.15, x < 0.35), And(y > 0.15, y < 0.35)),
            1.0,
            conditional(
                And(And(x > 0.65, x < 0.85), And(y > 0.15, y < 0.35)),
                1.0,
                conditional(
                    And(And(x > 0.15, x < 0.35), And(y > 0.65, y < 0.85)),
                    1.0,
                    conditional(
                        And(And(x > 0.65, x < 0.85), And(y > 0.65, y < 0.85)), 1.0, 0.0
                    ),
                ),
            ),
        )
    )
    return u, lb


def _udomark_nontrivial(amr):
    # Shared by test_udomark_nontrivial() here and
    # tests/test_parallel.py::test_udomark_nontrivial_parallel(), so the
    # two assert the identical count from one definition.
    u, lb = _udomark_nontrivial_lb(amr)
    mark = amr.udomark(u, lb, n=2)
    assert amr.countmark(mark) == 506


def test_udomark_nontrivial():
    _udomark_nontrivial(VIAMR())


def _udomark_restrict_case(amr):
    # Shared by test_udomark_restrict() here and
    # tests/test_parallel.py::test_udomark_restrict_parallel(), so the two
    # assert the identical mesh sizes/counts from one definition. Exercises
    # udomark(restrict=...), and thus VIAMR._filtermesh(), which otherwise
    # has no test coverage.
    u, lb = _udomark_nontrivial_lb(amr)
    mesh = u.function_space().mesh()
    assert amr.meshsizes(mesh)[1] == 800

    markactive = amr.udomark(u, lb, n=2, restrict="active")
    meshactive = markactive.function_space().mesh()
    assert amr.meshsizes(meshactive)[1] == 154
    assert amr.countmark(markactive) == 154

    markinactive = amr.udomark(u, lb, n=2, restrict="inactive")
    meshinactive = markinactive.function_space().mesh()
    assert amr.meshsizes(meshinactive)[1] == 750
    assert amr.countmark(markinactive) == 456

    # unrestricted call still works and is unaffected by the restricted calls
    markfull = amr.udomark(u, lb, n=2)
    assert amr.countmark(markfull) == 506


def test_udomark_restrict():
    amr = VIAMR(debug=True)
    _udomark_restrict_case(amr)
    u, lb = _udomark_nontrivial_lb(amr)
    with pytest.raises(ValueError):
        amr.udomark(u, lb, restrict="bogus")


def _safeactiveunmark_case(amr, f_ufl):
    # classical obstacle problem operator, -div(grad(u)) - f, with a paraboloid
    # obstacle psi = 0.1 - (x-0.5)^2 - (y-0.5)^2, so -div(grad(psi)) = 4
    # everywhere; sigma_psi = 4 - f_ufl is then a known constant
    mesh = UnitSquareMesh(4, 4)
    CG1, _ = amr.spaces(mesh)
    x, y = SpatialCoordinate(mesh)
    psi_ufl = 0.1 - (x - 0.5) ** 2 - (y - 0.5) ** 2
    lb0 = Function(CG1).interpolate(psi_ufl)
    uh0 = Function(CG1).interpolate(psi_ufl)  # fully active

    def F_strong(u, f):
        return -div(grad(u)) - f

    safe = amr.safeactiveunmark(uh0, lb0, F_strong, psi_ufl, f_ufl)
    return amr, uh0, lb0, safe


def test_safeactiveunmark_allsafe():
    # sigma_psi = 4 - (-0.5) = 4.5 > 0 everywhere: every active element is safe
    amr, uh0, lb0, safe = _safeactiveunmark_case(VIAMR(debug=True), Constant(-0.5))
    active0 = amr.elemactive(uh0, lb0)
    assert amr.countmark(safe) == amr.countmark(active0)


def test_safeactiveunmark_nonesafe():
    # sigma_psi = 4 - 10 = -6 < 0 everywhere: no active element is safe
    amr, _, _, safe = _safeactiveunmark_case(VIAMR(debug=True), Constant(10.0))
    assert amr.countmark(safe) == 0


def test_safeactiveunmark_restricted_to_active():
    # sigma_psi > 0 everywhere, but only the left half of the domain is active;
    # safe must never extend beyond the current *thinned* active set (see
    # VIAMR.thinelemactive()), i.e. it must exclude the border adjacent to
    # the discrete free boundary.  Uses an 8x8 (not 4x4) mesh so the active
    # region is wide enough to leave a nonempty thinned interior.
    amr = VIAMR(debug=True)
    mesh = UnitSquareMesh(8, 8)
    CG1, _ = amr.spaces(mesh)
    x, y = SpatialCoordinate(mesh)
    psi_ufl = 0.1 - (x - 0.5) ** 2 - (y - 0.5) ** 2
    lb0 = Function(CG1).interpolate(psi_ufl)
    uh0 = Function(CG1).interpolate(
        conditional(x < 0.5, psi_ufl, psi_ufl + 1.0)  # active only for x < 0.5
    )

    def F_strong(u, f):
        return -div(grad(u)) - f

    safe = amr.safeactiveunmark(uh0, lb0, F_strong, psi_ufl, Constant(-0.5))
    active0 = amr.elemactive(uh0, lb0)
    thinactive0 = amr.thinelemactive(uh0, lb0)
    assert amr.countmark(active0) < amr.countmark(amr.eleminactive(uh0, lb0))
    assert 0 < amr.countmark(thinactive0) < amr.countmark(active0)
    assert amr.countmark(safe) == amr.countmark(thinactive0)


def test_safeactiveunmark_notimplemented():
    amr, uh0, lb0, _ = _safeactiveunmark_case(VIAMR(debug=True), Constant(-0.5))

    def F_strong(u, f):
        return -div(grad(u)) - f

    x, y = SpatialCoordinate(uh0.function_space().mesh())
    psi_ufl = 0.1 - (x - 0.5) ** 2 - (y - 0.5) ** 2
    with pytest.raises(NotImplementedError):
        amr.safeactiveunmark(
            uh0, lb0, F_strong, psi_ufl, Constant(-0.5), psi_mode="data"
        )
    with pytest.raises(NotImplementedError):
        amr.safeactiveunmark(
            uh0, lb0, F_strong, psi_ufl, Constant(-0.5), f_mode="data"
        )


def test_upper_obstacle_elasto():
    # Classical Laplacian *upper*-obstacle problem (elastic-plastic torsion):
    #   -Delta u = 2C,  u <= psi,  u = 0 on boundary,
    # with tent obstacle psi(x,y) = min(y, 1-x, x, 1-y) on the unit square.
    # See R. Kornhuber (1994). Monotone multigrid methods for elliptic
    # variational inequalities, Numer. Math. 69, 167-184.
    # Exercises marking primitives with boxside="upper".
    mesh = UnitSquareMesh(16, 16, diagonal="crossed")
    amr = VIAMR(debug=True)
    CG1, DG0 = amr.spaces(mesh)
    x, y = SpatialCoordinate(mesh)
    psi_ufl = conditional(
        y <= x,
        conditional(y <= 1.0 - x, y, 1.0 - x),
        conditional(y <= 1.0 - x, x, 1.0 - y),
    )
    ub = Function(CG1).interpolate(psi_ufl)
    lb = Function(CG1).interpolate(Constant(-1.0e10))
    C = 2.5
    f = Constant(2.0 * C)

    u = Function(CG1)
    v = TestFunction(CG1)
    F = (inner(grad(u), grad(v)) - f * v) * dx
    bcs = DirichletBC(CG1, Constant(0.0), "on_boundary")
    problem = NonlinearVariationalProblem(F, u, bcs=bcs)
    sp = {
        "snes_type": "vinewtonrsls",
        "snes_vi_zero_tolerance": 1.0e-10,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp, options_prefix="s")
    solver.solve(bounds=(lb, ub))

    assert amr.checkadmissible(u, ub, boxside="upper")

    active = amr.elemactive(u, ub, boxside="upper")
    inactive = amr.eleminactive(u, ub, boxside="upper")
    assert 0 < amr.countmark(active) < DG0.dim()  # genuine contact set, not all/none
    assert amr.countmark(active) + amr.countmark(inactive) == DG0.dim()

    mark = amr.udomark(u, ub, boxside="upper", n=1)
    assert 0 < amr.countmark(mark) < DG0.dim()

    _, fb = amr.freeboundarygraph2D(u, ub, boxside="upper")
    assert len(fb) > 0  # contact plateau has a nonempty free boundary


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
    test_nsvmark_nontrivial()
    test_nsvmark_active_boundary()
    test_fixedrate_total()
    test_udomark_nontrivial()
    test_udomark_restrict()
    test_upper_obstacle_elasto()
