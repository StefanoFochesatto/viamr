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
    assert rCG1.dim() == 109


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


def test_nsv03mark_allinactive():
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

    mark, etainf, etad, sigmah, total_err = amr.nsv03mark(u, (lb, None), g, f, g)
    assert mark.function_space().ufl_element() == DG0.ufl_element()
    assert etad.function_space().ufl_element() == DG0.ufl_element()
    assert 0 <= amr.countmark(mark) <= DG0.dim()
    assert total_err > 0.0

    # also exercises fixedratemark()'s "total" branch
    tmark, ieta, tot2 = amr.gradrecinactivemark(u, lb, theta=0.5, method="total")
    assert tmark.function_space().ufl_element() == DG0.ufl_element()
    assert tot2 >= 0.0


def _nsv03mark_nontrivial_soln(amr, m=16):
    """Solve NSV03's own "Example 7.2" (their sec. 7.2): a constant obstacle
    chi=0 on (-1,1)^2 with radius r=0.7, whose exact solution is
    u(x) = (max(|x|^2-r^2,0))^2.  Unlike test_nsv03mark_allinactive() above, this
    gives nsv03mark() a genuine interior contact set, free boundary, and inactive
    boundary to work with.  Shared by _nsv03mark_nontrivial() here,
    tests/test_parallel.py::test_nsv03mark_nontrivial_parallel(), and (at two
    values of m) _nsv05mark_effectivity_case() below, which needs the exact
    solution to form an effectivity index."""
    r = 0.7
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


def _nsv03mark_nontrivial(amr):
    # Shared by test_nsv03mark_nontrivial() here and
    # tests/test_parallel.py::test_nsv03mark_nontrivial_parallel(), so the
    # two assert identical counts from one definition.
    mesh, uh, lb, f_ufl, g, g_ufl = _nsv03mark_nontrivial_soln(amr)
    CG1, DG0 = amr.spaces(mesh)
    assert amr.checkadmissible(uh, lb)

    mark, etainf, etad, sigmah, total_err = amr.nsv03mark(
        uh, (lb, None), g, f_ufl, g_ufl, dualtol=1.0e-8
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
    mark2, _, _, _, _ = amr.nsv03mark(
        uh, (lb, None), g, f_ufl, g_ufl, dualtol=1.0e-8, etadratio=0.0
    )
    assert amr.countmark(mark2) >= amr.countmark(mark)


def test_nsv03mark_nontrivial():
    _nsv03mark_nontrivial(VIAMR(debug=True))


def _pyramid_soln(amr, m):
    """Solve an obstacle problem whose obstacle has kinks *inside* the contact set:
    the pyramid chi = dist(x, boundary of Omega) - 1/5 on Omega = (-1,1)^2, with
    f = -5 and g = 0.  This is the example of section 3.2 of NSV05.  The load
    pushes the solution down onto the pyramid over a genuine two-dimensional
    contact region, and chi is concave and piecewise affine with ridges along the
    diagonals |x| = |y|.  A "crossed" RectangleMesh puts element edges exactly on
    those diagonals, so I_h chi = chi and the jump [[d_n u_h]] along a ridge is the
    fixed kink in grad(chi), which does *not* vanish under refinement.

    Shared by _nsv03mark_kink_soln() below, which uses the non-decaying ridge jump
    as a scaling regression, and by _nsv05mark_pyramid() below, for which this is
    NSV05's own full-localization example: the ridges lie *inside* the full-contact
    set Omega_h^0, so nsv05mark() must switch its residual off along them."""
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
    return (mesh,) + _pyramid_solve(amr, mesh)


def _pyramid_solve(amr, mesh):
    """Solve the pyramid problem of _pyramid_soln() on the given mesh, which need
    not be the initial crossed one; returns (uh, lb, f_ufl, g, g_ufl)."""
    CG1, _ = amr.spaces(mesh)
    f_ufl = Constant(-5.0)
    g_ufl = Constant(0.0)
    lb = Function(CG1).interpolate(_pyramid_lb_ufl(mesh))

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
    return uh, lb, f_ufl, g, g_ufl


def _nsv03mark_kink_soln(amr, m):
    """Run nsv03mark() on the pyramid of _pyramid_soln().  Returns
    (max_T etainf, Eh)."""
    _, uh, lb, f_ufl, g, g_ufl = _pyramid_soln(amr, m)
    _, etainf, _, _, Eh = amr.nsv03mark(uh, (lb, None), g, f_ufl, g_ufl)
    return amr.scalarrange(etainf)[1], Eh


def _nsv03mark_kink_decay(amr):
    """Regression for the scaling of the jump term in nsv03mark()'s R_infty.

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
    and the correct behavior.  Shared by test_nsv03mark_kink_decay() here and
    tests/test_parallel.py::test_nsv03mark_kink_decay_par()."""
    coarse_eta, coarse_Eh = _nsv03mark_kink_soln(amr, 8)
    fine_eta, fine_Eh = _nsv03mark_kink_soln(amr, 16)
    assert coarse_eta > 0.0 and coarse_Eh > 0.0
    assert fine_eta < 0.7 * coarse_eta
    assert fine_Eh < 0.7 * coarse_Eh


def test_nsv03mark_kink_decay():
    _nsv03mark_kink_decay(VIAMR(debug=True))


def _nsv03mark_active_boundary_soln(amr):
    """Solve examples/aol.py's obstacle problem: -Delta u = -1, u >= 0, on the
    rectangle [0,0.5]x[0,1], with Dirichlet data equal to the exact solution
    u(x,y) = max(0.25 r - 0.5 - 0.5 ln(0.5 r), 0), r = (x+1)^2+y^2.  Unlike
    _nsv03mark_nontrivial_soln() above, whose Dirichlet data keeps the whole
    boundary inactive, part of this rectangle's boundary genuinely touches
    the obstacle (r >= 2 there), exercising nsv03mark()'s NSV03-page-169
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


def test_nsv03mark_active_boundary():
    amr = VIAMR(debug=True)
    mesh, uh, lb, f_ufl, g, g_ufl = _nsv03mark_active_boundary_soln(amr)
    CG1, DG0 = amr.spaces(mesh)
    assert amr.checkadmissible(uh, lb)

    # boundary indicator, parallel-safe (same DirichletBC-apply idiom nsv03mark() uses)
    isbdry = Function(CG1).assign(0.0)
    DirichletBC(CG1, Constant(1.0), "on_boundary").apply(isbdry)

    # confirm this problem genuinely has both active and inactive boundary
    # nodes, i.e. it actually exercises the case nsv03mark() previously ignored
    gap = Function(CG1).interpolate(uh - lb)
    gap_boundary_only = Function(CG1).interpolate(
        conditional(isbdry > 0.5, gap, 1.0e10)
    )
    assert amr.scalarrange(gap_boundary_only)[0] < amr.activetol  # some node active
    assert amr.scalarrange(Function(CG1).interpolate(gap * isbdry))[1] > amr.activetol
    # ... and some node inactive

    mark, etainf, etad, sigmah, total_err = amr.nsv03mark(uh, (lb, None), g, f_ufl, g_ufl)
    assert sigmah.function_space().ufl_element() == CG1.ufl_element()

    # the point of the fix: sigma_h must respect the sign convention (also
    # asserted internally by nsv03mark()) and now be genuinely nonzero at some
    # boundary dof; before the fix, boundary values of sigma_h were
    # unconditionally zeroed by nsv03mark(), regardless of activity there
    sigmah_boundary = Function(CG1).interpolate(sigmah * isbdry)
    assert amr.scalarrange(sigmah_boundary)[0] >= -1.0e-10
    assert amr.scalarrange(sigmah_boundary)[1] > amr.activetol


def _deepfullcontact(amr, fullcontact):
    """DG0 indicator of the elements lying *deep* inside the discrete
    full-contact set Omega_h^0, i.e. those all of whose vertices carry a star
    contained in Omega_h^0.

    This, and not fullcontact itself, is where nsv05mark()'s eta must vanish
    identically.  Its element value eta_T takes a max of the star indicator
    eta_z over the vertices of T, so a full-contact element which happens to
    have a vertex whose star pokes outside Omega_h^0 can legitimately inherit a
    nonzero eta_z from there."""
    CG1, _ = amr.spaces(fullcontact.function_space().mesh())
    starfull = amr._elemtonodeextreme(fullcontact, CG1, minimum=True, defaultval=1.0)
    return amr._elemextreme(starfull, minimum=True, defaultval=1.0)


def _nsv05mark_pyramid(amr):
    """Full localization, which is the property NSV05 exists to provide, on
    NSV05's own section 3.2 example; see _pyramid_soln().  Also checks the
    invariants which hold for any input.

    The obstacle's ridges lie inside the contact set, so nsv05mark() must switch
    its residual off along them, while nsv03mark() -- whose residual has sigma_h
    subtracted from f but is not restricted to Omega_h^+ -- does not.  This is
    the mechanism behind the coarse contact-set meshes of NSV05 Figure 3.4.

    Shared by test_nsv05mark_pyramid() here and
    tests/test_parallel.py::test_nsv05mark_pyramid_par(), so the two assert the
    identical counts from one definition."""
    mesh, uh, lb, f_ufl, g, g_ufl = _pyramid_soln(amr, 16)
    CG1, DG0 = amr.spaces(mesh)

    mark, eta, sz, fullcontact, Eh = amr.nsv05mark(uh, (lb, None), g, f_ufl, g_ufl)

    # return contract
    assert mark.function_space().ufl_element() == DG0.ufl_element()
    assert eta.function_space().ufl_element() == DG0.ufl_element()
    assert fullcontact.function_space().ufl_element() == DG0.ufl_element()
    assert sz.function_space().ufl_element() == CG1.ufl_element()

    # invariants which hold unconditionally
    assert 0 < amr.countmark(mark) < DG0.dim()  # marks something, not everything
    assert Eh > 0.0
    assert amr.scalarrange(eta)[0] >= 0.0  # eta is a sum of norms
    assert Eh >= amr.scalarrange(eta)[1]  # separate global sups; see nsv05mark()

    # Omega_h^0 must be nonempty, or the localization assertions below are
    # vacuous.  Every full-contact element is in contact, but not conversely:
    # the sign conditions on f and J_h, and the requirement that *all* vertices
    # lie in C_h, drop the fringe of the contact set near the free boundary.
    active = amr.elemactive(uh, lb)
    nfull = amr.countmark(fullcontact)
    assert nfull > 0
    assert amr.countmark(Function(DG0).interpolate(fullcontact * (1.0 - active))) == 0
    assert nfull < amr.countmark(active)

    # the property itself: eta vanishes identically deep inside Omega_h^0, so
    # nothing there is marked, no matter how large the ridge jumps are
    deep = _deepfullcontact(amr, fullcontact)
    ndeep = amr.countmark(deep)
    assert ndeep > 0
    assert amr.scalarrange(Function(DG0).interpolate(eta * deep))[1] == 0.0
    assert amr.countmark(Function(DG0).interpolate(mark * deep)) == 0

    # ... in contrast to nsv03mark() on the same solution, whose residual stays
    # alive on the ridges.  Only the localization is compared, not the sizes of
    # the two estimators: nsv05mark() carries the prefactor C0 |log(h_min/diam
    # Omega)|^2 on its residual and nsv03mark() has no such factor, so E_h and
    # Etilde_h are not on a common scale.
    _, etainf, _, _, _ = amr.nsv03mark(uh, (lb, None), g, f_ufl, g_ufl)
    assert amr.scalarrange(Function(DG0).interpolate(etainf * deep))[1] > 0.0

    return nfull, ndeep, amr.countmark(mark)


# (number of full-contact elements, of those deep inside Omega_h^0, of marked
# elements) from _nsv05mark_pyramid().  Asserted in both the serial and the
# parallel test, so that the two are pinned to the same numbers.
_NSV05_PYRAMID_COUNTS = (440, 288, 120)


def test_nsv05mark_pyramid():
    assert _nsv05mark_pyramid(VIAMR(debug=True)) == _NSV05_PYRAMID_COUNTS


def _nsv05mark_effectivity_case(amr, m):
    """Run nsv05mark() on NSV03 Example 7.2 at mesh parameter m; see
    _nsv03mark_nontrivial_soln().  Returns (Eh, ||u - u_h||_{0,inf;Omega}),
    using the known exact solution u = (max(|x|^2-r^2,0))^2."""
    mesh, uh, lb, f_ufl, g, g_ufl = _nsv03mark_nontrivial_soln(amr, m=m)
    _, _, _, _, Eh = amr.nsv05mark(uh, (lb, None), g, f_ufl, g_ufl, dualtol=1.0e-8)
    x, y = SpatialCoordinate(mesh)
    u_ufl = max_value(x ** 2 + y ** 2 - 0.7 ** 2, 0.0) ** 2
    # CG4, not DG4: Firedrake's DG node set on simplices omits the vertices, so
    # maximizing over it under-samples a sup norm
    CG4 = FunctionSpace(mesh, "CG", 4)
    err = amr.scalarrange(Function(CG4).interpolate(abs(u_ufl - uh)))[1]
    assert Eh > 0.0 and err > 0.0
    return Eh, err


def _nsv05mark_effectivity(amr):
    """Regression for the h-scaling of nsv05mark()'s estimator Eh.

    Theorem 2.7 of NSV05 bounds ||u - u_h||_{0,inf;Omega} by E_h, but only with
    the constant C_*|log h_min|^2, which nsv05mark() replaces by C0 = 0.02
    following NSV05 section 3.  So reliability is not available as an assertion
    here.  What is available, and what catches a wrong power of h_z in any one
    term of eta_z, is that Eh and the true error must decay together: their
    ratio, i.e. the effectivity index, has to stay put as h halves.  Compare
    NSV05 Figure 3.2, where the two curves are parallel.

    The bounds below are wide enough to tolerate the preasymptotic range at
    these mesh sizes, and far from the failure they exist to catch, in which one
    term of the estimator scales like a different power of h than the others and
    the ratio drifts by an order of magnitude per refinement.

    Shared by test_nsv05mark_effectivity() here and
    tests/test_parallel.py::test_nsv05mark_effectivity_par()."""
    Eh_coarse, err_coarse = _nsv05mark_effectivity_case(amr, 8)
    Eh_fine, err_fine = _nsv05mark_effectivity_case(amr, 16)
    assert Eh_fine < 0.8 * Eh_coarse  # estimator decays
    assert err_fine < 0.8 * err_coarse  # ... and so does the true error
    ratio = (Eh_fine / err_fine) / (Eh_coarse / err_coarse)
    assert 0.4 < ratio < 2.5  # ... at comparable rates
    return Eh_coarse, err_coarse, Eh_fine, err_fine


def test_nsv05mark_effectivity():
    _nsv05mark_effectivity(VIAMR(debug=True))


def test_nsv05mark_asserts():
    """nsv05mark()'s two admissibility guards must actually fire.  They are the
    method's only defense against being handed something which is not a solution
    of the VI, and the primal one in particular is what licenses dropping
    ||(chi - u_h)^+||_{inf;Omega} from E_h."""
    amr = VIAMR(debug=True)
    mesh, uh, lb, f_ufl, g, g_ufl = _pyramid_soln(amr, 8)
    CG1, _ = amr.spaces(mesh)

    # primal: uh must lie above the discrete obstacle
    below = Function(CG1).interpolate(uh - Constant(1.0))
    with pytest.raises(AssertionError):
        amr.nsv05mark(below, (lb, None), g, f_ufl, g_ufl)

    # dual: NSV05 gives s_z <= 0 at interior nodes.  Handing the method the load
    # with the wrong sign leaves uh primal admissible but flips that: s_z = 0 at
    # the strictly inactive nodes of the true solution, so with -f in place of f
    # it becomes -2 int f phi_z > 0 there, f being negative on this problem.
    with pytest.raises(AssertionError):
        amr.nsv05mark(uh, (lb, None), g, -f_ufl, g_ufl)


def _pyramid_lb_ufl(mesh):
    """The pyramid obstacle of _pyramid_soln() as UFL.  On a crossed mesh its
    ridges lie on element edges, so I_h chi = chi exactly and chi is in CG1."""
    x, y = SpatialCoordinate(mesh)
    return min_value(1.0 - abs(x), 1.0 - abs(y)) - 0.2


def _nsvmark_lbufl_null(amr):
    """Supplying lb_ufl must cost nothing when chi really is in CG1, and it must
    stay that way under adaptive refinement.

    The pyramid obstacle is piecewise affine with ridges on element edges, so
    chi = I_h chi = lb pointwise.  Then ||(chi - u_h)^+|| is zero and the blocked
    gap against continuum chi coincides with the one against chi_h, so both
    estimators must return what they return with no continuum obstacle.  This is what lets
    examples/pyramid.py leave lb_ufl off.

    That example refines adaptively, though, so checking only the initial crossed
    mesh would not cover it.  The ridges survive SBR because bisection splits an
    edge at its midpoint, so a ridge which is a union of edges stays one, and the
    new edges it adds lie inside old elements.  Rather than rely on that argument,
    the loop below re-checks chi = I_h chi and the estimator equivalence on each
    of several SBR-refined meshes.  The chi = I_h chi check samples at CG4 nodes,
    whose element-interior points are exactly where a ridge cutting through a cell
    would show up.

    Shared by test_nsvmark_lbufl_null() here and
    tests/test_parallel.py::test_nsvmark_lbufl_null_par()."""
    mesh, uh, lb, f_ufl, g, g_ufl = _pyramid_soln(amr, 16)
    for level in range(3):
        lb_ufl = _pyramid_lb_ufl(mesh)
        CG4 = FunctionSpace(mesh, "CG", 4)
        chierr = amr.scalarrange(
            Function(CG4).interpolate(abs(lb_ufl - lb))
        )[1]
        assert chierr < 1.0e-14  # chi is still exactly representable in CG1

        mark5a, _, _, fc5a, Eh5a = amr.nsv05mark(uh, (lb, None), g, f_ufl, g_ufl)
        mark5b, _, _, fc5b, Eh5b = amr.nsv05mark(
            uh, (lb, None), g, f_ufl, g_ufl, bounds_ufl=(lb_ufl, None)
        )
        assert amr.countmark(mark5a) == amr.countmark(mark5b)
        assert amr.countmark(fc5a) == amr.countmark(fc5b)
        assert abs(Eh5b - Eh5a) <= 1.0e-12 * Eh5a

        mark3a, _, _, _, Eh3a = amr.nsv03mark(uh, (lb, None), g, f_ufl, g_ufl)
        mark3b, _, _, _, Eh3b = amr.nsv03mark(
            uh, (lb, None), g, f_ufl, g_ufl, bounds_ufl=(lb_ufl, None)
        )
        assert amr.countmark(mark3a) == amr.countmark(mark3b)
        assert abs(Eh3b - Eh3a) <= 1.0e-12 * Eh3a

        if level < 2:
            # refine the way examples/pyramid.py does, on nsv05mark()'s own
            # marking, and re-solve on the result
            mesh = amr.refinesbr2D(mesh, mark5a)
            uh, lb, f_ufl, g, g_ufl = _pyramid_solve(amr, mesh)


def test_nsvmark_lbufl_null():
    _nsvmark_lbufl_null(VIAMR(debug=True))


def _sphericalcap_soln(amr, m):
    """Solve the classical obstacle problem of examples/sphere.py on an m x m
    crossed mesh of (-2,2)^2: the obstacle is the spherical cap sqrt(1-r^2) for
    r <= r0 = 0.9, glued C^1 to a linear far field, with f = 0 and Dirichlet data
    equal to the known exact solution.  Returns
    (mesh, uh, lb, f_ufl, g, g_ufl, lb_ufl, u_ufl).

    In contrast to _pyramid_soln(), chi here is *curved*, so it is not
    representable in CG1.  The exact solution equals chi throughout the contact
    set while u_h equals I_h chi there, which makes chi - I_h chi a genuine and,
    on these meshes, dominant part of the pointwise error.  This is the case the
    lb_ufl kwarg exists for."""
    r0, afree = 0.9, 0.697965148223374
    A, B = 0.680259411891719, 0.471519893402112
    mesh = RectangleMesh(
        m, m, Lx=2.0, Ly=2.0, originX=-2.0, originY=-2.0, diagonal="crossed",
        distribution_parameters=VIAMR.PARALLEL_OVERLAP,
    )
    CG1, _ = amr.spaces(mesh)
    x, y = SpatialCoordinate(mesh)
    r = sqrt(x * x + y * y)
    psi0 = (1.0 - r0 * r0) ** 0.5
    lb_ufl = conditional(
        le(r, r0), sqrt(1.0 - r * r), psi0 + (-r0 / psi0) * (r - r0)
    )
    u_ufl = conditional(le(r, afree), lb_ufl, -A * ln(r) + B)
    f_ufl, g_ufl = Constant(0.0), u_ufl

    lb = Function(CG1, name="psi").interpolate(lb_ufl)
    uh = Function(CG1, name="uh").interpolate(max_value(lb, Constant(0.0)))
    vh = TestFunction(CG1)
    F = inner(grad(uh), grad(vh)) * dx - f_ufl * vh * dx
    g = Function(CG1).interpolate(g_ufl)
    ub = Function(CG1).interpolate(Constant(1.0e10))
    sp = {
        "snes_type": "vinewtonrsls",
        "snes_rtol": 1.0e-10,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    NonlinearVariationalSolver(
        NonlinearVariationalProblem(F, uh, DirichletBC(CG1, g, "on_boundary")),
        solver_parameters=sp, options_prefix="s",
    ).solve(bounds=(lb, ub))
    assert amr.checkadmissible(uh, lb)
    assert amr.countmark(amr.elemactive(uh, lb)) > 0
    return mesh, uh, lb, f_ufl, g, g_ufl, lb_ufl, u_ufl


def _nsvmark_curvedobstacle(amr):
    """Both NSV estimators must see the obstacle-approximation error when the
    obstacle is curved.

    On the spherical cap the exact solution equals chi throughout the contact
    set and u_h equals I_h chi there, so ||u - u_h||_{0,inf;Omega} is exactly
    ||chi - I_h chi||_{0,inf}, and it is attained *inside* the contact set.  That
    quantity is term 2 of NSV03 (7.1) and the second term of NSV05 Theorem 2.7,
    and with no continuum obstacle both methods drop it and so cannot be measuring the
    dominant error at all.  Passing lb_ufl must restore it, which is pinned here
    two ways: the estimator strictly grows, and it becomes an upper bound for the
    true error, i.e. reliable rather than accidentally the right size.

    Shared by test_nsvmark_curvedobstacle() here and
    tests/test_parallel.py::test_nsvmark_curvedobstacle_par()."""
    mesh, uh, lb, f_ufl, g, g_ufl, lb_ufl, u_ufl = _sphericalcap_soln(amr, 32)
    # CG4, not DG4: Firedrake's DG node set on simplices omits the vertices, so
    # maximizing over it under-samples a sup norm
    CG4 = FunctionSpace(mesh, "CG", 4)
    err = amr.scalarrange(Function(CG4).interpolate(abs(u_ufl - uh)))[1]
    # the whole error is the obstacle-approximation term, and it lives in contact
    drop = amr.scalarrange(
        Function(CG4).interpolate(max_value(lb_ufl - uh, 0.0))
    )[1]
    assert err > 0.0
    assert abs(drop - err) <= 1.0e-6 * err

    for marker in (amr.nsv03mark, amr.nsv05mark):
        _, _, _, _, Eh_without = marker(uh, (lb, None), g, f_ufl, g_ufl)
        _, _, _, _, Eh_with = marker(uh, (lb, None), g, f_ufl, g_ufl, bounds_ufl=(lb_ufl, None))
        assert Eh_with > Eh_without  # the dropped term is genuinely nonzero
        assert Eh_with >= err  # ... and it alone already bounds the error


def test_nsvmark_curvedobstacle():
    _nsvmark_curvedobstacle(VIAMR(debug=True))


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
    # eta descending is [2.25, 2, 1.75, 1.5, 1.5, 1.25, 1, 0.75], summing to 12,
    # so the cumulative sum reaches theta * 12 = 6 exactly at the third element.
    # That element is therefore the threshold, and the inclusive comparison of
    # the 'total' strategy marks it along with the two above it.
    assert abs(ethresh - 1.75) < 1.0e-10
    assert amr.countmark(mark) == 3
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


def _nsv03mark_bilateral_soln(amr, m=32):
    """Solve a *bilateral* obstacle problem with the Laplacian, in which both
    obstacles are genuinely touched, and for which an exact solution is known.

    We are not aware of such a benchmark in the literature; the bilateral
    problems we could find are posed on advection-diffusion (as in
    examples/pollutant.py) rather than on the Laplacian.  So this solution is
    constructed here.  On Omega = (-2,2)^2, with constant obstacles chi_lo = -1
    and chi_up = 1, u is radial:

        r <= 0.5        upper contact,  u = 1
        0.5 <= r <= 1   free,           u = 1 - 6s^2 + 4s^3,          s = 2r - 1
        1 <= r <= 1.5   lower contact,  u = -1
        1.5 <= r <= 2   free,           u = -1 + 6t^2 - 8t^3 + 3t^4,  t = 2r - 3
        r >= 2          free,           u = 0

    and the load is

        f = 48                                   r <= 0.5
        f = 48 - 96 s + 24 s(1-s)/r              0.5 <= r <= 1     (48 -> -48)
        f = -48                                  1 <= r <= 1.5
        f = -48 + 192 t - 144 t^2 - 24 t(1-t)^2/r  1.5 <= r <= 2   (-48 -> 0)
        f = 0                                    r >= 2

    Checks on the construction.  Writing sigma = -Lap(u) - f, in the sign
    convention of doc/nsv-box/box.tex, we get sigma = -48 <= 0 on the
    upper-contact disc and sigma = +48 >= 0 on the lower-contact annulus, which
    are the two signs Proposition 3.4 there requires, and sigma = 0 on all three
    free regions.  u is C^1 across every free boundary, since u' = -24 s(1-s)
    and u' = 24 t(1-t)^2 both vanish at their endpoints, so no spurious measure
    sits on a free boundary.  The quartic on the outer annulus is chosen to make
    u'(2) = u''(2) = 0, which is what lets f be continuous everywhere, including
    where u meets the u = 0 corner region.  Every boundary point of the square
    has r >= 2, so g = 0.

    A one-signed f can only reach one obstacle, so f must change sign here; that
    is why the load is large compared to the obstacle gap.  Note the four free
    boundaries are circles, so no triangulation aligns with them, and the
    transition elements where sigma_h changes sign -- the set T_h^+, which is
    what forces the star dilation of Remark 5.2 in box.tex -- are genuinely
    present.

    Shared by _nsv03mark_bilateral() here and
    tests/test_parallel.py::test_nsv03mark_bilateral_par()."""
    mesh = RectangleMesh(
        m,
        m,
        2.0,
        2.0,
        originX=-2.0,
        originY=-2.0,
        diagonal="crossed",
        distribution_parameters=VIAMR.PARALLEL_OVERLAP,
    )
    CG1, _ = amr.spaces(mesh)
    x, y = SpatialCoordinate(mesh)
    r = sqrt(x ** 2 + y ** 2)
    # guards the u'/r terms, which are only ever used where r >= 0.5
    rs = max_value(r, 0.25)
    s = 2 * r - 1
    t = 2 * r - 3
    u_ufl = conditional(
        r < 0.5,
        Constant(1.0),
        conditional(
            r < 1.0,
            1 - 6 * s ** 2 + 4 * s ** 3,
            conditional(
                r < 1.5,
                Constant(-1.0),
                conditional(
                    r < 2.0,
                    -1 + 6 * t ** 2 - 8 * t ** 3 + 3 * t ** 4,
                    Constant(0.0),
                ),
            ),
        ),
    )
    f_ufl = conditional(
        r < 0.5,
        Constant(48.0),
        conditional(
            r < 1.0,
            48 - 96 * s + 24 * s * (1 - s) / rs,
            conditional(
                r < 1.5,
                Constant(-48.0),
                conditional(
                    r < 2.0,
                    -48 + 192 * t - 144 * t ** 2 - 24 * t * (1 - t) ** 2 / rs,
                    Constant(0.0),
                ),
            ),
        ),
    )

    lb = Function(CG1).interpolate(Constant(-1.0))
    ub = Function(CG1).interpolate(Constant(1.0))
    g = Function(CG1).interpolate(Constant(0.0))

    uh = Function(CG1, name="uh")
    vh = TestFunction(CG1)
    F = inner(grad(uh), grad(vh)) * dx - f_ufl * vh * dx
    bcs = DirichletBC(CG1, g, "on_boundary")
    problem = NonlinearVariationalProblem(F, uh, bcs)
    sp = {
        "snes_type": "vinewtonrsls",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp, options_prefix="s")
    solver.solve(bounds=(lb, ub))
    return mesh, uh, lb, ub, f_ufl, g, u_ufl


def _nsv03mark_bilateral(amr):
    # Shared by test_nsv03mark_bilateral() here and
    # tests/test_parallel.py::test_nsv03mark_bilateral_par(), so the two assert
    # the same things from one definition.
    mesh, uh, lb, ub, f_ufl, g, u_ufl = _nsv03mark_bilateral_soln(amr)
    CG1, DG0 = amr.spaces(mesh)
    assert amr.checkadmissible(uh, lb, boxside="lower")
    assert amr.checkadmissible(uh, ub, boxside="upper")

    # both obstacles must actually be touched, or this is not a bilateral test
    nlo = amr.countmark(amr.elemactive(uh, lb, boxside="lower"))
    nup = amr.countmark(amr.elemactive(uh, ub, boxside="upper"))
    assert 0 < nlo < DG0.dim()
    assert 0 < nup < DG0.dim()

    mark, etainf, etad, sigmah, Eh = amr.nsv03mark(uh, (lb, ub), g, f_ufl, g)
    assert mark.function_space().ufl_element() == DG0.ufl_element()
    assert sigmah.function_space().ufl_element() == CG1.ufl_element()
    assert 0 < amr.countmark(mark) < DG0.dim()  # marks something, not everything
    assert amr.scalarrange(etad)[0] >= 0.0  # eta_d is a norm, so non-negative

    # sigma_h must be strictly signed on *both* contact sets, and it recovers
    # the exact multiplier values +-48 there.  This is the property no unilateral
    # estimator can represent, so it is the heart of this test.
    smin, smax = amr.scalarrange(sigmah)
    assert abs(smin + 48.0) < 1.0e-8
    assert abs(smax - 48.0) < 1.0e-8

    # reliability against the known exact solution; see NSV03 (7.1) and Theorem
    # 6.6 of doc/nsv-box/box.tex
    CG4 = FunctionSpace(mesh, "CG", 4)
    err = amr.scalarrange(Function(CG4).interpolate(abs(u_ufl - uh)))[1]
    assert err > 0.0
    assert Eh >= err

    # the unilateral call must refuse this solution rather than silently
    # returning a wrong estimator: sigma_h is genuinely negative on the
    # upper-contact disc, which trips the global-sign assertion
    with pytest.raises(AssertionError):
        amr.nsv03mark(uh, (lb, None), g, f_ufl, g)


def test_nsv03mark_bilateral():
    _nsv03mark_bilateral(VIAMR(debug=True))


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
    test_nsv03mark_allinactive()
    test_nsv03mark_nontrivial()
    test_nsv03mark_active_boundary()
    test_nsv03mark_bilateral()
    test_nsv05mark_pyramid()
    test_nsv05mark_effectivity()
    test_nsv05mark_asserts()
    test_nsvmark_lbufl_null()
    test_nsvmark_curvedobstacle()
    test_fixedrate_total()
    test_udomark_nontrivial()
    test_udomark_restrict()
    test_upper_obstacle_elasto()
