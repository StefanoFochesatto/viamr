import numpy as np
import pytest
from mpi4py import MPI
from firedrake import *
from firedrake.petsc import PETSc
from viamr import VIAMR, BanzMarkingsMixin, poincare_constant, banz_constants


def _setup_2d_obstacle(n=12):
    mesh = UnitSquareMesh(n, n)
    amr = VIAMR()
    CG1, DG0 = amr.spaces(mesh)
    (x, y) = SpatialCoordinate(mesh)
    psi = Function(CG1).interpolate(0.1 - (x - 0.5) ** 2 - (y - 0.5) ** 2)
    u = Function(CG1)
    v = TestFunction(CG1)
    f = Constant(0.0)
    F = inner(grad(u), grad(v)) * dx - f * v * dx
    bc = DirichletBC(CG1, Constant(0.0), "on_boundary")
    problem = NonlinearVariationalProblem(F, u, bcs=bc)
    sp = {
        "snes_type": "vinewtonrsls",
        "snes_rtol": 1.0e-8,
        "snes_atol": 1.0e-12,
    }
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp)
    ub = Function(CG1).interpolate(Constant(PETSc.INFINITY))
    solver.solve(bounds=(psi, ub))

    # Construct discrete multiplier approximation
    kappa = Function(CG1).interpolate(conditional(u <= psi + 1.0e-5, 2.0, 0.0))
    return mesh, amr, CG1, DG0, u, psi, kappa, f


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_constants_validation():
    # Check default constants
    consts = banz_constants()
    assert abs(consts["c_aux"] - 10.0) < 1.0e-12
    assert abs(consts["c_cons"] - 36.0) < 1.0e-12
    assert abs(consts["c_comp"] - 12.0) < 1.0e-12
    assert abs(consts["E"] - 5.0) < 1.0e-12
    assert abs(consts["F"] - 6.0) < 1.0e-12
    assert abs(consts["theta"] - 1.5) < 1.0e-12

    # Check parameter intervals raise ValueError
    with pytest.raises(ValueError, match="beta must be positive"):
        banz_constants(beta=0.0)
    with pytest.raises(ValueError, match="beta must be positive"):
        banz_constants(beta=-1.0)
    with pytest.raises(ValueError, match="c must be positive"):
        banz_constants(c=0.0)
    with pytest.raises(ValueError, match="c_gamma must be positive"):
        banz_constants(c_gamma=0.0)
    with pytest.raises(ValueError, match="chat_gamma must be positive"):
        banz_constants(chat_gamma=0.0)
    with pytest.raises(ValueError, match="delta must lie in"):
        banz_constants(beta=1.0, delta=1.0)
    with pytest.raises(ValueError, match="delta must lie in"):
        banz_constants(beta=1.0, delta=-0.1)
    with pytest.raises(ValueError, match="epsilon must lie in"):
        banz_constants(epsilon=0.0)
    with pytest.raises(ValueError, match="epsilon must lie in"):
        banz_constants(epsilon=1.0)  # for defaults, 1/theta = 2/3 < 1.0

    # Custom valid parameters
    c_custom = banz_constants(beta=2.0, delta=1.0, epsilon=0.1)
    assert c_custom["c_aux"] > 0
    assert c_custom["c_cons"] > 0
    assert c_custom["c_comp"] > 0


def test_poincare_bounding_constant():
    # 1D mesh
    mesh1d = IntervalMesh(20, -1.0, 3.0)
    cp1d = poincare_constant(mesh1d)
    assert abs(cp1d - 4.0 / np.pi) < 1.0e-12

    # 2D unit square mesh
    mesh2d_sq = UnitSquareMesh(5, 5)
    cp2d_sq = poincare_constant(mesh2d_sq)
    assert abs(cp2d_sq - 1.0 / np.pi) < 1.0e-12

    # 2D rectangle mesh
    mesh2d_rect = RectangleMesh(5, 10, 2.0, 5.0)
    cp2d_rect = poincare_constant(mesh2d_rect)
    assert abs(cp2d_rect - 5.0 / np.pi) < 1.0e-12

    # 3D box mesh
    mesh3d = BoxMesh(2, 2, 2, 1.0, 3.0, 2.0)
    cp3d = poincare_constant(mesh3d)
    assert abs(cp3d - 3.0 / np.pi) < 1.0e-12


def test_banzmark_missing_kappa():
    mesh = UnitIntervalMesh(5)
    amr = VIAMR()
    CG1, _ = amr.spaces(mesh)
    u = Function(CG1).assign(0.0)
    with pytest.raises(ValueError, match="kappa is a required parameter"):
        amr.banzmark(u, bounds=(-1.0, None), kappa=None)


def test_1d_manufactured_exact():
    # Manufactured 1D exact solution where:
    #   - auxiliary residual is 0 (res = f + kappa = -1/3 + 1/3 = 0, jumps = 0)
    #   - nonpenetration is 0 (u = 0 >= psi = -1, so (psi - u)_+ = 0)
    #   - multiplier negativity is 0 (kappa = 1/3 >= 0, so kappa_- = 0)
    #   - pairing = (kappa_+, u - psi) = 1/3 * 1.0 = 1/3
    #   - xi = eps * pairing = 1/3 * 1/3 = 1/9
    #   - eta_up^2 = 10 * 0 + 36 * (1/9) = 4.0 => eta_up = 2.0
    mesh = UnitIntervalMesh(10)
    amr = VIAMR()
    CG1, DG0 = amr.spaces(mesh)

    u = Function(CG1).assign(0.0)
    psi = Constant(-1.0)
    kappa = Constant(1.0 / 3.0)
    f = Constant(-1.0 / 3.0)

    mark, fields, eta_up = amr.banzmark(
        u, bounds=(psi, None), kappa=kappa, f_ufl=f, cutoff="sco"
    )

    assert abs(fields["aux_eta2"] - 0.0) < 1.0e-12
    assert abs(fields["N2"] - 0.0) < 1.0e-12
    assert abs(fields["grad_zeta2"] - 0.0) < 1.0e-12
    assert abs(fields["comp_pairing"] - 1.0 / 3.0) < 1.0e-12
    assert abs(fields["xi"] - 1.0 / 9.0) < 1.0e-12
    assert abs(eta_up - 2.0) < 1.0e-12

    assert abs(fields["c_aux"] - 10.0) < 1.0e-12
    assert abs(fields["c_cons"] - 36.0) < 1.0e-12
    assert abs(fields["c_comp"] - 12.0) < 1.0e-12

    # Weighted consistency test: sum_T eta_T^2 == eta_up^2
    with fields["eta"].dat.vec_ro as vec:
        local_sum = float(np.sum(vec.array_r ** 2))
    sum_eta2 = mesh.comm.allreduce(local_sum, op=MPI.SUM)
    assert abs(sum_eta2 - eta_up ** 2) < 1.0e-12


def test_2d_marking_and_fields():
    mesh, amr, CG1, DG0, u, psi, kappa, f = _setup_2d_obstacle(n=10)

    mark, fields, eta_up = amr.banzmark(
        u,
        bounds=(psi, None),
        kappa=kappa,
        f_ufl=f,
        cutoff="sco",
        method="total",
        theta=0.5,
    )

    assert mark.function_space().ufl_element() == DG0.ufl_element()
    assert fields["eta"].function_space().ufl_element() == DG0.ufl_element()
    assert fields["eta_aux"].function_space().ufl_element() == DG0.ufl_element()
    assert fields["xi_T"].function_space().ufl_element() == DG0.ufl_element()

    assert 0 <= amr.countmark(mark) <= DG0.dim()
    assert eta_up > 0.0
    assert fields["eta_up"] == eta_up

    # Check all required keys in fields
    for k in (
        "eta",
        "eta_aux",
        "aux_eta2",
        "xi",
        "xi_T",
        "N2",
        "grad_zeta2",
        "comp_pairing",
        "c_aux",
        "c_cons",
        "c_comp",
        "C_P",
        "ethresh",
        "eta_up",
        "zeta",
        "cutoff",
        "constants",
    ):
        assert k in fields


def test_weighted_consistency():
    mesh, amr, CG1, DG0, u, psi, kappa, f = _setup_2d_obstacle(n=8)

    for cutoff in ("sco", "feco"):
        mark, fields, eta_up = amr.banzmark(
            u, bounds=(psi, None), kappa=kappa, f_ufl=f, cutoff=cutoff
        )
        with fields["eta"].dat.vec_ro as vec:
            local_sum = float(np.sum(vec.array_r ** 2))
        sum_eta2 = mesh.comm.allreduce(local_sum, op=MPI.SUM)
        assert abs(sum_eta2 - eta_up ** 2) < 1.0e-10


def test_sco_vs_feco():
    mesh, amr, CG1, DG0, u, psi, kappa, f = _setup_2d_obstacle(n=8)

    _, fields_sco, eta_up_sco = amr.banzmark(
        u, bounds=(psi, None), kappa=kappa, f_ufl=f, cutoff="sco"
    )
    _, fields_feco, eta_up_feco = amr.banzmark(
        u, bounds=(psi, None), kappa=kappa, f_ufl=f, cutoff="feco"
    )

    assert fields_sco["zeta"] is None
    assert fields_feco["zeta"] is not None
    assert eta_up_sco > 0.0
    assert eta_up_feco > 0.0

    # Minimizer property: Xi_eps(FECO) <= Xi_eps(SCO) (up to discretization tolerance)
    assert fields_feco["xi"] <= fields_sco["xi"] + 1.0e-8


def test_refinesbr2d_coupling():
    mesh, amr, CG1, DG0, u, psi, kappa, f = _setup_2d_obstacle(n=8)

    mark, fields, eta_up = amr.banzmark(
        u,
        bounds=(psi, None),
        kappa=kappa,
        f_ufl=f,
        cutoff="sco",
        method="total",
        theta=0.3,
    )
    nmarked = amr.countmark(mark)
    assert nmarked > 0

    rmesh = amr.refinesbr2D(mesh, mark)
    _, rDG0 = amr.spaces(rmesh)
    assert rDG0.dim() > DG0.dim()


# ---------------------------------------------------------------------------
# MPI parallel test
# ---------------------------------------------------------------------------


@pytest.mark.parallel(nprocs=2)
def test_banzmark_parallel():
    mesh, amr, CG1, DG0, u, psi, kappa, f = _setup_2d_obstacle(n=10)

    # Test Poincare constant
    cp = poincare_constant(mesh)
    assert abs(cp - 1.0 / np.pi) < 1.0e-12

    # Test SCO banzmark
    mark_sco, fields_sco, eta_up_sco = amr.banzmark(
        u,
        bounds=(psi, None),
        kappa=kappa,
        f_ufl=f,
        cutoff="sco",
        method="total",
        theta=0.5,
    )

    assert eta_up_sco > 0.0
    nmarked_sco = amr.countmark(mark_sco)
    assert 0 < nmarked_sco <= DG0.dim()

    # Weighted consistency in parallel for SCO
    with fields_sco["eta"].dat.vec_ro as vec:
        local_sum = float(np.sum(vec.array_r ** 2))
    total_sco = mesh.comm.allreduce(local_sum, op=MPI.SUM)
    assert abs(total_sco - eta_up_sco ** 2) < 1.0e-10

    # Test FECO banzmark in parallel
    mark_feco, fields_feco, eta_up_feco = amr.banzmark(
        u,
        bounds=(psi, None),
        kappa=kappa,
        f_ufl=f,
        cutoff="feco",
        method="total",
        theta=0.5,
    )

    assert eta_up_feco > 0.0
    nmarked_feco = amr.countmark(mark_feco)
    assert 0 < nmarked_feco <= DG0.dim()

    # Weighted consistency in parallel for FECO
    with fields_feco["eta"].dat.vec_ro as vec:
        local_sum = float(np.sum(vec.array_r ** 2))
    total_feco = mesh.comm.allreduce(local_sum, op=MPI.SUM)
    assert abs(total_feco - eta_up_feco ** 2) < 1.0e-10

    # Minimizer property in parallel
    assert fields_feco["xi"] <= fields_sco["xi"] + 1.0e-8
