import math
import numpy as np
import pytest
from firedrake import *
from viamr import VIAMR, BanzFields


def test_banz_constants():
    amr = VIAMR()
    c = amr._banz_constants(delta=0.5, epsilon=1.0 / 3.0)
    assert np.isclose(c["c"], 1.0)
    assert np.isclose(c["beta"], 1.0)
    assert np.isclose(c["c_gamma"], 1.0)
    assert np.isclose(c["chat_gamma"], 1.0)
    assert np.isclose(c["E"], 5.0)
    assert np.isclose(c["F"], 6.0)
    assert np.isclose(c["theta"], 1.5)
    assert np.isclose(c["c_aux"], 10.0)
    assert np.isclose(c["c_cons"], 36.0)
    assert np.isclose(c["c_comp"], 12.0)

    # Invalid delta
    with pytest.raises(ValueError):
        amr._banz_constants(delta=1.5)
    with pytest.raises(ValueError):
        amr._banz_constants(delta=-0.1)

    # Invalid epsilon
    with pytest.raises(ValueError):
        amr._banz_constants(delta=0.5, epsilon=0.8)  # max is 2/3 ≈ 0.6667
    with pytest.raises(ValueError):
        amr._banz_constants(delta=0.5, epsilon=-0.1)


def test_banz_poincare():
    amr = VIAMR()
    mesh1 = UnitSquareMesh(4, 4)
    cp1 = amr._banz_poincare(mesh1)
    assert np.isclose(cp1, 1.0 / np.pi)

    mesh2 = RectangleMesh(4, 4, 2.0, 3.0)
    cp2 = amr._banz_poincare(mesh2)
    assert np.isclose(cp2, 3.0 / np.pi)


def test_banz_1d_manufactured_exact():
    r"""Test the 1D manufactured example from Section 6 of the draft paper:
    Omega = (0, 1), f = -1, psi = 0, u = 0, lambda = 1.
    Candidate iterate: v(x) = 0.5 * x * (1 - x), kappa = 2.
    Then:
      -v'' = 1 = f + kappa -> auxiliary equation is satisfied exactly!
      R_Omega(v, kappa; v') = 0.
      q = v - psi = v >= 0 -> q_- = 0.
      kappa = 2 >= 0 -> kappa_- = 0, N_Omega = 0.
      comp_term = (kappa_+, q_+) = \int_0^1 2 * 0.5 * x * (1 - x) dx = 1/6.
      eta_up^2 = 10 * 0^2 + 36 * 0 + 36 * 0 + 12 * (1/6) = 2.0.
    """
    amr = VIAMR()
    mesh = IntervalMesh(20, 1.0)
    V = FunctionSpace(mesh, "CG", 2)
    x, = SpatialCoordinate(mesh)
    v = Function(V).interpolate(0.5 * x * (1.0 - x))
    kappa = Constant(2.0)
    phi = Constant(0.0)
    f = Constant(-1.0)

    mark, fields, Eh = amr.banzmark(
        v,
        (phi, None),
        kappa,
        f_ufl=f,
        tau="grad",
        C_P=1.0 / np.pi,
        quadrature_degree=8,
    )
    info = fields.info

    assert np.isclose(info["R"], 0.0, atol=1e-12)
    assert np.isclose(info["N2"], 0.0, atol=1e-14)
    assert np.isclose(info["grad_q_minus2"], 0.0, atol=1e-14)
    assert np.isclose(info["comp_term"], 1.0 / 6.0, rtol=1e-6)
    assert np.isclose(info["eta_up2"], 2.0, rtol=1e-6)
    assert np.isclose(Eh, np.sqrt(2.0), rtol=1e-6)
    assert fields.dominant_branch == "constraint"


def test_banz_2d_marking_and_fields():
    amr = VIAMR()
    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, "CG", 1)
    v = Function(V)
    kappa = Constant(1.0)
    phi = Constant(0.0)
    f = Constant(0.0)

    # Test "total" method (Doerfler marking)
    mark, fields, Eh = amr.banzmark(
        v, (phi, None), kappa, f_ufl=f, method="total", theta=0.5
    )
    assert isinstance(fields, BanzFields)
    assert "eta" in fields
    assert "load" in fields
    assert "nonpenetration" in fields
    assert "complementarity" in fields
    assert "info" in fields

    assert fields.eta_up == Eh
    assert fields.aux_part2 > 0.0
    assert isinstance(fields.summary(), str)
    assert len(fields.summary()) > 0

    DG0 = fields["eta"].function_space()
    assert mark.function_space() == DG0
    # Marks are binary {0, 1}
    mark_vals = mark.dat.data_ro
    assert np.all(np.isin(mark_vals, [0.0, 1.0]))
    assert np.sum(mark_vals) > 0

    # Test "max" method
    mark_max, _, Eh_max = amr.banzmark(
        v, (phi, None), kappa, f_ufl=f, method="max", theta=0.5
    )
    assert np.isclose(Eh, Eh_max)
    assert np.all(np.isin(mark_max.dat.data_ro, [0.0, 1.0]))


def test_banz_branch_dominance():
    amr = VIAMR()
    mesh = UnitSquareMesh(6, 6)
    V = FunctionSpace(mesh, "CG", 1)

    # Case 1: Large load f, zero constraint violation -> auxiliary dominated
    v1 = Function(V)
    kappa1 = Constant(0.0)
    phi1 = Constant(0.0)
    f1 = Constant(100.0)
    _, fields1, _ = amr.banzmark(v1, (phi1, None), kappa1, f_ufl=f1)
    assert fields1.dominant_branch == "auxiliary"

    # Case 2: Exact auxiliary (f + kappa = 0), but penetration and negative kappa
    x, y = SpatialCoordinate(mesh)
    v2 = Function(V).interpolate(-1.0 * x)
    kappa2 = Constant(-2.0)
    phi2 = Constant(0.0)
    f2 = Constant(2.0)
    _, fields2, _ = amr.banzmark(v2, (phi2, None), kappa2, f_ufl=f2, tau="grad")
    assert fields2.dominant_branch == "constraint"


def test_banz_health_monitors():
    amr = VIAMR()
    mesh = UnitSquareMesh(6, 6)
    V = FunctionSpace(mesh, "CG", 1)
    v = Function(V)
    kappa = Constant(0.5)
    phi = Constant(0.0)
    f = Constant(1.0)
    bcs = [DirichletBC(V, 0.0, "on_boundary")]

    # Check aux residual in Gram norm directly
    rho = amr.banz_aux_residual_gram_norm(v, kappa, f_load=f, bcs=bcs)
    assert rho > 0.0
    assert math.isfinite(rho)

    # Check moment defect
    recon = Function(V)
    d_hp = amr.banz_moment_defect(v, phi, recon=recon)
    assert np.isclose(d_hp, 0.0)

    # Check calling banzmark with bcs and recon passes through to info
    _, fields, _ = amr.banzmark(
        v, (phi, None), kappa, f_ufl=f, bcs=bcs, recon=recon
    )
    assert fields.info["rho"] is not None
    assert np.isclose(fields.info["rho"], rho)
    assert fields.info["d_hp"] is not None


def test_banz_refine_coupling():
    amr = VIAMR()
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    v = Function(V)
    kappa = Constant(1.0)
    phi = Constant(0.0)

    mark, _, _ = amr.banzmark(v, (phi, None), kappa, method="total", theta=0.5)
    initial_cells = mesh.num_cells()
    rmesh = amr.refinesbr2D(mesh, mark)
    assert rmesh.num_cells() > initial_cells


def test_banz_input_validation():
    amr = VIAMR()
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    v = Function(V)

    # Upper obstacle given -> raises NotImplementedError
    with pytest.raises(NotImplementedError):
        amr.banzmark(v, (Constant(0.0), Constant(1.0)), kappa=Constant(1.0))

    # Missing lower bound
    with pytest.raises(ValueError):
        amr.banzmark(v, (None, None), kappa=Constant(1.0))

    # Missing kappa
    with pytest.raises(ValueError):
        amr.banzmark(v, (Constant(0.0), None), kappa=None)

    # Invalid flux tau
    with pytest.raises(NotImplementedError):
        amr.banzmark(
            v, (Constant(0.0), None), kappa=Constant(1.0), tau="invalid"
        )


@pytest.mark.parallel(nprocs=2)
def test_banz_parallel():
    amr = VIAMR()
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    v = Function(V)
    kappa = Constant(1.0)
    phi = Constant(0.0)
    f = Constant(0.0)

    mark, fields, Eh = amr.banzmark(v, (phi, None), kappa, f_ufl=f, method="total")
    assert Eh > 0.0
    assert math.isfinite(Eh)
    # Check Poincaré constant agrees across ranks
    assert np.isclose(fields.info["C_P"], 1.0 / np.pi)
    # Check that marks are created on each rank
    assert mark.function_space().mesh() == mesh
