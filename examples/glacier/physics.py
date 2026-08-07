# Physical formulas for glaciers, and associated constants.

import firedrake as fd
from pyop2.mpi import MPI

debug = False  # debugging solver failures

# physical constants
secpera = 31556926.0  # seconds per year
n = 3.0  # exponent in Glen-Nye flow law
g = 9.81  # acceleration of gravity on earth
rho = 910.0  # density of ice
A = 1.0e-16 / secpera  # ice softness in flow law

# derived constants
p = n + 1  # typical:  p = 4
Gamma = 2 * A * (rho * g) ** n / (n + 2)  # coefficient in shallow ice approximation

# private constants of transformed SIA
_omega = (p - 1) / (2 * p)  #  \omega = 3/8
_mu = (p + 1) / (2 * p)  #  \mu = 5/8
_r = p / (p - 1)  #  r = 4/3


def scalarrange(w):
    """Utility function to return the range of a generic scalar field.  Correct in parallel."""
    locmin = w.dat.data.min()
    locmax = w.dat.data.max()
    gmin = w.function_space().mesh().comm.allreduce(locmin, op=MPI.MIN)
    gmax = w.function_space().mesh().comm.allreduce(locmax, op=MPI.MAX)
    return gmin, gmax


def admissible(w, psi):
    """Admissibility predicate for w>=psi.  Correct in parallel.
    Check nonnegativity by admissible(w, Constant(0.0))."""
    delta = fd.Function(w.function_space()).interpolate(w - psi)
    locmin = delta.dat.data.min()
    gmin = delta.function_space().mesh().comm.allreduce(locmin, op=MPI.MIN)
    return gmin >= 0.0


def u2H(u):
    return u ** _omega


def H2u(H):
    return H ** (1.0 / _omega)


def Phi_u(u, b):
    """Computes the tilting effect of the bed gradient:
      \Phi(u) = (1/\omega) u^\mu \grad b.
    In fact we regularize the power to avoid NaN or Inf from the fractional power:
      u^\mu --> (u + eps)^\mu."""
    eps = 1.0  # eps=1 regularization is small; typical u is 1e3 to 1e9
    return (1.0 / _omega) * (u + eps) ** _mu * fd.grad(b)


def weakform_u(u, a, b, Z=None, epsreg=0.01, qdegree=5):
    """Return the nonlinear weak form for the u (transformed thickness) form
    of the shallow ice approximation.

    When Z=None this is the weak form corresponding to (3) in METHOD.md.
    If Z is given then this is (4).  In either case a(x) is given.  (Elevation
    -dependent surface mass balance *must* be handled by an outer iteration.)

    Even for steep beds, setting the quadrature degree in the weak form, i.e.
    "dx(degree=Q)" with Q=4,5,6,7, seems to produce about the same result as
    the automatic mechanism.  However, Q=2 is distinctly worse."""
    v = fd.TestFunction(u.function_space())
    if Z is not None:
        du_tilt = fd.grad(u) + Z
    else:
        du_tilt = fd.grad(u) + Phi_u(u, b)
    if debug:
        dumag = fd.Function(u.function_space()).interpolate(
            fd.inner(du_tilt, du_tilt) ** 0.5
        )
        dumin, dumax = scalarrange(dumag)
        print(f"  DEBUG: {dumin:.2e} <= |du_tilt| <= {dumax:.2e}")
    Dp = (fd.inner(du_tilt, du_tilt) + epsreg) ** ((p - 2) / 2)
    C = Gamma * _omega ** (p - 1)
    return (
        C * Dp * fd.inner(du_tilt, fd.grad(v)) * fd.dx(degree=qdegree) - a * v * fd.dx
    )


def residual_u_ufl(u, a, b, epsreg = 0.01):
    """Return a UFL expression for the strong form residual of the
    u (transformed thickness) form of the shallow ice approximation.
    Also return the coefficient Z."""
    du_tilt = fd.grad(u) + Phi_u(u, b)
    Dp = (fd.inner(du_tilt, du_tilt) + epsreg) ** ((p - 2) / 2)
    C = Gamma * _omega ** (p - 1)
    Z = C * Dp
    res = -fd.div(Z * du_tilt) - a  # formally a Poisson equation here
    return res, Z


def weakform_s(s, a, b, qdegree=5, epsplap=1.0e-4, epsH=20.0):
    """Return the nonlinear weak form for the s (transformed thickness) form
    of the shallow ice approximation.  There are two regularizations in the
    degenerate-diffusivity coefficient Dp: epsplap is a unitless surface slope
    regularization, and epsH in meters is a thickness regularization."""
    v = fd.TestFunction(s.function_space())
    gs = fd.grad(s)
    dsp = (fd.inner(gs, gs) + epsplap ** 2) ** ((p - 2) / 2)
    Z = Gamma * (s - b + epsH) ** (p + 1) * dsp
    return Z * fd.inner(gs, fd.grad(v)) * fd.dx(degree=qdegree) - a * v * fd.dx


def NOTUSED_residual_s_ufl(s, a, b, epsplap=1.0e-4, epsH=20.0):
    """Return a UFL expression for the strong form residual of the
    s (surface elevation) form of the shallow ice approximation.
    Also return the coefficient Z."""
    gs = fd.grad(s)
    dsp = (fd.inner(gs, gs) + epsplap ** 2) ** ((p - 2) / 2)
    Z = Gamma * (s - b + epsH) ** (p + 1) * dsp
    res = fd.conditional(s > b + epsH, - fd.div(Z * gs) - a, fd.Constant(0.0))  # formally a Poisson equation here
    return res, Z


def surfacevelocity(s, H):
    CU = ((n + 2) / (n + 1)) * Gamma
    dss = fd.inner(fd.grad(s), fd.grad(s))
    return CU * H ** p * dss ** ((p - 2) / 2) * fd.grad(s)
