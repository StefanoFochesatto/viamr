# Formulas for synthetic glacier examples.
#
# The "dome" case is a manufactured solution from e.g. Bueler (2016).
# Here dome_s_ufl() is the exact surface, and dome_a_ufl() is the SMB.
# Thus (b=0, dome_a_ufl(), dome_s_ufl()) exactly solve the VI (PDE) by
# construction.
#
# Note that steady.py imports dome_a_ufl() for -prob cap|range, only as
# a plausible-looking radially-symmetric forcing function.  Also steady.py
# optionally uses model_a_ufl() under option -elevdepend.
#
# The bumpy bed formula bumps_b_ufl() is used by steady.py too.

import numpy as np
from pyop2.mpi import MPI
import firedrake as fd
from firedrake.petsc import PETSc
from physics import secpera, n, Gamma, H2u

# constants (same for all problems)
L = 1800.0e3  # domain is [0,L]^2, with fields centered at (xc,xc)
xc = L / 2

# dome parameters
dome_L = 750.0e3
dome_H0 = 3600.0


# exact solution for surface elevation s, and thickness H, to dome problem
def dome_s_ufl(x, n=3.0):
    # https://github.com/bueler/sia-fve/blob/master/petsc/base/exactsia.c#L83
    r = fd.sqrt(fd.dot(x - fd.as_vector([xc, xc]), x - fd.as_vector([xc, xc])))
    mm = 1 + 1 / n
    qq = n / (2 * n + 2)
    CC = dome_H0 / (1 - 1 / n) ** qq
    z = r / dome_L
    tmp = mm * z - 1 / n + (1 - z) ** mm - z ** mm
    expr = CC * tmp ** qq
    return fd.conditional(fd.lt(r, dome_L), expr, 0)


# accumulation a for dome problem
def dome_a_ufl(x, n=3.0, problem="cap"):
    # https://github.com/bueler/sia-fve/blob/master/petsc/base/exactsia.c#L51
    R = fd.sqrt(fd.dot(x - fd.as_vector([xc, xc]), x - fd.as_vector([xc, xc])))
    r = fd.conditional(fd.lt(R, 0.01), 0.01, R)
    r = fd.conditional(fd.gt(r, dome_L - 0.01), dome_L - 0.01, r)
    s = r / dome_L
    C = dome_H0 ** (2 * n + 2) * Gamma / (2 * dome_L * (1 - 1 / n)) ** n
    pp = 1 / n
    tmp1 = s ** pp + (1 - s) ** pp - 1
    tmp2 = 2 * s ** pp + (1 - s) ** (pp - 1) * (1 - 2 * s) - 1
    a0 = (C / r) * tmp1 ** (n - 1) * tmp2
    if problem == "range":
        dxc = x[0] - xc
        dyc = x[1] - xc
        dd = L / 30
        aneg = -3.0e-8  # roughly the min of a0
        return fd.conditional(
            fd.gt(R, dome_L),
            a0,
            fd.conditional(
                fd.lt(dxc ** 2, (1.9 * dd) ** 2),
                aneg,
                fd.conditional(fd.lt(dyc ** 2, (1.1 * dd) ** 2), aneg, a0),
            ),
        )
    else:
        return a0


def dome_normerrors(uh, Hh):
    """Returns:
      * H^1 seminorm error in u
      * relative H^1 norm error in u,
      * L^infty norm error in H,
      * uexact itself (in CG2).
    For L^infty error in H we want the max nodal error, so V=CG1 is fine."""
    V = uh.function_space()
    x = fd.SpatialCoordinate(V.mesh())
    Hdiff = fd.Function(V).interpolate(Hh - dome_s_ufl(x))
    with Hdiff.dat.vec_ro as v:
        HerrLinf = abs(v).max()[1]
    CG2 = fd.FunctionSpace(V.mesh(), "CG", 2)
    uexact = H2u(dome_s_ufl(x))
    dus = fd.inner(fd.grad(uexact - uh), fd.grad(uexact - uh))
    uerrH1semi = fd.assemble(dus * fd.dx(degree=6)) ** 0.5
    uerrH1rel = fd.errornorm(uexact, uh, norm_type="H1") / fd.norm(
        uexact, norm_type="H1"
    )
    return uerrH1semi, uerrH1rel, HerrLinf


def dome_radiuserror(mesh, vfb):
    """For -prob "dome", compute the maximum of free-boundary radius errors
    from the output of VIAMR.freeboundarygraph2D().  The exact free boundary
    is a circle of radius domeL with center (L/2,L/2).  Returns the maximum
    radius error."""
    vfb = np.array(vfb)
    mymax = PETSc.NINFINITY
    if len(vfb) > 0:
        x, y = vfb[:, 0], vfb[:, 1]
        drfb = np.abs(np.sqrt((x - L / 2) ** 2 + (y - L / 2) ** 2) - dome_L)
        mymax = np.max(drfb)
    return float(mesh.comm.allreduce(mymax, op=MPI.MAX))


# bed elevation b for cap and range problems
def bumps_b_ufl(x, problem="cap"):
    if problem == "range":
        B0 = 400.0  # (m); amplitude of bumps
    else:
        B0 = 200.0  # (m); amplitude of bumps
    xx, yy = x[0] / L, x[1] / L
    b = (
        +5.0 * fd.sin(fd.pi * xx) * fd.sin(fd.pi * yy)
        + fd.sin(fd.pi * xx) * fd.sin(3 * fd.pi * yy)
        - fd.sin(2 * fd.pi * xx) * fd.sin(fd.pi * yy)
        + fd.sin(3 * fd.pi * xx) * fd.sin(3 * fd.pi * yy)
        + fd.sin(3 * fd.pi * xx) * fd.sin(5 * fd.pi * yy)
        + fd.sin(4 * fd.pi * xx) * fd.sin(4 * fd.pi * yy)
        - 0.5 * fd.sin(4 * fd.pi * xx) * fd.sin(5 * fd.pi * yy)
        - fd.sin(5 * fd.pi * xx) * fd.sin(2 * fd.pi * yy)
        - 0.5 * fd.sin(10 * fd.pi * xx) * fd.sin(10 * fd.pi * yy)
        + 0.5 * fd.sin(19 * fd.pi * xx) * fd.sin(11 * fd.pi * yy)
        + 0.5 * fd.sin(12 * fd.pi * xx) * fd.sin(17 * fd.pi * yy)
    )
    return B0 * b


def model_a_ufl(s, sELA=1000.0, dsNEXT=100.0, alpha=0.0001 / secpera, alpharat=0.01):
    """Model of surface mass balance a(s) where alpha is lapse rate below sELA
    and above sELA there is a lower-slope (by alpharat) logarithmic function."""
    tau = dsNEXT - sELA
    beta = alpharat * alpha * dsNEXT
    return fd.conditional(
        s < sELA, alpha * (s - sELA), beta * (fd.ln(s + tau) - fd.ln(dsNEXT))
    )
