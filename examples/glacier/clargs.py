des = """
Solves a 2D steady, isothermal shallow ice approximation glacier obstacle problem.
See README.md for examples and METHOD.md for documentation of the mathematics.

The domain is a square [0,L]^2 with L = 1800.0 km, except that with option -bdata
the domain is read from the file.

The default problem (option -prob cap) uses a bumpy bed topography, while the
surface mass balance is radially-symmetric (depends only on horizontal location).
Option -prob range generates a different SMB, which results in a disconnected
glacier.  Option -bdata reads the bed elevation from a NetCDF (.nc) file.
An elevation-dependent surface mass balance model is turned on with -elevdepend.
Set -sELA for equilibrium line altitude.

We apply vinewtonrsls + mumps, a VI-adapted direct-solver Newton method as the
PETSc solver.  For -elevdepend runs add -picard, which wraps an outer Picard
iteration around the Newton solver, to iterate on the surface-elevation dependence.

We apply the UDO (n=1 by default) method for free-boundary refinement.  To this
we add the weighted "BV00" form of the BR78 residual estimator in the inactive set.

For a flat-bed "dome" case with known exact solution, see dome.py instead.
"""

from argparse import ArgumentParser, RawTextHelpFormatter

parser = ArgumentParser(description=des, formatter_class=RawTextHelpFormatter)

parser.add_argument(
    "-bdata",
    metavar="FILE",
    type=str,
    default="",
    help='read b(x,y) data from "topg" variable in this NetCDF file (.nc)',
)
parser.add_argument(
    "-box",
    metavar="X",
    type=float,
    nargs=4,
    default=[0.0, 1800.0e3, 0.0, 1800.0e3],
    help="bounding box for -opvdsub output",
)
parser.add_argument(
    "-elevdepend",
    action="store_true",
    default=False,
    help="compute surface mass balance from an elevation-dependent model",
)
parser.add_argument(
    "-hmin",
    type=float,
    default=-1,
    help="do not refine below this diameter (default: -1; ignores hmin)",
)
parser.add_argument(
    "-m",
    type=int,
    default=20,
    metavar="M",
    help="number of cells in each direction on initial mesh [default=20]",
)
parser.add_argument(
    "-opvd",
    metavar="FILE",
    type=str,
    default="result.pvd",
    help="name for Paraview-format output file (.pvd)",
)
parser.add_argument(
    "-opvdsub",
    metavar="FILE",
    type=str,
    default="",
    help="output file (.pvd) into which we extract a submesh defined by -box",
)
parser.add_argument(
    "-picard",
    action="store_true",
    default=False,
    help="use Picard iteration, a wrapper around the Newton solver; required for -elevdepend",
)
parser.add_argument(
    "-pcount",
    type=int,
    default=10,
    metavar="P",
    help="number of Picard frozen-tilt (and a(s) if -elevdepend) iterations [default=10]",
)
parser.add_argument(
    "-primal",
    type=str,
    default="s",
    metavar="X",
    choices=["s", "u"],
    help="primal variable: s (surface elevation) or u (transformed thickness) [default=s]",
)
parser.add_argument(
    "-prob",
    type=str,
    default="cap",
    metavar="X",
    choices=["cap", "range"],
    help="problem {cap, range} [default=cap; ignored for -bdata]",
)
parser.add_argument(
    "-refine",
    type=int,
    default=2,
    metavar="R",
    help="number of AMR refinements [default=2]",
)
parser.add_argument(
    "-sELA",
    type=float,
    default=1000.0,
    metavar="X",
    help="equilibrium line altitude to use if -elevdepend [default=1000.0]",
)
parser.add_argument(
    "-theta",
    type=float,
    default=0.5,
    metavar="X",
    help="theta in fixed-rate marking strategy in inactive set [default=0.5]",
)
parser.add_argument(
    "-udo_n",
    type=int,
    default=1,
    metavar="N",
    help="use udomark(.., n=N) [default=1]",
)
parser.add_argument(
    "-uniform",
    type=int,
    default=0,
    metavar="R",
    help="initial R refinements are uniform [default=0]",
)
