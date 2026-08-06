des = """
Solves a 2D steady, isothermal shallow ice approximation glacier obstacle problem.
See README.md for examples and METHOD.md for documentation of the mathematics.

The domain is a square [0,L]^2 with L = 1800.0 km, except that with option -bdata
the domain is read from the file.

By default (-prob cap) we use a random, but smooth, bed topography, while the
surface mass balance is radially-symmetric and depends on horizontal location.
Option -prob range generates a different SMB; results in a disconnected
glacier.  Option -prob dome solves a flat bed case with radially-symmetric
surface mass balance, where the exact solution is known and the numerical error
is reported.  Option -bdata reads the bed elevation from a NetCDF (.nc) file.

An elevation-dependent surface mass balance model is turned on with -elevdepend.
Set -sELA for equilibrium line altitude.  This case does not allow -newton.

We apply the UDO method for free-boundary refinement.  The default mode
does n=1 UDO at the free boundary, plus gradient-recovery estimation in the
inactive set.

The default VI solver is Picard iteration on the tilt; see (Jouvet & Bueler, 2012).
We apply vinewtonrsls (+ mumps) for each tilt.  A full Newton iteration, simply
vinewtonrsls, is turned on with -newton, but it may not converge in harder cases.
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
    "-newton",
    action="store_true",
    default=False,
    help="use straight Newton instead of Picard+Newton",
)
parser.add_argument(
    "-ocsv",
    metavar="FILE",
    type=str,
    default="",
    help="output file name for dome error report (.csv)",
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
    "-pcount",
    type=int,
    default=10,
    metavar="P",
    help="number of Picard frozen-tilt (and a(s) if -elevdepend) iterations [default=10]",
)
parser.add_argument(
    "-prob",
    type=str,
    default="cap",
    metavar="X",
    choices=["cap", "range", "dome"],
    help="choose problem from {cap, range, dome}",
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
    help="theta to use in fixed-rate marking strategy in inactive set [default=0.5]",
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
