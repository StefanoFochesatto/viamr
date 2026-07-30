# examples/

## overview of the examples

### 1. classical obstacle problem examples

The short programs `sphere.py`, `spiral.py`, and `aol.py` show many core abilities of the `VIAMR` class.  They are all basic model problems because they apply adaptive refinement to the classical obstacle problem, where the operator is the uniformly-elliptic Laplacian.

  * `sphere.py` refines an initially homogeneous mesh on a radially-symmetric problem from Chapter 12 of Bueler (2021).  Five algorithms are applied namely, UDO+BR, NSV, uniform refinement, and AVM.  The NSV method is from Nochetto, Siebert, and Veeser (2003).  Some methods mark elements for refinement near the free boundary, and in all methods refinement in the inactive set also occurs.  Note that the AVM method depends on the [animate](https://github.com/mesh-adaptation/animate) library; see below.  The target complexity set at the start of `sphere.py` is intended to generate a (more or less) apples-to-apples comparison of the methods.  View the `gap` variable in the output Paraview files to see the active, inactive, and free boundary sets.  See the `error` variable to see the distribution of numerical error.

  * `spiral.py` does a similar comparison on a classical obstacle problem from Graeser & Kornhuber (2009).  Only 2 methods are demonstrated by default, namely: UDO+BR and NSV.

  * `aol.py` is a simple example that only does one level of refinement on a problem from

    M. Ainsworth, J.T. Oden, and C. Lee, Local a posteriori error estimators for variational inequalities, Numerical Methods for Partial Differential Equations 9 (1993), pp. 23–33.

  This code is quoted in full in the paper, and it produces a figure therein.

Several more examples also solve the classical obstacle problem:

  * `blisters.py` solves a classical obstacle problem which generates a large active set, covering more than 80% of the domain.  Resolving the connectedness of the inactive set in this example requires high resolution.

  * `nsv.py` compares UDO+BR to the method from

    Nochetto, R. H., Siebert, K. G., & Veeser, A. (2003). Pointwise a posteriori error control for elliptic obstacle problems. Numerische Mathematik, 95(1), 163-195.

  on "7.2 Example: Constant Obstacle" from that source.

FIXME `parabola1d.py1`, `suttmeier.py`


### 2. other uniformly-elliptic examples

FIXME `pollutant.py`


### 3. examples with degenerate operators

FIXME `porous.py`

The `glaciers/` directory contains another example; see `glaciers/README.md` and `glaciers/METHOD.md` for what it is doing and how to run it.


## running the examples

First install VIAMR using instructions in the [README.md in the parent directory](../README.md).  Make sure to activate the Firedrake virtual environment.  Then simply run, for example:

```
python3 sphere.py
```

All of the codes write `.pvd` files for viewing in Paraview.  Often the `gap` field will show the active and inactive sets most easily.


### cleaning up

Clean up all `result*` files and subdirectories etc. with

```
make clean
```
