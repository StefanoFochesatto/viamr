# examples/

## overview

All Python codes in this directory demonstrate the methods from the paper.  The examples are organized, below, into those that solve the classical obstacle problem, which is unilateral and which has the uniformly-elliptic Laplacian as the operator, one code that solves a different uniformly-elliptic problem, and two important non-trivial applications which solve degenerate-diffusion obstacle problems.

The method acryonyms are as follows, with complete discussions given in the text of the paper:
  * AMR = adaptive mesh refinement
  * AVM = averaged-metric mesh adaptation
  * BR = a posteriori estimator from Babuska & Rheinboldt (1989)
  * GR = a posteriori estimator based on gradient recovery
  * NSV = a posteriori estimator and AMR method from Nochetto, Siebert, and Veeser (2003)
  * UDO = unstructured dilation operator, for refinement near the free boundary
  * VCD = variable-coefficient diffusion, for refinement near the free boundary
  * VI = variational inequality

See the paper for citations to the literature.

## the examples

### 1. classical obstacle problem examples

The short programs `aol.py`, `sphere.py`, and `spiral.py` might be the examples to start with.  They show many core abilities of the `VIAMR` class.  They are all basic model problems because they apply adaptive refinement to the classical obstacle problem, where the operator is the uniformly-elliptic Laplacian.

  * `aol.py` solves a problem from Ainsworth, Oden, & Lee (1993).  This simple example is quoted in full in the paper, and it produces a figure there.  It only does one level of refinement.

  * `sphere.py` solves a radially-symmetric problem from Chapter 12 of Bueler (2021).  Four algorithms are applied by default: UDO+BR, NSV, uniform refinement, and AVM.  In each case we refine an initially homogeneous mesh.  Some methods mark elements for refinement near the free boundary, and all methods refine in the inactive set.  Note that the AVM method depends on the [animate](https://github.com/mesh-adaptation/animate) library; see below.  The target complexity set at the start of `sphere.py` is intended to generate a (more or less) apples-to-apples comparison of the methods.  View the `gap` variable in the output Paraview files to see the active, inactive, and free boundary sets.  See the `error` variable to see the distribution of numerical error.

  * `spiral.py` does a similar comparison to `sphere.py`, but on a classical obstacle problem from Graeser & Kornhuber (2009).  Only 2 methods are demonstrated by default, namely: UDO+BR and NSV.

These examples also solve the classical obstacle problem:

  * `blisters.py` solves a classical obstacle problem which generates a large active set, covering more than 80% of the domain.  Resolving the connectedness of the inactive set in this example requires high resolution.

  * `nsv.py` compares UDO+BR to the method from Nochetto, Siebert, & Veeser (2003).  The problem solved is "7.2 Example: Constant Obstacle" from that source.

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
