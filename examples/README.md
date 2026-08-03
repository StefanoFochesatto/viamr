# examples/

## overview

The Python example programs in this directory demonstrate the methods from the paper.

In this document, examples are organized by the problems they solve.  First are those that solve the classical obstacle problem, which is unilateral with the uniformly-elliptic Laplacian as the operator.  One other example solves a uniformly-elliptic problem, but it is advective and bilateral.  Finally, two important non-trivial applications solve degenerate-diffusion obstacle problems.

For reading the codes and this document, some acryonyms are as follows; see the text of the paper for more detail:
  * AMR = adaptive mesh refinement
  * AVM = averaged-metric mesh adaptation
  * BR = a posteriori estimator from Babuska & Rheinboldt (1989)
  * GR = a posteriori estimator based on gradient recovery
  * NSV = a posteriori estimator and AMR method from Nochetto, Siebert, and Veeser (2003)
  * UDO = unstructured dilation operator, for refinement near the free boundary
  * VCD = variable-coefficient diffusion, for refinement near the free boundary
  * VI = variational inequality


## the examples

### 1. classical obstacle problem examples

The short program `aol.py` might be the one example to start with.

  * `aol.py` solves a problem from Ainsworth, Oden, & Lee (1993).  This simple example is quoted in full in the paper, and it produces a figure there.  It only does one level of refinement.

Next see the richer examples  `sphere.py` and `spiral.py`.  These examples together show most abilities of the `VIAMR` class.  They are all apply adaptive refinement to the classical obstacle problem, wherein the operator is the uniformly-elliptic Laplacian and the inequality is a unilateral lower bound.

  * `sphere.py` solves a radially-symmetric problem from Chapter 12 of Bueler (2021).  Four algorithms are applied by default: UDO+BR, NSV, uniform refinement, and AVM.  In each case we refine an initially homogeneous mesh to a target complexity level.  Some methods mark elements for refinement near the free boundary, and all methods refine in the inactive set.  Note that the AVM method depends on the [animate](https://github.com/mesh-adaptation/animate) library; see below.  The target complexity settings are intended to generate (more or less) apples-to-apples comparison of the methods.  View the `gap` variable in the output Paraview files to see the active, inactive, and free boundary sets.  See the `error` variable to see the distribution of numerical error.

  * `spiral.py` does a similar comparison to `sphere.py`, but on a classical obstacle problem from Graeser & Kornhuber (2009).  No exact solution is known.  Only UDO+BR and NSV methods are demonstrated (by default).

These 2 addtional examples also solve the classical obstacle problem:

  * `nsv.py` compares UDO+BR to the NSV method from Nochetto, Siebert, & Veeser (2003).  The problem solved is "7.2 Example: Constant Obstacle" from that reference.  A convergence figure is generated, which is explained in the paper.

  * `blisters.py` solves a classical obstacle problem which generates a large active set, covering more than 80% of the domain.  Resolving the connectedness of the inactive set in this example requires high resolution.

### 2. other uniformly-elliptic examples

This example demonstrates bilateral bounds.

  * `pollutant.py` solves a 3D advection-diffusion problem with upper and lower bounds $0 \le u \le 1$ on the solution $u$.  This problem is the d=3 case of Example 8.3 in Bueler & Farrell (2024).  The operator is the Laplacian plus advection by a divergence-free wind field.  As the Peclet number is O(100), it is advection-dominated.  The solver uses $n=1$ UDO AMR on the upper- and lower-constraint free boundaries.

### 3. examples with degenerate operators

A major purpose of the paper is to extend AMR methods for obstacle problems to degenerate operators of porous-media type, including the example of primary interest, for glacier geometry.

  * `porous.py` solves a 2D steady-state porous-media model problem, for which the exact solution is known.  Only UDO+BR is applied.

  * The `glaciers/` directory contains a highly-nontrivial degenerate-operator example based on a realistic problem.  See the paper, and `glaciers/METHOD.md`, for the mathematical problem and the FE solution method.  See `glaciers/README.md` for how to run the example.  The main code is `glaciers/steady.py`.


## running the examples

First install VIAMR using instructions in the [README.md in the parent directory](../README.md).  Make sure to activate the Firedrake virtual environment.

Then simply run, for example:

```
python3 sphere.py
```

All of the codes write `.pvd` files for viewing in Paraview.  Many unilateral codes write a `gap` field, the difference between the solution and the obstacle, and looking at that field, especially with a tight threshold, will show the active and inactive sets most easily.


### cleaning up

Clean up all `result*` files and subdirectories etc. with

```
make clean
```
