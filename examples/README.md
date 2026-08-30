# examples/

## overview

The Python example programs in this directory demonstrate the methods from the paper.

Examples are organized below by 3 categories of problems they solve.

  1. Those that solve the _classical obstacle problem_, which is unilateral with the (uniformly-elliptic) Laplacian as the operator.  Theoretical results are concentrated on this model problem.
  2. Another example, also uniformly-elliptic, but _advective_ and _bilateral_.
  3. Examples which solve _degenerate-diffusion_ obstacle problems, including the important glacier application.

### acronyms (and citations)

For reading the codes, and this document, some acryonyms and citations, listed alphabetically, are as follows:

  * AMR = _adaptive mesh refinement_
  * AVM = _averaged-metric_ mesh adaptation
  * BR = a posteriori estimator, for computed inactive sets, from Babuska & Rheinboldt (1989)
  * BV = similar to BR, but weighted following Bernardi & Verfurth (2000)
  * GR = a posteriori estimator based on _gradient recovery_
  * NSV = a posteriori estimators from Nochetto, Siebert, and Veeser (2003, 2005)
  * UDO = _unstructured dilation operator_, for refinement near the free boundary
  * VCD = _variable-coefficient diffusion_, a UDO variant for refinement near the free boundary
  * VI = _variational inequality_

See the text of the paper for more details.

## the examples

### 1. classical obstacle problem

The short program `aol.py` might be the starting point:

  * `aol.py` solves a problem from Ainsworth, Oden, & Lee (1993).  This simple example is quoted in full, and produces a figure, in the paper.  It only does one level of refinement.

Next see the richer examples `sphere.py` and `spiral.py`, which show most methods implemented by the `VIAMR` class:

  * `sphere.py` solves a radially-symmetric problem from Chapter 12 of Bueler (2021).  Five algorithms are applied by default: UDO+BR, NSV03,, NSV05, uniform refinement, and AVM.  In each case we refine an initially homogeneous mesh to a target complexity level.  Note that the AVM method depends on the [animate](https://github.com/mesh-adaptation/animate) library; see below.  The target complexity settings are intended to generate (more or less) apples-to-apples comparison of the methods.  View the `gap` variable in the output Paraview files to see the active, inactive, and free boundary sets.  See the `error` variable to see the distribution of numerical error.

  * `spiral.py` does a similar comparison to `sphere.py`, but on a classical obstacle problem from Graeser & Kornhuber (2009).  No exact solution is known.

These 2 addtional examples also solve the classical obstacle problem:

  * `nsv.py` compares UDO+BR to the NSV methods from Nochetto, Siebert, & Veeser (2003, 2005).  Option `-prob easy`, the default, solves   Only UDO+BR and NSV methods are demonstrated (by default)."7.2 Example: Constant Obstacle" from NSV03; it has a known exact solution, so convergence and effectivity figures are generated.  Option `-prob pyramid` solves the diamond-domain, pyramid-obstacle problem from subsection 3.2 of NSV05.  It has no known exact solution, so it only reports the estimators and plots their decay.

  * `blisters.py` solves a classical obstacle problem which generates a large active set, covering more than 80% of the domain.  Resolving the connectedness of the inactive set in this example requires high resolution.

### 2. other uniformly-elliptic examples

This example demonstrates bilateral bounds.

  * `pollutant.py` solves a 3D advection-diffusion problem with upper and lower bounds $0 \le u \le 1$ on the solution $u$.  This problem is the d=3 case of Example 8.3 in Bueler & Farrell (2024).  The operator is the Laplacian plus advection by a divergence-free wind field.  It is advection-dominated (Peclet number is O(100)).  The solver uses $n=1$ UDO+BR AMR on the upper- and lower-constraint free boundaries.

### 3. examples with degenerate operators

A major purpose of the paper is to extend AMR methods for obstacle problems to degenerate operators of porous-media type, including the example of primary interest, for glacier geometry.

  * `porous.py` solves a 2D steady-state porous-media model problem, for which the exact solution is known.  Only UDO+BV is applied.

  * The `glaciers/` directory contains a highly-nontrivial degenerate-operator example, including a realistic glacier application.  See the paper, and `glaciers/METHOD.md`, for the mathematical problem and the FE solution method.  See `glaciers/README.md` for how to run the example.  The main code is `glaciers/steady.py`.


## running the examples

First install VIAMR using instructions in the [README.md in the parent directory](../README.md).  Make sure to activate the Firedrake virtual environment.

Then simply run, for example:

```
python3 sphere.py
```

All of the codes write `.pvd` files for viewing in Paraview.  Some codes generate `.png` convergence or effectivity result images.  Many unilateral codes write a `gap` field, the difference between the solution and the obstacle, and looking at that field, especially with a tight threshold, will show the active and inactive sets most easily.


### cleaning up

Clean up all `result*` files and subdirectories etc. with

```
make clean
```
