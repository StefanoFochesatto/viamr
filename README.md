# VIAMR

This repository contains Python [Firedrake](https://www.firedrakeproject.org) algorithms for adaptive mesh refinement (AMR) and mesh adaptation for variational inequalities (VIs).  The constraint set for these problems must be defined by a lower- and upper-bound inequalities.

In describing algorithms we use the language of _active_ and _inactive_ sets.  In an _active set_, also known as a _coincidence set_, one of the bound inequalities holds as an equality.  In the _inactive set_ the constraints are strict inequalities, so the solution satisfies a partial differential equation (PDE).

Our primary AMR goals in this context are to generate rapid convergence in solution norm, to generated accurate computed free boundaries, and to be able to measure geometrical errors in free boundaries and active sets.

Our library defines the class `VIAMR` in `viamr/viamr.py`.  It bundles 3 kinds of strategies for deciding _where_ to refine: free-boundary-proximity heuristics, classical residual/jump estimators applied only in inactive sets, and a whole-domain estimator designed for VIs.  These methods produce DG0 indicators (markings), with $\{0,1\}$ values, which can be combined by unioning if desired.  Markings can be fed to 2 tag-and-refine skeleton-based mesh refinement methods, [PETSc's](https://petsc.org/release/) (limited to 2D) or [Netgen's](https://ngsolve.org/).  All of these algorithms are parallel, with excellent weak scaling.

Metric-based mesh adaptation, i.e. re-meshing, is also supported, via the [animate](https://github.com/mesh-adaptation/animate).

Solution-norm error is a standard way to evaluate quality, but the library supports a diagnostic layer which measures active-set geometrical error (Jaccard distances) and free-boundary location accuracy (Hausdorff metric).  Free-boundary accuracy is often a goal for computations using variational inequalities.

These codes extend S. Fochesatto (2024). _Adaptive mesh refinement for variational inequalities_, Master of Science project, UAF.  They are the subject of a paper in progress.

## Dependencies

To get started, install Firedrake following the instructions at the [Firedrake install page](https://www.firedrakeproject.org/install.html#).

Now activate the virtual environment (venv). Typically something like:

```
source ~/venv-firedrake/bin/activate
```

Now pip install [shapely](https://pypi.org/project/shapely/), vtk, and [ngspetsc](https://github.com/NGSolve/ngsPETSc) in the venv:

```
pip install vtk ngspetsc shapely
```

To use metric-based mesh adaptation, the [animate](https://github.com/mesh-adaptation/animate) library is used.  To install this follow the instructions at the [installation wiki page](https://github.com/mesh-adaptation/docs/wiki/Installation-Instructions).

## Installation

### clone

Clone the VIAMR repository and enter the directory

```
git clone https://github.com/StefanoFochesatto/viamr.git
cd viamr/
```

### install

Either install editable with pip:

```
pip install -e .
```
or plain:

```
pip install .
```

### Using Docker

A docker image is available, with most of the setup complete. To get started ensure that you have [Docker](https://docs.docker.com/engine/install/) installed and running on your system.

Pull the Docker image:

```
docker pull stefanofochesatto/viamr:latest
```

Run a Docker container from the image:

```
docker run --rm -it -v ${HOME}:${HOME} stefanofochesatto/viamr:latest
```

FYI: The `--rm` flag will remove the container once it exits. The `-it` flag runs the container with an interactive shell environment (`ctrl + d` to exit). Finally `-v ${HOME}:${HOME}` is giving the container access to your `HOME` directory so you can navigate your files within the interactive shell environment.

Once the docker container is up and running, you can activate the python environment as usual. You'll also want to reinstall VI-AMR as the docker image was built with a previous version of the library. (automatic builds are low priority)

## Usage

These basic examples demonstrate refinement with the UDO, NSV and AVM methods.  First make sure that the firedrake virtual environment is active.  Then do:
```
cd examples/
python3 sphere.py
python3 spiral.py
```

The sphere problem has a known exact solution while the spiral problem does not.

View the output fields in `result_*.pvd` using [Paraview](https://www.paraview.org/).  These files contain the obstacle `psi`, the solution `u`, and the gap `u - psi`. The `result_sphere.pvd` file also contains the numerical error `|u - uexact|`.

## Generating meshes

Meshes can be created using the [Firedrake utility mesh generators](https://www.firedrakeproject.org/_modules/firedrake/utility_meshes.html).  Alternatively, one can create Netgen meshes with e.g. `SplineGeometry().GenerateMesh()`.  The resulting meshes have different refinement capabilities.

## Known limitations

See the list of known limitations in the [doc string for the VIAMR class](viamr/viamr.py).
Future bug fixes and feature improvements in Netgen, ngsPETSc, and PETSc DMPlex might change this situation, but for now see the known limitations below.

## Clearing caches

Firedrake will cache compiled weak forms.  At times, e.g. for addressing quadrature degree issues, and related irritating warnings, it is desirable to clear such caches:
```
python3 -c "import firedrake.tsfc_interface; firedrake.tsfc_interface.clear_cache()"
```

## Testing

Software tests use [pytest](https://docs.pytest.org/en/stable/index.html).  In the main directory `VI-AMR/` do
```
pytest .
```

The tests themselves are in `tests/`.

Tests marked `@pytest.mark.parallel(nprocs=N)` are found and run automatically under `mpiexec -n N` as part of this one command.  Note that VIAMR uses its own small, dependency-free harness instead, in `tests/conftest.py`, so you can just run `pytest .`  (Parallel testing with previously used a plugin with a [known bug](https://github.com/firedrakeproject/mpi-pytest/issues/13).)

For an HTML coverage report from these tests do:
```
pip install pytest-cov
pytest --cov-report html --cov=viamr tests/
firefox htmlcov/index.html
```
