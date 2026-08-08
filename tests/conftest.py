"""Dependency-free harness for running MPI-parallel tests via plain `pytest`.

Tests marked

    @pytest.mark.parallel(nprocs=N)

are run under `mpiexec -n N python -m pytest <file>::<test> -q`, in their own
subprocess, *before* pytest imports the containing test module in this
process.  Importing firedrake (as every test file here does, at module level)
initializes MPI as a side effect, and forking a nested mpiexec from a process
that has already called MPI_Init is unreliable.  So parallel tests are found
here by statically parsing test file source with `ast` -- no import, so MPI
is never touched in this process before the fork -- and run in a subprocess
ahead of collection.  Their node IDs are then dropped from the collected item
list so they are not also executed serially in-process afterwards.

Just run `pytest .` (or `pytest tests/`, or a specific file/test) from
anywhere in the repo; parallel tests are found and run automatically.

Scope: this only understands `@pytest.mark.parallel(nprocs=<int>)` with a
single integer nprocs (the only form currently used in this repo), and
`@pytest.mark.skip`/`skipif` (which suppress external execution, leaving the
test to show up as normally skipped).  It does not implement pytest-mpi's
parametrized-nprocs-list feature or -k/-m filtering of externally-run tests.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

_CHILD_FLAG = "_VIAMR_PARALLEL_CHILD"
"""Environment variable set on the subprocess spawned by mpiexec below, so
that its own pytest_configure (which also loads this conftest.py) does not
try to recursively fork another mpiexec for the same test."""

_results = {}  # nodeid -> (passed, nprocs, output)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "parallel(nprocs): run this test under mpiexec -n nprocs"
    )
    if os.environ.get(_CHILD_FLAG):
        return  # this process IS one of the mpiexec-spawned workers; just run normally
    for filepath, restrict in _target_files(config):
        for funcname, nprocs in _find_parallel_tests(filepath):
            if restrict is not None and funcname != restrict:
                continue
            nodeid = _nodeid(config, filepath, funcname)
            _results[nodeid] = _run_parallel_test(nodeid, filepath, funcname, nprocs)


def pytest_collection_modifyitems(config, items):
    if not _results:
        return
    keep = []
    for item in items:
        if item.nodeid in _results:
            passed, nprocs, _ = _results[item.nodeid]
            status = "PASSED" if passed else "FAILED"
            print(f"[parallel] {item.nodeid}: already run under mpiexec -n {nprocs}: {status}")
        else:
            keep.append(item)
    items[:] = keep


def pytest_sessionfinish(session, exitstatus):
    if not _results:
        return
    failures = {n: r for n, r in _results.items() if not r[0]}
    if failures:
        for nodeid, (_, nprocs, output) in failures.items():
            print(f"\n===== FAILED (parallel, nprocs={nprocs}): {nodeid} =====")
            print(output)
        session.exitstatus = 1
    elif session.exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
        # every in-process item was one we already ran externally (and it
        # passed); don't report "no tests ran" for what was really a pass
        session.exitstatus = pytest.ExitCode.OK


def _nodeid(config, filepath, funcname):
    rel = filepath.resolve().relative_to(config.rootpath).as_posix()
    return f"{rel}::{funcname}"


def _target_files(config):
    """Yield (filepath, restrict_funcname_or_None) for every candidate test
    file implied by the pytest invocation's positional args."""
    args = [a for a in config.args if not str(a).startswith("-")]
    if not args:
        args = [str(config.rootpath)]
    patterns = list(config.getini("python_files") or ["test_*.py", "*_test.py"])
    for a in args:
        filepart, sep, restrict = str(a).partition("::")
        restrict = restrict.split("::")[0] if sep else None
        path = Path(filepart).resolve()
        if path.is_file():
            yield path, restrict
        elif path.is_dir():
            found = set()
            for pattern in patterns:
                found.update(path.rglob(pattern))
            for filepath in sorted(found):
                yield filepath, None


def _find_parallel_tests(filepath):
    """Statically find not-skipped top-level functions decorated with
    @pytest.mark.parallel(nprocs=N).  Does not import filepath."""
    try:
        tree = ast.parse(filepath.read_text(), filename=str(filepath))
    except (SyntaxError, OSError):
        return []
    out = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        nprocs, skipped = None, False
        for dec in node.decorator_list:
            name, kwargs = _mark_decorator(dec)
            if name == "parallel":
                nprocs = kwargs.get("nprocs")
            elif name in ("skip", "skipif"):
                skipped = True
        if nprocs is not None and not skipped:
            out.append((node.name, nprocs))
    return out


def _mark_decorator(dec):
    """Return (mark_name, kwargs) for a `@pytest.mark.X(...)` or
    `@pytest.mark.X` decorator AST node; (None, {}) if it doesn't match."""
    call = dec if isinstance(dec, ast.Call) else None
    attr = call.func if call else dec
    if not isinstance(attr, ast.Attribute):
        return None, {}
    if not (isinstance(attr.value, ast.Attribute) and attr.value.attr == "mark"):
        return None, {}
    kwargs = {}
    if call is not None:
        for kw in call.keywords:
            if isinstance(kw.value, ast.Constant):
                kwargs[kw.arg] = kw.value.value
        for arg in call.args:
            if isinstance(arg, ast.Constant) and "nprocs" not in kwargs:
                kwargs["nprocs"] = arg.value
    return attr.attr, kwargs


def _run_parallel_test(nodeid, filepath, funcname, nprocs):
    cmd = [
        "mpiexec", "--oversubscribe", "-n", str(nprocs),
        sys.executable, "-m", "pytest", "-q", "--no-header",
        f"{filepath}::{funcname}",
    ]
    env = dict(os.environ, **{_CHILD_FLAG: "1"})
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    passed = proc.returncode == 0
    print(f"[parallel] running {nodeid} under mpiexec -n {nprocs} ... "
          f"{'ok' if passed else 'FAILED'}")
    return passed, nprocs, proc.stdout + proc.stderr
