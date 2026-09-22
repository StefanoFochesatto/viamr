import numpy as np
from firedrake import *
from firedrake.petsc import PETSc

haveanimate = True
try:
    import animate
except ImportError:
    haveanimate = False


class AMAMixin:
    r"""Mixed into VIAMR (see viamr.py): construct metrics for metric-based
    mesh adaptation via the Animate library.  Note AMA = "averaged-metric adaptation".
    This approach is an alternative to the marking + skeleton-based-refinement
    methods that are the rest of the class.

    AMAMixin is not usable separately from VIAMR.  Specifically,
    _isotropicfbmetric() calls VIAMR._vcdsmooth().  Also, self.metricparameters
    and self.debug are set by VIAMR.__init__().

    TODO: Convert buildaveragedmetric() and _isotropicfbmetric() to the
    bounds=(lb,ub) signature used by the marking methods in VIAMR, namely
    fbmark(), inactivemark(), and nsvmark(), and by the set-indicator methods
    nodalactive(), elemactive(), and eleminactive().  These two AMAMixin methods
    are the last ones taking the older unilateral pair of a bound plus
    boxside="lower"/"upper"; the only other holdout in the library is
    VIAMR.freeboundarygraph2D().  The conversion is not merely mechanical: with
    box bounds there are two free boundaries, so the isotropic metric would be
    built from two smoothed indicators, one per bound.  The options are to sum
    the two gradient magnitudes before interpolating the metric, or to build two
    isotropic metrics and combine them by animate intersection, as
    buildaveragedmetric(intersect=True) already does for the isotropic and
    Hessian metrics.  Which is right should be settled by a bilateral example,
    e.g. examples/pollutant.py or examples/bilateral.py, and not by analogy
    alone.

    Public API: setmetricparameters(), buildaveragedmetric().

    Adapting the mesh with the built metric is the caller's job.  Here is how
    to do that:
        metric = amr.buildaveragedmetric(mesh, uh, lb)
        newmesh = animate.adapt(mesh, metric)

    Keeping the two steps separate lets a caller time or otherwise inspect
    VIAMR's metric-construction work independently of the Mmg/ParMmg (see
    https://www.mmgtools.org/) remesh that animate.adapt() performs.

    Note that animate.adapt() is a thin wrapper on PETSc's DMAdaptMetric().
    That method defaults to Mmg in serial and ParMmg in parallel, unless
    -dm_adaptor is set."""

    def setmetricparameters(self, **kwargs):
        """Set self.metricparameters, used by buildaveragedmetric(), from keyword
        arguments target_complexity (target number of mesh nodes; default
        3000.0), h_min (minimum allowed edge length; default 0.0), and h_max
        (maximum allowed edge length; default no limit).  Must be called
        before buildaveragedmetric()."""
        tc = kwargs.pop("target_complexity", 3000.0)
        hmin = kwargs.pop("h_min", 0.0)
        hmax = kwargs.pop("h_max", PETSc.INFINITY)
        if self.debug:
            assert np.isreal(tc) and tc > 0
            assert np.isreal(hmin) and hmin >= 0
            assert np.isreal(hmax) and hmax > 0
        mp = {
            "target_complexity": tc,  # target number of nodes
            "p": 2.0,  # normalisation order
            "h_min": hmin,  # minimum allowed edge length
            "h_max": hmax,  # maximum allowed edge length
        }
        self.metricparameters = {"dm_plex_metric": mp}
        return None

    def _isotropicfbmetric(self, mesh, uh, bound, CG1, P1tensor, boxside="lower"):
        """Construct a normalized free-boundary isotropic metric from abs(grad(s)),
        where s is the smoothed nodal active set indicator from _vcdsmooth(),
        i.e. the first stage of the VCD algorithm in fbmark().  Compare "L2"
        option in animate.compute_isotropic_metric(); here we already have a P1
        indicator.)"""
        assert haveanimate, "animate import failed, method unavailable"
        bounds = (None, bound) if boxside == "upper" else (bound, None)
        s = self._vcdsmooth(uh, bounds)
        maggrads = Function(CG1).interpolate(sqrt(dot(grad(s), grad(s))))
        VIMetric = animate.RiemannianMetric(P1tensor)
        VIMetric.set_parameters(self.metricparameters)
        VIMetric.interpolate(maggrads * Identity(mesh.topological_dimension))
        VIMetric.normalise()  # normalize *before* averaging
        return VIMetric

    def _hessianmetric(self, mesh, uh, P1tensor):
        """Construct a normalized metric from the Hessian of uh.  This is motivated
        by the interpolation error formula."""
        assert haveanimate, "animate import failed, method unavailable"
        hessmetric = animate.RiemannianMetric(P1tensor)
        hessmetric.set_parameters(self.metricparameters)
        # re method: default "mixed_L2" is more expensive
        hessmetric.compute_hessian(uh, method="L2")
        hessmetric.normalise()  # normalize *before* averaging
        return hessmetric

    def buildaveragedmetric(self, mesh, uh, bound, boxside="lower", gamma=0.50, intersect=False):
        """From the solution uh, of a unilateral obstacle problem with the given
        bound (boxside="lower" or "upper"; see the TODO in the AMAMixin doc
        string above), constructs both
        an anisotropic Hessian-based metric and an isotropic metric computed from the
        magnitude of the gradient of the smoothed VCD indicator.  These metrics are averaged
        (linearly-combined) using gamma:
          M(x) = gamma (isotropic) + (1-gamma) (anisotropic)
        The result M(x) is an anisotropic metric which is free-boundary aware.

        If intersect=True then does Animate intersect, instead of gamma average.

        Returns the animate.RiemannianMetric itself; this method does not call
        animate.adapt().  That's the caller's job; see the AMAMixin doc string above."""

        assert haveanimate, "animate import failed, method unavailable"
        assert (
            self.metricparameters is not None
        ), "call setmetricparameters() before calling buildaveragedmetric()"

        # Get the function spaces
        CG1, _ = self.spaces(mesh)
        P1tensor = TensorFunctionSpace(mesh, "CG", 1)

        # Branch on gamma to avoid unecesarry computation of both metrics
        if gamma == 1:  # isotropic metric only case
            return self._isotropicfbmetric(mesh, uh, bound, CG1, P1tensor, boxside=boxside)

        elif gamma == 0:  # hessian metric only case
            return self._hessianmetric(mesh, uh, P1tensor)

        else:
            # Default case where both metrics are computed
            VIMetric = self._isotropicfbmetric(mesh, uh, bound, CG1, P1tensor, boxside=boxside)
            hessmetric = self._hessianmetric(mesh, uh, P1tensor)
            # average or intersect
            if intersect:
                VIMetric.intersect(hessmetric)
            else:
                VIMetric.average(hessmetric, weights=[gamma, 1.0 - gamma])
            return VIMetric
