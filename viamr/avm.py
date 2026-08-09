import numpy as np
from firedrake import *
from firedrake.petsc import PETSc

haveanimate = True
try:
    import animate
except ImportError:
    haveanimate = False


class AVMMixin:
    r"""Mixed into VIAMR (see viamr.py): metric-based mesh adaptation via the
    Animate library (AVM = "averaged metric"), as an alternative to the
    marking + skeleton-based-refinement methods that are the rest of the
    class.  Not usable on its own -- _isotropicfbmetric() calls VIAMR's own
    vcdmark(), and self.metricparameters/self.debug are set by
    VIAMR.__init__().

    Public API: setmetricparameters(), adaptaveragedmetric()."""

    def setmetricparameters(self, **kwargs):
        """Set self.metricparameters, used by adaptaveragedmetric(), from keyword
        arguments target_complexity (target number of mesh nodes; default 3000.0),
        h_min (minimum allowed edge length; default 0.0), and h_max (maximum
        allowed edge length; default no limit).  Must be called before
        adaptaveragedmetric()."""
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

    def _isotropicfbmetric(self, mesh, uh, lb, CG1, P1tensor):
        """Construct a normalized free-boundary isotropic metric from abs(grad(s)),
        where s is the (smooth) output of vcdmark().  Compare "L2" option in
        animate.compute_isotropic_metric(); here we already have a P1 indicator.)"""
        assert haveanimate, "animate import failed, method unavailable"
        s = self.vcdmark(uh, lb, returnSmooth=True)
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

    def adaptaveragedmetric(
        self, mesh, uh, lb, gamma=0.50, intersect=False, metric=False
    ):
        """From the solution uh, of an obstacle problem with obstacle lb, constructs both
        an anisotropic Hessian-based metric and an isotropic metric computed from the
        magnitude of the gradient of the smoothed VCD indicator.  These metrics are averaged
        (linearly-combined) using gamma:
          M(x) = gamma (isotropic) + (1-gamma) (anisotropic)
        The result M(x) is an anisotropic metric which is free-boundary aware.  The mesh
        is adapted, according to the metric parameters, by calling the Animate
        mesh-adaptation library, which applies the Pragmatic mesher by default.
        If intersect=True then does Animate intersect (instead of gamma average).
        If metric=True then returns the metric itself, not the mesh."""

        assert haveanimate, "animate import failed, method unavailable"
        assert (
            self.metricparameters is not None
        ), "call setmetricparameters() before calling adaptaveragedmetric()"

        # Get the function spaces
        CG1, _ = self.spaces(mesh)
        P1tensor = TensorFunctionSpace(mesh, "CG", 1)

        # Branch on gamma to avoid unecesarry computation of both metrics
        if gamma == 1:  # isotropic metric only case
            VIMetric = self._isotropicfbmetric(mesh, uh, lb, CG1, P1tensor)
            if metric:
                return VIMetric
            return animate.adapt(mesh, VIMetric)

        elif gamma == 0:  # hessian metric only case
            hessmetric = self._hessianmetric(mesh, uh, P1tensor)
            if metric:
                return hessmetric
            return animate.adapt(mesh, hessmetric)

        else:
            # Default case where both metrics are computed
            VIMetric = self._isotropicfbmetric(mesh, uh, lb, CG1, P1tensor)
            hessmetric = self._hessianmetric(mesh, uh, P1tensor)
            # average or intersect
            if intersect:
                VIMetric.intersect(hessmetric)
            else:
                VIMetric.average(hessmetric, weights=[gamma, 1.0 - gamma])
            if metric:
                return VIMetric
            return animate.adapt(mesh, VIMetric)
