import numpy as np
import firedrake as fd
from viamr import VIAMR
from functools import cached_property
from firedrake.dmhooks import get_appctx


class DataNetCDF:
    def __init__(self, filename, vname, xname="x1", yname="y1"):
        """constructor opens NetCDF4 file (filename) and reads variable (varname)
        into 2d numpy array; this defines a rectangular structured grid"""
        self.filename = filename
        import netCDF4

        data = netCDF4.Dataset(filename)
        data.set_auto_mask(False)  # otherwise irritating masked arrays
        self.vname = vname
        self.v = data.variables[vname][0, :, :].T  # transpose immediately
        self.x = data.variables[xname]
        self.y = data.variables[yname]
        self.mx, self.my = np.shape(self.v)
        assert self.mx == len(self.x)
        assert self.my == len(self.y)
        self.ll = (min(self.x), min(self.y))  # lower left
        self.ur = (max(self.x), max(self.y))  # upper right
        self.Wx, self.Wy = (
            self.ur[0] - self.ll[0],
            self.ur[1] - self.ll[1],
        )  # width, height
        self.hx, self.hy = self.x[1] - self.x[0], self.y[1] - self.y[0]

    def describe_grid(self, print=print, indent=4):
        indentstr = indent * " "
        llstr = f"({self.ll[0]/1000.0:.3f},{self.ll[1]/1000.0:.3f})"
        urstr = f"({self.ur[0]/1000.0:.3f},{self.ur[1]/1000.0:.3f})"
        print(f"{indentstr}rectangle from {self.filename}: {llstr}-->{urstr} km")
        print(
            f"{indentstr}  {self.mx} x {self.my} grid with {self.hx/1000.0:.3f} x {self.hy/1000.0:.3f} km spacing"
        )

    def preview(self):
        import matplotlib.pyplot as plt

        plt.pcolormesh(self.x, self.y, self.v.T, shading="nearest")
        plt.axis("equal")
        plt.title(f"{self.vname} (CLOSE FIGURE TO CONTINUE)")
        plt.show()

    def rectmesh(self, m):
        """generate a Firedrake rectangular mesh matching data mesh domain
        but with m elements in the x dimension."""
        mx = m
        my = int((self.Wy / self.Wx) * m)
        mesh = fd.RectangleMesh(
            mx,
            my,
            self.ur[0],
            self.ur[1],
            originX=self.ll[0],
            originY=self.ll[1],
            diagonal="crossed",
            distribution_parameters=VIAMR.PARALLEL_OVERLAP,
        )
        return mesh, mx, my

    def function(self, delnear=100.0e3, degree=1):
        """return a Firedrake CG_degree function on a rectangular
        Firedrake data mesh matching the vertices read from NetCDF file

        FIXME not parallel.  just add halos?"""
        dmesh = fd.RectangleMesh(
            self.mx - 1,
            self.my - 1,
            self.ur[0],
            self.ur[1],
            originX=self.ll[0],
            originY=self.ll[1],
            distribution_parameters=VIAMR.PARALLEL_OVERLAP,
        )
        dCG1 = fd.FunctionSpace(dmesh, "CG", 1)
        fCG1 = fd.Function(dCG1)
        for k in range(len(fCG1.dat.data)):
            xk, yk = dmesh.coordinates.dat.data[k]
            i = int((xk - self.ll[0]) / self.hx)
            j = int((yk - self.ll[1]) / self.hy)
            fCG1.dat.data[k] = self.v[i][j]
        if degree == 1:
            return fCG1  # done already
        else:
            assert degree > 1
            V = fd.FunctionSpace(dmesh, "CG", degree)
            f = fd.Function(V).project(fCG1)  # push up the detail a bit
            return f


class ZeroBelowSeaLevel(fd.DirichletBC):

    def __init__(self, V, g, sealevel=0.0):
        self.sl = sealevel
        super().__init__(V, g, None)

    @cached_property
    def nodes(self):
        V = self.function_space()
        # get application ctx from where it was stored, namely the coordinates DM
        ctx = get_appctx(V.mesh().coordinates.function_space().dm)
        assert ctx is not None, f"got None for appctx from {V.mesh()} coordinates DM"
        assert "b" in ctx, f"key 'b' not in context dictionary returned by DM"
        # return nodes with bed elevation less than sea level
        b = fd.Function(V).interpolate(ctx["b"])
        return np.where(b.dat.data_ro_with_halos < self.sl)[0]
