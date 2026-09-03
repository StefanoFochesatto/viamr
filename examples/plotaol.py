from aol import Function, amr, mesh, mark, refinedmesh   # run it

import numpy as np
import matplotlib.pyplot as plt
from firedrake.pyplot import tricontourf, tripcolor, triplot

_, DG0 = amr.spaces(mesh)

fig, axes = plt.subplots()

triplot(mesh, axes=axes, boundary_kw={"colors": "k"})
axes.set_aspect("equal")
axes.set_axis_off()
fig.savefig("aol-mesh.png", bbox_inches='tight')
plt.cla()

tripcolor(Function(DG0).interpolate(0.7 * mark), axes=axes, cmap='Greys', clim=(0.0,1.0))
triplot(mesh, axes=axes, boundary_kw={"colors": "k"})
yy = np.linspace(0.0, 1.0, 101)
xx = np.sqrt(2.0 - yy ** 2) - 1.0
axes.plot(xx, yy, 'r')
axes.set_aspect("equal")
axes.set_axis_off()
fig.savefig("aol-marked.png", bbox_inches='tight')
plt.cla()

triplot(refinedmesh, axes=axes, boundary_kw={"colors": "k"})
axes.plot(xx, yy, 'r')
axes.set_aspect("equal")
axes.set_axis_off()
fig.savefig("aol-refinedmesh.png", bbox_inches='tight')
