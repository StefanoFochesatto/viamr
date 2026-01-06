# compute element-wise maximum of absolute value of a CG1 function
#   into a DG0 function; this illustrates use of par_loop() from pyop2

from firedrake import *  # op2 is in namespace


def maxabselem(source):
    V = source.function_space()  # should work for any nodal basis space
    DG0 = FunctionSpace(V.mesh(), "DG", 0)
    target = Function(DG0)
    target.assign(0.0)
    dofs = V.finat_element.space_dimension()
    kernel = op2.Kernel(
        """
    void max_abs_to_dg0(double *target, double const *source)
    {
      /* Figure out max over cells */
      double tmp = 0.0;
      for (int i = 0; i < %(source_ndofs)s; i++) {
        tmp = tmp > fabs(source[i]) ? tmp : fabs(source[i]);
      }

      /* Push that value to DG0 dof */
      target[0] = tmp;
    }"""
        % {
            "source_ndofs": dofs,
        },
        "max_abs_to_dg0",
    )
    op2.par_loop(
        kernel,
        V.mesh().cell_set,
        target.dat(op2.MAX, target.cell_node_map()),
        source.dat(op2.READ, source.cell_node_map()),
    )
    return target


mesh = UnitSquareMesh(2, 1)
x, y = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "CG", degree=1)
f = Function(V, name="f").interpolate(2 * x - y)

maxaf = maxabselem(f)
maxaf.rename("max(|f|)")
VTKFile("result_maxfunction.pvd").write(f, maxaf)

print(maxaf.dat.data_ro)  # = 2, 2, 1, 1 on the four elements
