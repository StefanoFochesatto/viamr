# from NSV03:
#
#   Nochetto, R. H., Siebert, K. G., & Veeser, A. (2003). Pointwise
#   a posteriori error control for elliptic obstacle problems.
#   Numerische Mathematik, 95(1), 163-195.
#
# does the following:
#   1. computes sigma_h from section 2.1 in NSV03
#   2. computes the "practical estimator" \eta_\infty and \eta_d in formula (7.1) of NSV03
#   3. solves "7.2 Example: Constant Obstacle"

from firedrake import *
from viamr import VIAMR
from firedrake.petsc import PETSc

# major parameters
d = 2  # spatial dimension
m = 3  # initial mesh resolution
levs = 4 if d == 2 else 3  # number of refinements
nUDO = 0  # observe that {sigma_h * u_h > 0} is same as UDO mark with nUDO=0
figure = False  # generate figure to compare to NSV03
primaltol = 0.0  # for admissibility: u_h >= -primaltol
dualtol = 1.0e-10  # used for admissibilty (sigma_h >= -dualtol) *and* when computing estimator


def maxabselem(source):
    """Compute element-wise maximum of absolute value of source, returning
    a DG0 field.  This should work in parallel for any nodal basis space,
    e.g. CG_k or DG_k for any k."""
    V = source.function_space()
    DG0 = FunctionSpace(V.mesh(), "DG", 0)
    target = Function(DG0, name="max |source| as DG0").assign(0.0)
    kernel = op2.Kernel(
        """
    void max_abs(double *target, double const *source)
    {
      /* Evaluate max over cell */
      double tmp = 0.0;
      for (int i = 0; i < %(ndofs)s; i++) {
        tmp = tmp > fabs(source[i]) ? tmp : fabs(source[i]);
      }

      /* As DG0 dof */
      target[0] = tmp;
    }"""
        % {
            "ndofs": V.finat_element.space_dimension(),
        },
        "max_abs",
    )
    op2.par_loop(
        kernel,
        V.mesh().cell_set,
        target.dat(op2.MAX, target.cell_node_map()),
        source.dat(op2.READ, source.cell_node_map()),
    )
    return target


def thinelemactive(u, psi, activetol=1.0e-10):
    """Compute element active set indicator into DG0, but "thinned" so that a
    cell is marked as active only if this cell *and its neighboring cells* are
    active, according to activetol.  Returns a DG0 element-wise indicator, with
    active elements having value 1.
      The implementation is inspired by VIAMR.udomark().  The active elements
    are captured first using maxabselem() above (not VIAMR.elemactive()).  Then
    the neighbor elements of *inactive* elements are found, and they are
    effectively removed from the active element indicator.
    """
    # set up
    W = u.function_space()
    assert W == psi.function_space()
    mesh = W.mesh()
    d = mesh.cell_dimension()
    dm = mesh.topology_dm
    plexelementlist = mesh.cell_closure[:, -1]
    # will need map from DMPlex to firedrake indices
    # (Is there a better way to do this in dmcommon?)
    dm2fd = np.argsort(plexelementlist)
    # element-wise maximum of gap=u-psi, into DG0
    gap = Function(W).interpolate(u - psi)
    assert min(gap.dat.data_ro) >= 0.0
    gapmax = maxabselem(gap)
    # get DMPlex element indices of inactive cells using dmplex cell indices
    inactivecells = [
        plexelementlist[k]
        for k, value in enumerate(gapmax.dat.data_ro_with_halos)
        if value >= activetol  # test *in*active
    ]
    # vertex closure: indices of vertices which are incident to an inactive
    #   element, then flatten and remove duplicates
    incvertices = [dm.getTransitiveClosure(k)[0][-d - 1 :] for k in inactivecells]
    incvertices = np.unique(np.ravel(incvertices))
    # star: indices of all elements which are incident to the incidentVertices
    #   note that getTransitiveClosure() with useCone=False gives the star
    #   note that the number of elements incident to a vertex is not predictable
    #   then flatten and remove duplicates
    kmin, kmax = dm.getHeightStratum(0)[:2]
    neighborindices = []
    for j in incvertices:
        star = dm.getTransitiveClosure(j, useCone=False)[0]
        mark = np.where((star >= kmin) & (star < kmax))
        neighborindices.extend(star[mark])
    neighborindices = np.unique(np.ravel(neighborindices))
    # generate DG0 thin element active indicator by zeroing-out neighbors
    # of inactive cells
    DG0 = FunctionSpace(mesh, "DG", 0)
    z = Function(DG0).interpolate(Constant(1.0))  # mark *all* cells 1.0
    for k in neighborindices:
        # parallel communication *here*:
        z.dat.data_wo_with_halos[dm2fd[k]] = 0.0  # remove inactive etc.
    return z


assert d in [2, 3]
if d == 2:
    mesh = RectangleMesh(m, m, 1.0, 1.0, originX=-1.0, originY=-1.0, diagonal="crossed")
else:
    # 3D SBR refinement needs Netgen mesh and Netgen refinement (and produces bad meshes)
    from netgen.occ import *

    box = Box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    mesh = Mesh(OCCGeometry(box, dim=3).GenerateMesh(maxh=0.8))

sp = {
    "snes_type": "vinewtonrsls",
    "snes_converged_reason": None,
    # "snes_monitor": None,
    # "snes_view": None,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

print = PETSc.Sys.Print  # enables correct printing in parallel
print(f"solving {d}D example from Nochetto, Siebert, & Veeser (2003) ...")
r = 0.7  # parameter in defining problem
dofs, errs = [], []
for j in range(levs):
    x = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "CG", 1)

    # UFL expressions for source function and boundary values
    if d == 2:
        x2 = x[0] ** 2 + x[1] ** 2
    else:
        x2 = x[0] ** 2 + x[1] ** 2 + x[2] ** 2
    circle = x2 - r ** 2
    f_ufl = conditional(
        x2 <= r ** 2, -8.0 * r ** 2 * (1.0 - circle), -4.0 * (2.0 * x2 + d * circle)
    )
    g_ufl = circle ** 2  # note this is quartic, so P4 interpolation should be exact

    # initialize by cross-mesh interpolation, i.e. do mesh sequencing
    uh = Function(V, name="u_h (solution)").interpolate(uh if j > 0 else 0.0)

    # state the problem
    vh = TestFunction(V)
    F = inner(grad(uh), grad(vh)) * dx - f_ufl * vh * dx
    g = Function(V).interpolate(g_ufl)  # = I_h g in NSV03
    bcs = DirichletBC(V, g, "on_boundary")
    problem = NonlinearVariationalProblem(F, uh, bcs)
    psih = Function(V).interpolate(0.0)
    INFupper = Function(V).interpolate(Constant(PETSc.INFINITY))

    # solve the problem
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=sp, options_prefix="s"
    )
    solver.solve(bounds=(psih, INFupper))
    # following admissibility check removes a term from the estimator
    assert min(uh.dat.data_ro) >= 0.0

    # error relative to exact (UFL) solution
    u_ufl = conditional(x2 <= r ** 2, 0.0, circle ** 2)
    dofs.append(V.dim())
    errs.append(float(errornorm(u_ufl, uh)))
    print(f"  level {j}: nodes = {dofs[-1]}, |u-u_h|_2 = {errs[-1]:.3e}")

    # compute UDO marking; note fmark is written to file for comparison
    amr = VIAMR()
    fmark = amr.udomark(uh, psih, n=nUDO)
    residual = -div(grad(uh))
    (imark, _, _) = amr.brinactivemark(uh, psih, residual, theta=0.5)
    mark = amr.unionmarks(fmark, imark)

    # get next mesh by refinement
    if j == levs - 1:
        break
    if d == 2:
        mesh = amr.refinemarkedelements(mesh, mark)  # PETSc DM refinement
    else:
        mesh = mesh.refine_marked_elements(mark)  # Netgen refinement

# optional convergence figure
if figure and mesh.comm.rank == 0:
    import matplotlib.pyplot as plt
    import numpy as np

    dofs, errs = np.array(dofs), np.array(errs)
    # print(np.polyfit(np.log(dofs), np.log(errs), 1))
    plt.loglog(dofs, errs, "ko", label=r"$\|u - u_h\|_0$")
    y = dofs ** (-2.0 / d)
    y = y * errs[0] / y[0]  # fix constant so that it aligns
    plt.loglog(dofs, y, "k:", label=f"DOFs^(-2/d) for d={d}")
    plt.legend()
    plt.grid(True)
    plt.xlabel("DOFs")
    plt.ylabel("error")
    plt.title("compare Figure 7.1 in Nochetto, Siebert, & Veeser (2003)")
    plt.show()

# the rest of these actions are just for the final mesh

# compute some quantities for output file
fmark.rename("UDO FB mark")
uerr = Function(V, name="u_err = u_h - u_exact").interpolate(uh - u_ufl)

# Following section 2.1 of NSV03, compute residual sigmah in V=P1,
#   but use opposite sign convention so sigmah >= 0.   Note that
#   complementarity would be
#     uh >= 0,  sigmah >= 0,  uh sigmah = 0
#   because psih=0.
# step 1: create cofunction with values int_Omega phi_i dx for *all* nodes i
phi = TestFunction(V)
scaleh = assemble(phi * dx)  # cofunction; we *do not* want riesz_representation() here
# step 2: compute unscaled sigma_h
res = assemble((inner(grad(uh), grad(phi)) - f_ufl * phi) * dx)  # cofunction
# step 3: divide numpy arrays to give correct scale
sigmah = Function(V, name="sigma_h (residual)")
sigmah.dat.data[:] = res.dat.data_ro / scaleh.dat.data_ro  # divide numpy arrays
# all boundary nodes are inactive *in this example*
#    section 2.1 of NSV03 addresses cases where boundary nodes are active
#    perhaps use:  n = FacetNormal(mesh); ?? inner(grad(uh), n) * omegah * ds
DirichletBC(V, Constant(0.0), "on_boundary").apply(sigmah)

# check dual admissiblity (up to tolerance)
assert min(sigmah.dat.data_ro) >= -dualtol

# Rinf is computed from (3.7) in NSV03 using p=\infty and p'=1:
#   R_\infty = h_T^{-1} \|[[\partial_n u_h]]\|* + X
# where by (2.3) in NSV03:
#    X = |f + sigma_h| if entire neighborhood of T is active (note sign switch on sigma_h)
#    X = |f|           otherwise
# and where
#    \|.\|* = \|.\|_{\infty; \partial T \setminus \partial \Omega}
# and where
#    [[z]] is the jump in z along an edge
n = FacetNormal(mesh)
DG0 = FunctionSpace(mesh, "DG", 0)
hT = project(CellSize(mesh), DG0)
v0 = TestFunction(DG0)
jumpu = assemble(jump(grad(uh) * v0, n) * dS).riesz_representation()  # in DG0
thinactive = thinelemactive(uh, psih)
X_ufl = thinactive * abs(f_ufl + sigmah) + (1 - thinactive) * abs(f_ufl)
Rinf = Function(DG0).interpolate((abs(jumpu) / hT) + X_ufl)

# compute local "practical estimator" from formula (7.1) in NSV03
# namely *for each closed triangle T*:
#   \eta_\infty =
#        C_0 h_T^2 \|R_\infty\|_\infty
#      + \|(\chi - u_h)^+\|_\infty                [= 0 since uh >= 0.0 = chi here]
#      + 1_{sigma_h > 0} * \|(u_h - \chi)^+\|_\infty   [require on T: sigma_h > dualtol]
#      + \|g - I_h g\|_{\infty;\partial\Omega \cap T}   [exact g is in CG4, I_h g is in CG1]
C0 = 0.1
gaph = Function(V).interpolate(uh - psih)  # = "(u_h - \chi)_+" since uh >= psih
sigmahT = Function(DG0).interpolate(sigmah)
# note that blockgap is nonzero in same cells as UDO n=0 fmark
blockgap_ufl = conditional(sigmahT > dualtol, maxabselem(gaph), 0.0)
blockgap = Function(DG0).interpolate(blockgap_ufl)
DG4 = FunctionSpace(mesh, "DG", 4)
adg = maxabselem(Function(DG4).interpolate(g_ufl - g))  # in DG0, over all of Omega
# bdryerr is a DG0 function, but only nonzero along boundary
bdryerr = assemble(adg * v0 * ds).riesz_representation()
etainf = Function(DG0, name="eta_{inf,T}")
etainf.interpolate(C0 * hT ** 2 * maxabselem(Rinf) + blockgap + bdryerr)

# FIXME also \eta_d

outfile = "result_nsv.pvd"
print(f"writing to {outfile} ...")
active = amr.elemactive(uh, psih)
active.rename("active")
thinactive.rename("thin active")
if mesh.comm.size > 1:
    rank = Function(FunctionSpace(mesh, "DG", 0))
    rank.dat.data[:] = mesh.comm.rank
    rank.rename("rank")
    VTKFile(outfile).write(uh, uerr, sigmah, etainf, fmark, active, thinactive, rank)
else:
    VTKFile(outfile).write(uh, uerr, sigmah, etainf, fmark, active, thinactive)
