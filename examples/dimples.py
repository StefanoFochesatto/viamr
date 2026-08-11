# FIXME  create an example to demonstrate safe-active un-marking
#
# start from the sphere problem and add gaussian dimples scattered on
# the upper hemisphere obstacle to make psi.  change the f to be negative
# everywhere, but not too negative (compared to laplacian of psi in the
# dimples)
#
# now do two problems and three AMR:
# problems:  1. classical obstacle problem -div(grad(u)) - f
#            2. porous-like operator -div((u-psi)^{gamma-1} grad(u)) - f
# amr: 1. NSV
#      2. UDO+(BR|BV00) without VIAMR.safeactiveunmark()
#      3. same          with VIAMR.safeactiveunmark()
#
# the main ideas:
#   * for the classical problem, you *need* to refine in the coarse active sets
#     to pick up the (potential or actual according to f vs lap(u)) small
#     inactive sets in the dimples
#   * for the porous-like problem, with a state-degenerate coefficient, it is safe
#     to skip marking and refinement in the active set as long as f<0
