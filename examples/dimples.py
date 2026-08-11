# FIXME  create an example to demonstrate meaning of safe-active marking
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
#      2. UDO+(BR|BV00) without VIAMR.statedegeneratemark()
#      3. same          with VIAMR.statedegeneratemark()
#
# the main ideas:
#   * for the classical problem, you need to refine in the coarse active sets
#     to pick up the (potential or actual according to f vs lap(u)) small
#     inactive sets in the dimples
#   * for the porous-like problem, if you apply state-degeneracy marking
#     then, where f<0, you can avoid further refinement in active sets 