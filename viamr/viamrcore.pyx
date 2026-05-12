cimport petsc4py.PETSc as PETSc

# --- PETSc C API Declarations ---

cdef extern from "petsc.h" nogil:
    ctypedef int PetscInt
    ctypedef int PetscErrorCode
    ctypedef int PetscBool
    ctypedef struct _p_IS
    ctypedef _p_IS* PetscIS "IS"
    ctypedef struct _p_PetscSF
    ctypedef _p_PetscSF* PetscSF "PetscSF"
    PetscBool PETSC_TRUE
    PetscBool PETSC_FALSE

cdef extern from "petscdmplex.h" nogil:
    PetscErrorCode DMPlexLabelComplete(PETSc.PetscDM, PETSc.PetscDMLabel)
    PetscErrorCode DMPlexLabelCompleteStar(PETSc.PetscDM, PETSc.PetscDMLabel)

cdef extern from "petscdmlabel.h" nogil:
    PetscErrorCode DMLabelPropagateBegin(PETSc.PetscDMLabel, PetscSF)
    PetscErrorCode DMLabelPropagateEnd(PETSc.PetscDMLabel, PetscSF)

cdef extern from "petscdm.h" nogil:
    PetscErrorCode DMGetPointSF(PETSc.PetscDM, PetscSF*)

# --- Implementation ---

def udo_mark_cells(dm_obj, label_obj, int n_layers):
    """
    Expands a DMLabel through n_layers of topological neighborhoods.
    Uses the idiomatic PETSc DMPlexLabelComplete/Star routines.
    """
    cdef PETSc.DM dm_py = dm_obj
    cdef PETSc.PetscDM dm = dm_py.dm
    cdef PETSc.DMLabel label_py = label_obj
    cdef PETSc.PetscDMLabel label = label_py.dmlabel
    cdef PetscSF sf = NULL
    
    # Get the Point Star Forest for parallel communication
    DMGetPointSF(dm, &sf)
    
    for l in range(n_layers):
        # 1. Downward pass: Mark the closure of currently marked points (Cells -> Vertices)
        DMPlexLabelComplete(dm, label)
        
        # 2. Parallel Sync: Share marked boundary points with neighbors
        if sf != NULL:
            DMLabelPropagateBegin(label, sf)
            DMLabelPropagateEnd(label, sf)
            
        # 3. Upward pass: Mark the star of currently marked points (Vertices -> Cells)
        DMPlexLabelCompleteStar(dm, label)
            
    return
