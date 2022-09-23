cimport numpy as np  # noqa
import numpy as np
np.import_array()

def matrix_as_typed_memory_view(np.ndarray matrix):
    cdef double[:,::1] M = matrix
    return M
