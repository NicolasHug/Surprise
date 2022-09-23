import numpy as np
from .util import matrix_as_typed_memory_view

class PickableMixin:
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        # replace Typed MemoryView of a numpy array by array itself in order to enable pickling
        self_dict["sim"] = np.asarray(self.sim, dtype=np.double)
        return self_dict

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        # TODO: pass which attributes are memoryview to the constructor ?
        self.sim = matrix_as_typed_memory_view(state["sim"])