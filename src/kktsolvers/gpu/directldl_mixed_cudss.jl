import cupy as cp
import numpy as np
from numba import cuda, njit
from cudss import CudssSolver, cudss, cudss_set, ldiv

class CUDSSDirectLDLSolverMixed:
    def __init__(self, KKT, x, b):
        dim = KKT.shape[0]

        self.KKTgpu = KKT

        val = cp.array(KKT.data, dtype=np.float32)
        self.KKTFloat32 = cp.sparse.csr_matrix((val, KKT.indices, KKT.indptr), shape=KKT.shape)
        self.cudssSolver = CudssSolver(self.KKTFloat32, "S", 'F')

        self.xFloat32 = cp.zeros(dim, dtype=np.float32)
        self.bFloat32 = cp.zeros(dim, dtype=np.float32)

        cudss("analysis", self.cudssSolver, self.xFloat32, self.bFloat32)
        cudss("factorization", self.cudssSolver, self.xFloat32, self.bFloat32)

GPUSolversDict = {}
GPUSolversDict['cudssmixed'] = CUDSSDirectLDLSolverMixed

def required_matrix_shape():
    return 'full'

def refactor(ldlsolver):
    ldlsolver.KKTFloat32.data = ldlsolver.KKTgpu.data.astype(np.float32)

    cudss_set(ldlsolver.cudssSolver.matrix, ldlsolver.KKTFloat32)
    cudss("factorization", ldlsolver.cudssSolver, ldlsolver.xFloat32, ldlsolver.bFloat32)

    return True

def solve(ldlsolver, x, b):
    bFloat32 = ldlsolver.bFloat32
    bFloat32[:] = b

    ldiv(ldlsolver.xFloat32, ldlsolver.cudssSolver, bFloat32)

    x[:] = ldlsolver.xFloat32
