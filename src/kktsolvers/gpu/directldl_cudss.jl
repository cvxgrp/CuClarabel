import cupy as cp
import numpy as np
from numba import cuda, njit
from cudss import CudssSolver, cudss, cudss_set, ldiv

class CUDSSDirectLDLSolver:
    def __init__(self, KKT, x, b):
        self.KKTgpu = KKT
        self.cudssSolver = CudssSolver(KKT, "S", 'F')
        cudss("analysis", self.cudssSolver, x, b)
        cudss("factorization", self.cudssSolver, x, b)
        self.x = x
        self.b = b

def required_matrix_shape():
    return 'full'

def refactor(ldlsolver):
    cudss_set(ldlsolver.cudssSolver.matrix, ldlsolver.KKTgpu)
    cudss("factorization", ldlsolver.cudssSolver, ldlsolver.x, ldlsolver.b)
    return True

def solve(ldlsolver, x, b):
    ldiv(x, ldlsolver.cudssSolver, b)
