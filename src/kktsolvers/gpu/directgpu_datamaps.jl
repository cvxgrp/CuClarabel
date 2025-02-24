import cupy as cp
import numpy as np
from numba import cuda, njit

class SparseExpansionFullMap:
    pass

def pdim(maps):
    return sum(pdim(map) for map in maps)

def nnz_vec(maps):
    return sum(nnz_vec(map) for map in maps)

class SOCExpansionFullMap(SparseExpansionFullMap):
    def __init__(self, cone):
        self.u = cp.empty(cone.numel(), dtype=np.int32)
        self.v = cp.empty(cone.numel(), dtype=np.int32)
        self.ut = cp.empty(cone.numel(), dtype=np.int32)
        self.vt = cp.empty(cone.numel(), dtype=np.int32)
        self.D = cp.zeros(2, dtype=np.int32)

def pdim_SOCExpansionFullMap(map):
    return 2

def nnz_vec_SOCExpansionFullMap(map):
    return 4 * len(map.u)

def Dsigns_SOCExpansionFullMap(map):
    return (-1, 1)

def expansion_fullmap_SOCExpansionFullMap(cone):
    return SOCExpansionFullMap(cone)

@njit
def _csc_colcount_sparsecone_full_SOCExpansionFullMap(cone, map, K, row, col):
    nvars = cone.numel()
    _csc_colcount_colvec(K, nvars, row, col)
    _csc_colcount_colvec(K, nvars, row, col + 1)
    _csc_colcount_rowvec(K, nvars, col, row)
    _csc_colcount_rowvec(K, nvars, col + 1, row)
    _csc_colcount_diag(K, col, pdim_SOCExpansionFullMap(map))

@njit
def _csc_fill_sparsecone_full_SOCExpansionFullMap(cone, map, K, row, col):
    _csc_fill_colvec(K, map.v, row, col)
    _csc_fill_colvec(K, map.u, row, col + 1)
    _csc_fill_rowvec(K, map.vt, col, row)
    _csc_fill_rowvec(K, map.ut, col + 1, row)
    _csc_fill_diag(K, map.D, col, pdim_SOCExpansionFullMap(map))

@njit
def _csc_update_sparsecone_full_SOCExpansionFullMap(cone, map, updateFcn, scaleFcn):
    η2 = cone.η**2
    updateFcn(map.u, cone.sparse_data.u)
    updateFcn(map.v, cone.sparse_data.v)
    updateFcn(map.ut, cone.sparse_data.u)
    updateFcn(map.vt, cone.sparse_data.v)
    scaleFcn(map.u, -η2)
    scaleFcn(map.v, -η2)
    scaleFcn(map.ut, -η2)
    scaleFcn(map.vt, -η2)
    updateFcn(map.D, [-η2, +η2])

class GenPowExpansionFullMap(SparseExpansionFullMap):
    def __init__(self, cone):
        self.p = cp.empty(cone.numel(), dtype=np.int32)
        self.q = cp.empty(cone.dim1(), dtype=np.int32)
        self.r = cp.empty(cone.dim2(), dtype=np.int32)
        self.pt = cp.empty(cone.numel(), dtype=np.int32)
        self.qt = cp.empty(cone.dim1(), dtype=np.int32)
        self.rt = cp.empty(cone.dim2(), dtype=np.int32)
        self.D = cp.zeros(3, dtype=np.int32)

def pdim_GenPowExpansionFullMap(map):
    return 3

def nnz_vec_GenPowExpansionFullMap(map):
    return (len(map.p) + len(map.q) + len(map.r)) * 2

def Dsigns_GenPowExpansionFullMap(map):
    return (-1, -1, +1)

def expansion_fullmap_GenPowExpansionFullMap(cone):
    return GenPowExpansionFullMap(cone)

@njit
def _csc_colcount_sparsecone_full_GenPowExpansionFullMap(cone, map, K, row, col):
    nvars = cone.numel()
    dim1 = cone.dim1()
    dim2 = cone.dim2()
    _csc_colcount_colvec(K, dim1, row, col)
    _csc_colcount_colvec(K, dim2, row + dim1, col + 1)
    _csc_colcount_colvec(K, nvars, row, col + 2)
    _csc_colcount_rowvec(K, dim1, col, row)
    _csc_colcount_rowvec(K, dim2, col + 1, row + dim1)
    _csc_colcount_rowvec(K, nvars, col + 2, row)
    _csc_colcount_diag(K, col, pdim_GenPowExpansionFullMap(map))

@njit
def _csc_fill_sparsecone_full_GenPowExpansionFullMap(cone, map, K, row, col):
    dim1 = cone.dim1()
    _csc_fill_colvec(K, map.q, row, col)
    _csc_fill_colvec(K, map.r, row + dim1, col + 1)
    _csc_fill_colvec(K, map.p, row, col + 2)
    _csc_fill_rowvec(K, map.qt, col, row)
    _csc_fill_rowvec(K, map.rt, col + 1, row + dim1)
    _csc_fill_rowvec(K, map.pt, col + 2, row)
    _csc_fill_diag(K, map.D, col, pdim_GenPowExpansionFullMap(map))

@njit
def _csc_update_sparsecone_full_GenPowExpansionFullMap(cone, map, updateFcn, scaleFcn):
    data = cone.data
    sqrtμ = np.sqrt(data.μ)
    updateFcn(map.q, data.q)
    updateFcn(map.r, data.r)
    updateFcn(map.p, data.p)
    updateFcn(map.qt, data.q)
    updateFcn(map.rt, data.r)
    updateFcn(map.pt, data.p)
    scaleFcn(map.q, -sqrtμ)
    scaleFcn(map.r, -sqrtμ)
    scaleFcn(map.p, -sqrtμ)
    scaleFcn(map.qt, -sqrtμ)
    scaleFcn(map.rt, -sqrtμ)
    scaleFcn(map.pt, -sqrtμ)
    updateFcn(map.D, [-1, -1, 1])

class FullDataMap:
    def __init__(self, Pmat, Amat, cones):
        m, n = Amat.shape
        self.P = cp.zeros(Pmat.nnz, dtype=np.int32)
        self.A = cp.zeros(Amat.nnz, dtype=np.int32)
        self.At = cp.zeros(Amat.nnz, dtype=np.int32)
        self.diagP = cp.zeros(n, dtype=np.int32)
        self.Hsblocks = _allocate_kkt_Hsblocks(np.int32, cones)
        nsparse = sum(1 for cone in cones if cone.is_sparse_expandable())
        self.sparse_maps = []
        for cone in cones:
            if cone.is_sparse_expandable():
                self.sparse_maps.append(expansion_fullmap_GenPowExpansionFullMap(cone))
        self.diag_full = cp.zeros(m + n + pdim(self.sparse_maps), dtype=np.int32)

class GPUDataMap:
    def __init__(self, Pmat, Amat, cones, mapcpu):
        m, n = Amat.shape
        self.P = cp.zeros(0, dtype=np.int32) if mapcpu.P.size == 0 else cp.asarray(mapcpu.P)
        self.A = cp.zeros(0, dtype=np.int32) if mapcpu.A.size == 0 else cp.asarray(mapcpu.A)
        self.At = cp.zeros(0, dtype=np.int32) if mapcpu.At.size == 0 else cp.asarray(mapcpu.At)
        self.diagP = cp.asarray(mapcpu.diagP)
        self.Hsblocks = cp.asarray(mapcpu.Hsblocks)
        self.diag_full = cp.asarray(mapcpu.diag_full)
