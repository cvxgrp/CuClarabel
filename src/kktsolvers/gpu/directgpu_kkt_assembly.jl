import cupy as cp
import numpy as np
from numba import cuda, njit
from scipy.sparse import csc_matrix

def _assemble_full_kkt_matrix(P, A, cones, shape='triu'):
    map = FullDataMap(P, A, cones)
    m, n = A.shape
    p = pdim(map.sparse_maps)

    nnz_diagP = _count_diagonal_entries_full(P)
    nnz_Hsblocks = len(map.Hsblocks)

    nnzKKT = (P.nnz + n - nnz_diagP + 2 * A.nnz + nnz_Hsblocks + 2 * nnz_vec(map.sparse_maps) + p)

    K = csc_matrix((m + n + p, m + n + p), dtype=P.dtype)

    _full_kkt_assemble_colcounts(K, P, A, cones, map)
    _full_kkt_assemble_fill(K, P, A, cones, map)

    return K, map

def _full_kkt_assemble_colcounts(K, P, A, cones, map):
    m, n = A.shape

    K.indptr[:] = 0

    _csc_colcount_block_full(K, P, A, 1)
    _csc_colcount_missing_diag_full(K, P, 1)
    _csc_colcount_block(K, A, n + 1, 'T')

    pcol = m + n + 1
    sparse_map_iter = iter(map.sparse_maps)

    for i, cone in enumerate(cones):
        row = cones.rng_cones[i][0] + n

        blockdim = cone.numel()
        if Hs_is_diagonal(cone):
            _csc_colcount_diag(K, row, blockdim)
        else:
            _csc_colcount_dense_full(K, row, blockdim)

        if is_sparse_expandable(cone):
            thismap = next(sparse_map_iter)
            _csc_colcount_sparsecone_full(cone, thismap, K, row, pcol)
            pcol += pdim(thismap)

    return

def _full_kkt_assemble_fill(K, P, A, cones, map):
    m, n = A.shape

    _csc_colcount_to_colptr(K)

    _csc_fill_P_block_with_missing_diag_full(K, P, map.P)
    _csc_fill_block(K, A, map.A, n + 1, 1, 'N')
    _csc_fill_block(K, A, map.At, 1, n + 1, 'T')

    pcol = m + n + 1
    sparse_map_iter = iter(map.sparse_maps)

    for i, cone in enumerate(cones):
        row = cones.rng_cones[i][0] + n

        blockdim = cone.numel()
        block = map.Hsblocks[cones.rng_blocks[i]]

        if Hs_is_diagonal(cone):
            _csc_fill_diag(K, block, row, blockdim)
        else:
            _csc_fill_dense_full(K, block, row, blockdim)

        if is_sparse_expandable(cone):
            thismap = next(sparse_map_iter)
            _csc_fill_sparsecone_full(cone, thismap, K, row, pcol)
            pcol += pdim(thismap)

    _kkt_backshift_colptrs(K)

    _map_diag_full(K, map.diag_full)
    map.diagP[:] = map.diag_full[:n]

    return
