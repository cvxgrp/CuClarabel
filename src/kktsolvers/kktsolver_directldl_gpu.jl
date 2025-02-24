import cupy as cp
import numpy as np
from numba import cuda, njit
from scipy.sparse import csc_matrix

class GPULDLKKTSolver:
    def __init__(self, P, A, cones, m, n, settings):
        self.m = m
        self.n = n

        self.x = cp.zeros(m + n, dtype=np.float64)
        self.b = cp.zeros(m + n, dtype=np.float64)
        self.work1 = cp.zeros(m + n, dtype=np.float64)
        self.work2 = cp.zeros(m + n, dtype=np.float64)

        self.mapcpu = FullDataMap(P, A, cones)
        self.mapgpu = GPUDataMap(P, A, cones, self.mapcpu)

        self.Dsigns = cp.zeros(m + n, dtype=np.int32)
        self._fill_Dsigns(self.Dsigns, m, n, self.mapcpu)

        self.Hsblocks = cp.zeros_like(self._allocate_kkt_Hsblocks(np.float64, cones))

        self.KKTcpu, self.mapcpu = self._assemble_full_kkt_matrix(P, A, cones, 'full')
        self.KKTgpu = cp.sparse.csr_matrix(self.KKTcpu)

        self.settings = settings
        self.GPUsolver = CUDSSDirectLDLSolver(self.KKTgpu, self.x, self.b)

        self.diagonal_regularizer = 0.0

    def _fill_Dsigns(self, Dsigns, m, n, mapcpu):
        Dsigns[:n] = 1
        Dsigns[n:n + m] = -1
        for i, cone in enumerate(mapcpu.sparse_maps):
            Dsigns[n + m + i] = -1 if cone.Dsigns[0] == -1 else 1

    def _allocate_kkt_Hsblocks(self, dtype, cones):
        return np.zeros(sum(cone.numel() for cone in cones), dtype=dtype)

    def _assemble_full_kkt_matrix(self, P, A, cones, shape):
        map = FullDataMap(P, A, cones)
        m, n = A.shape
        p = pdim(map.sparse_maps)

        nnz_diagP = self._count_diagonal_entries_full(P)
        nnz_Hsblocks = len(map.Hsblocks)

        nnzKKT = (P.nnz + n - nnz_diagP + 2 * A.nnz + nnz_Hsblocks + 2 * nnz_vec(map.sparse_maps) + p)

        K = csc_matrix((m + n + p, m + n + p), dtype=P.dtype)

        self._full_kkt_assemble_colcounts(K, P, A, cones, map)
        self._full_kkt_assemble_fill(K, P, A, cones, map)

        return K, map

    def _full_kkt_assemble_colcounts(self, K, P, A, cones, map):
        m, n = A.shape

        K.indptr[:] = 0

        self._csc_colcount_block_full(K, P, A, 1)
        self._csc_colcount_missing_diag_full(K, P, 1)
        self._csc_colcount_block(K, A, n + 1, 'T')

        pcol = m + n + 1
        sparse_map_iter = iter(map.sparse_maps)

        for i, cone in enumerate(cones):
            row = cones.rng_cones[i][0] + n

            blockdim = cone.numel()
            if self.Hs_is_diagonal(cone):
                self._csc_colcount_diag(K, row, blockdim)
            else:
                self._csc_colcount_dense_full(K, row, blockdim)

            if self.is_sparse_expandable(cone):
                thismap = next(sparse_map_iter)
                self._csc_colcount_sparsecone_full(cone, thismap, K, row, pcol)
                pcol += pdim(thismap)

    def _full_kkt_assemble_fill(self, K, P, A, cones, map):
        m, n = A.shape

        self._csc_colcount_to_colptr(K)

        self._csc_fill_P_block_with_missing_diag_full(K, P, map.P)
        self._csc_fill_block(K, A, map.A, n + 1, 1, 'N')
        self._csc_fill_block(K, A, map.At, 1, n + 1, 'T')

        pcol = m + n + 1
        sparse_map_iter = iter(map.sparse_maps)

        for i, cone in enumerate(cones):
            row = cones.rng_cones[i][0] + n

            blockdim = cone.numel()
            block = map.Hsblocks[cones.rng_blocks[i]]

            if self.Hs_is_diagonal(cone):
                self._csc_fill_diag(K, block, row, blockdim)
            else:
                self._csc_fill_dense_full(K, block, row, blockdim)

            if self.is_sparse_expandable(cone):
                thismap = next(sparse_map_iter)
                self._csc_fill_sparsecone_full(cone, thismap, K, row, pcol)
                pcol += pdim(thismap)

        self._kkt_backshift_colptrs(K)

        self._map_diag_full(K, map.diag_full)
        map.diagP[:] = map.diag_full[:n]

    def _update_values(self, GPUsolver, KKT, index, values):
        KKT.data[index] = values

    def _update_diag_values_KKT(self, KKT, index, values):
        KKT.data[index] = values

    def kktsolver_update(self, cones):
        GPUsolver = self.GPUsolver
        return self._kktsolver_update_inner(GPUsolver, cones)

    def _kktsolver_update_inner(self, GPUsolver, cones):
        map = self.mapgpu
        KKT = self.KKTgpu

        self.get_Hs(cones, self.Hsblocks)
        self.Hsblocks *= -1.0
        self._update_values(GPUsolver, KKT, map.Hsblocks, self.Hsblocks)

        return self._kktsolver_regularize_and_refactor(GPUsolver)

    def _kktsolver_regularize_and_refactor(self, GPUsolver):
        settings = self.settings
        map = self.mapgpu
        KKTgpu = self.KKTgpu
        Dsigns = self.Dsigns
        diag_kkt = self.work1
        diag_shifted = self.work2

        if settings.static_regularization_enable:
            diag_kkt[:] = KKTgpu.data[map.diag_full]
            epsilon = self._compute_regularizer(diag_kkt, settings)

            diag_shifted[:] = diag_kkt
            diag_shifted += Dsigns * epsilon

            self._update_diag_values_KKT(KKTgpu, map.diag_full, diag_shifted)
            self.diagonal_regularizer = epsilon

        is_success = self.refactor(GPUsolver)

        if settings.static_regularization_enable:
            self._update_diag_values_KKT(KKTgpu, map.diag_full, diag_kkt)

        return is_success

    def _compute_regularizer(self, diag_kkt, settings):
        maxdiag = np.linalg.norm(diag_kkt, np.inf)
        regularizer = settings.static_regularization_constant + settings.static_regularization_proportional * maxdiag
        return regularizer

    def kktsolver_setrhs(self, rhsx, rhsz):
        b = self.b
        m, n = self.m, self.n

        b[:n] = rhsx
        b[n:n + m] = rhsz

    def kktsolver_getlhs(self, lhsx, lhsz):
        x = self.x
        m, n = self.m, self.n

        if lhsx is not None:
            lhsx[:] = x[:n]
        if lhsz is not None:
            lhsz[:] = x[n:n + m]

    def kktsolver_solve(self, lhsx, lhsz):
        x, b = self.x, self.b

        self.solve(self.GPUsolver, x, b)

        is_success = False
        if self.settings.iterative_refinement_enable:
            is_success = self._iterative_refinement(self.GPUsolver)
        else:
            is_success = np.all(np.isfinite(x))

        if is_success:
            self.kktsolver_getlhs(lhsx, lhsz)

        return is_success

    def _iterative_refinement(self, GPUsolver):
        x, b = self.x, self.b
        e, dx = self.work1, self.work2
        settings = self.settings

        IR_reltol = settings.iterative_refinement_reltol
        IR_abstol = settings.iterative_refinement_abstol
        IR_maxiter = settings.iterative_refinement_max_iter
        IR_stopratio = settings.iterative_refinement_stop_ratio

        KKT = self.KKTgpu
        normb = np.linalg.norm(b, np.inf)

        norme = self._get_refine_error(e, b, KKT, x)
        if not np.isfinite(norme):
            return False

        for i in range(IR_maxiter):
            if norme <= IR_abstol + IR_reltol * normb:
                break
            lastnorme = norme

            self.solve(GPUsolver, dx, e)

            dx += x
            norme = self._get_refine_error(e, b, KKT, dx)
            if not np.isfinite(norme):
                return False

            improved_ratio = lastnorme / norme
            if improved_ratio < IR_stopratio:
                if improved_ratio > 1.0:
                    x, dx = dx, x
                break
            x, dx = dx, x

        self.x, self.work2 = x, dx

        return True

    def _get_refine_error(self, e, b, KKT, xi):
        e[:] = b - KKT @ xi
        norme = np.linalg.norm(e, np.inf)
        return norme

    def solve(self, GPUsolver, x, b):
        solve(GPUsolver, x, b)

    def refactor(self, GPUsolver):
        return refactor(GPUsolver)

    def get_Hs(self, cones, Hsblocks):
        get_Hs(cones, Hsblocks)
