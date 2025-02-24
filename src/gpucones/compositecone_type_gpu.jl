import cupy as cp
import numpy as np
from numba import cuda, njit

class CompositeConeGPU:
    def __init__(self, cpucones):
        self.cones = cpucones.cones
        self.type_counts = cpucones.type_counts
        self._is_symmetric = cpucones._is_symmetric

        self.n_zero = self.type_counts.get('ZeroCone', 0)
        self.n_nn = self.type_counts.get('NonnegativeCone', 0)
        self.n_linear = self.n_zero + self.n_nn
        self.n_soc = self.type_counts.get('SecondOrderCone', 0)
        self.n_exp = self.type_counts.get('ExponentialCone', 0)
        self.n_pow = self.type_counts.get('PowerCone', 0)
        self.n_psd = self.type_counts.get('PSDTriangleCone', 0)

        self.idx_eq = [i for i in range(self.n_linear) if isinstance(self.cones[i], ZeroCone)]
        self.idx_inq = [i for i in range(self.n_linear) if not isinstance(self.cones[i], ZeroCone)]

        self.numel = sum(cone.numel() for cone in self.cones)
        self.degree = sum(cone.degree() for cone in self.cones)

        self.numel_linear = sum(cone.numel() for cone in self.cones[:self.n_linear])
        self.max_linear = max(cone.numel() for cone in self.cones[:self.n_linear])
        self.numel_soc = sum(cone.numel() for cone in self.cones[self.n_linear:self.n_linear + self.n_soc])

        self.w = cp.empty(self.numel_linear + self.numel_soc, dtype=cpucones.cones[0].dtype)
        self.λ = cp.empty(self.numel_linear + self.numel_soc, dtype=cpucones.cones[0].dtype)
        self.η = cp.empty(self.n_soc, dtype=cpucones.cones[0].dtype)

        self.αp = cp.array([cone.α for cone in self.cones[self.n_linear + self.n_soc + self.n_exp:self.n_linear + self.n_soc + self.n_exp + self.n_pow]], dtype=cpucones.cones[0].dtype)
        self.H_dual = cp.empty((self.n_exp + self.n_pow, 3, 3), dtype=cpucones.cones[0].dtype)
        self.Hs = cp.empty((self.n_exp + self.n_pow, 3, 3), dtype=cpucones.cones[0].dtype)
        self.grad = cp.empty((self.n_exp + self.n_pow, 3), dtype=cpucones.cones[0].dtype)

        self.psd_dim = self.cones[self.n_linear + self.n_soc + self.n_exp + self.n_pow].n if self.n_psd > 0 else 0
        for i in range(1, self.n_psd):
            if self.psd_dim != self.cones[self.n_linear + self.n_soc + self.n_exp + self.n_pow + i].n:
                raise ValueError("Not all positive definite cones have the same dimensionality!")

        self.chol1 = cp.zeros((self.psd_dim, self.psd_dim, self.n_psd), dtype=cpucones.cones[0].dtype)
        self.chol2 = cp.zeros((self.psd_dim, self.psd_dim, self.n_psd), dtype=cpucones.cones[0].dtype)
        self.SVD = cp.zeros((self.psd_dim, self.psd_dim, self.n_psd), dtype=cpucones.cones[0].dtype)

        self.λpsd = cp.zeros((self.psd_dim, self.n_psd), dtype=cpucones.cones[0].dtype)
        self.Λisqrt = cp.zeros((self.psd_dim, self.n_psd), dtype=cpucones.cones[0].dtype)
        self.R = cp.zeros((self.psd_dim, self.psd_dim, self.n_psd), dtype=cpucones.cones[0].dtype)
        self.Rinv = cp.zeros((self.psd_dim, self.psd_dim, self.n_psd), dtype=cpucones.cones[0].dtype)
        self.Hspsd = cp.zeros((self.psd_dim * (self.psd_dim + 1) // 2, self.psd_dim * (self.psd_dim + 1) // 2, self.n_psd), dtype=cpucones.cones[0].dtype)

        self.workmat1 = cp.zeros((self.psd_dim, self.psd_dim, self.n_psd), dtype=cpucones.cones[0].dtype)
        self.workmat2 = cp.zeros((self.psd_dim, self.psd_dim, self.n_psd), dtype=cpucones.cones[0].dtype)
        self.workmat3 = cp.zeros((self.psd_dim, self.psd_dim, self.n_psd), dtype=cpucones.cones[0].dtype)
        self.workvec = cp.zeros(self.psd_dim * (self.psd_dim + 1) // 2 * self.n_psd, dtype=cpucones.cones[0].dtype)

        self.α = cp.empty(sum(cone.numel() for cone in self.cones), dtype=cpucones.cones[0].dtype)

    def __getitem__(self, i):
        return self.cones[i]

    def __len__(self):
        return len(self.cones)

    def __iter__(self):
        return iter(self.cones)

    def get_type_count(self, type):
        return self.type_counts.get(type, 0)
