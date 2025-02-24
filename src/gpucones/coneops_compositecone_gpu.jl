import cupy as cp
import numpy as np
from numba import cuda, njit

def degree(cones):
    return cones.degree

def numel(cones):
    return cones.numel

def is_symmetric(cones):
    return cones._is_symmetric

def allows_primal_dual_scaling(cones):
    return all(allows_primal_dual_scaling(cone) for cone in cones)

@njit
def margins_nonnegative(z, α, rng_cones, idx_inq, αmin):
    val = 0
    for i in idx_inq:
        rng = rng_cones[i]
        α[rng] = np.minimum(α[rng], z[rng])
        αmin = np.minimum(αmin, np.min(α[rng]))
        val += np.sum(np.maximum(0, z[rng]))
    return αmin, val

@njit
def margins_soc(z, α, rng_cones, n_shift, n_soc, αmin):
    val = 0
    for i in range(n_soc):
        shift_i = i + n_shift
        rng = rng_cones[shift_i]
        size_i = len(rng)
        zi = z[rng]
        α[shift_i] = np.minimum(α[shift_i], zi[0] - np.sqrt(np.sum(zi[1:]**2)))
        αmin = np.minimum(αmin, α[shift_i])
        val += np.sum(np.maximum(0, zi))
    return αmin, val

@njit
def margins_psd(Z, z, rng_cones, n_shift, n_psd, αmin):
    val = 0
    for i in range(n_psd):
        shift_i = i + n_shift
        rng = rng_cones[shift_i]
        zi = z[rng]
        Z[i] = np.reshape(zi, (int(np.sqrt(len(zi))), -1))
        e = np.linalg.eigvalsh(Z[i])
        αmin = np.minimum(αmin, np.min(e))
        val += np.sum(np.maximum(0, e))
    return αmin, val

def margins(cones, z, pd):
    αmin = np.finfo(z.dtype).max
    β = 0

    n_linear = cones.n_linear
    n_nn = cones.n_nn
    n_soc = cones.n_soc
    n_psd = cones.n_psd
    rng_cones = cones.rng_cones
    idx_inq = cones.idx_inq
    Z = cones.workmat1

    α = cones.α
    α.fill(αmin)

    if n_nn > 0:
        αmin, val = margins_nonnegative(z, α, rng_cones, idx_inq, αmin)
        β += val

    if n_soc > 0:
        n_shift = n_linear
        αmin, val = margins_soc(z, α, rng_cones, n_shift, n_soc, αmin)
        β += val

    if n_psd > 0:
        n_shift = n_linear + n_soc
        αmin, val = margins_psd(Z, z, rng_cones, n_shift, n_psd, αmin)
        β += val

    return αmin, β

@njit
def scaled_unit_shift_zero(z, rng_cones, idx_eq, pd):
    for i in idx_eq:
        rng = rng_cones[i]
        z[rng] = 0

@njit
def scaled_unit_shift_nonnegative(z, rng_cones, idx_inq, α):
    for i in idx_inq:
        rng = rng_cones[i]
        z[rng] = α

@njit
def scaled_unit_shift_soc(z, rng_cones, α, n_shift, n_soc):
    for i in range(n_soc):
        shift_i = i + n_shift
        rng = rng_cones[shift_i]
        z[rng[0]] += α

@njit
def scaled_unit_shift_psd(z, α, rng_cones, psd_dim, n_shift, n_psd):
    for i in range(n_psd):
        shift_i = i + n_shift
        rng = rng_cones[shift_i]
        for k in range(psd_dim):
            z[rng[k * (k + 1) // 2 + k]] += α

def scaled_unit_shift(cones, z, α, pd):
    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_psd = cones.n_psd
    psd_dim = cones.psd_dim
    rng_cones = cones.rng_cones
    idx_eq = cones.idx_eq
    idx_inq = cones.idx_inq

    scaled_unit_shift_zero(z, rng_cones, idx_eq, pd)
    scaled_unit_shift_nonnegative(z, rng_cones, idx_inq, α)

    if n_soc > 0:
        n_shift = n_linear
        scaled_unit_shift_soc(z, rng_cones, α, n_shift, n_soc)

    if n_psd > 0:
        n_shift = n_linear + n_soc
        scaled_unit_shift_psd(z, α, rng_cones, psd_dim, n_shift, n_psd)

    return

@njit
def unit_initialization_zero(z, s, rng_cones, idx_eq):
    for i in idx_eq:
        rng = rng_cones[i]
        z[rng] = 0
        s[rng] = 0

@njit
def unit_initialization_nonnegative(z, s, rng_cones, idx_inq):
    for i in idx_inq:
        rng = rng_cones[i]
        z[rng] = 1
        s[rng] = 1

@njit
def unit_initialization_soc(z, s, rng_cones, n_shift, n_soc):
    for i in range(n_soc):
        shift_i = i + n_shift
        rng = rng_cones[shift_i]
        z[rng[0]] = 1
        z[rng[1:]] = 0
        s[rng[0]] = 1
        s[rng[1:]] = 0

@njit
def unit_initialization_exp(z, s, rng_cones, n_shift, n_exp):
    for i in range(n_exp):
        shift_i = i + n_shift
        rng = rng_cones[shift_i]
        s[rng[0]] = -1.051383945322714
        s[rng[1]] = 0.556409619469370
        s[rng[2]] = 1.258967884768947
        z[rng] = s[rng]

@njit
def unit_initialization_pow(z, s, αp, rng_cones, n_shift, n_pow):
    for i in range(n_pow):
        shift_i = i + n_shift
        rng = rng_cones[shift_i]
        s[rng[0]] = np.sqrt(1 + αp[i])
        s[rng[1]] = np.sqrt(1 + (1 - αp[i]))
        s[rng[2]] = 0
        z[rng] = s[rng]

@njit
def unit_initialization_psd(z, s, rng_cones, psd_dim, n_shift, n_psd):
    for i in range(n_psd):
        shift_i = i + n_shift
        rng = rng_cones[shift_i]
        z[rng] = 0
        s[rng] = 0
        for k in range(psd_dim):
            z[rng[k * (k + 1) // 2 + k]] = 1
            s[rng[k * (k + 1) // 2 + k]] = 1

def unit_initialization(cones, z, s):
    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    psd_dim = cones.psd_dim

    αp = cones.αp
    rng_cones = cones.rng_cones
    idx_eq = cones.idx_eq
    idx_inq = cones.idx_inq

    unit_initialization_zero(z, s, rng_cones, idx_eq)
    unit_initialization_nonnegative(z, s, rng_cones, idx_inq)

    if n_soc > 0:
        n_shift = n_linear
        unit_initialization_soc(z, s, rng_cones, n_shift, n_soc)

    if n_exp > 0:
        n_shift = n_linear + n_soc
        unit_initialization_exp(z, s, rng_cones, n_shift, n_exp)

    if n_pow > 0:
        n_shift = n_linear + n_soc + n_exp
        unit_initialization_pow(z, s, αp, rng_cones, n_shift, n_pow)

    if n_psd > 0:
        n_shift = n_linear + n_soc + n_exp + n_pow
        unit_initialization_psd(z, s, rng_cones, psd_dim, n_shift, n_psd)

    return

@njit
def set_identity_scaling_nonnegative(w, rng_cones, idx_inq):
    for i in idx_inq:
        rng = rng_cones[i]
        w[rng] = 1

@njit
def set_identity_scaling_soc(w, η, rng_cones, n_linear, n_soc):
    for i in range(n_soc):
        shift_i = i + n_linear
        rng = rng_cones[shift_i]
        w[rng[0]] = 1
        w[rng[1:]] = 0
        η[i] = 1

@njit
def set_identity_scaling_psd(R, Rinv, Hspsd, psd_dim, n_psd):
    for i in range(n_psd):
        for k in range(psd_dim):
            R[k, k, i] = 1
            Rinv[k, k, i] = 1
        for k in range(psd_dim * (psd_dim + 1) // 2):
            Hspsd[k, k, i] = 1

def set_identity_scaling(cones):
    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_psd = cones.n_psd
    rng_cones = cones.rng_cones
    idx_inq = cones.idx_inq
    w = cones.w
    η = cones.η
    R = cones.R
    Rinv = cones.Rinv
    Hspsd = cones.Hspsd
    psd_dim = cones.psd_dim

    set_identity_scaling_nonnegative(w, rng_cones, idx_inq)

    if n_soc > 0:
        set_identity_scaling_soc(w, η, rng_cones, n_linear, n_soc)

    if n_psd > 0:
        set_identity_scaling_psd(R, Rinv, Hspsd, psd_dim, n_psd)

    return

@njit
def update_scaling_nonnegative(s, z, w, λ, rng_cones, idx_inq):
    for i in idx_inq:
        rng = rng_cones[i]
        w[rng] = s[rng] / np.sqrt(s[rng] * z[rng])
        λ[rng] = np.sqrt(s[rng] * z[rng])

@njit
def update_scaling_soc(s, z, w, λ, η, rng_cones, n_shift, n_soc):
    for i in range(n_soc):
        shift_i = i + n_shift
        rng = rng_cones[shift_i]
        zi = z[rng]
        si = s[rng]
        wi = w[rng]
        λi = λ[rng]

        zscale = np.sqrt(zi[0]**2 - np.sum(zi[1:]**2))
        sscale = np.sqrt(si[0]**2 - np.sum(si[1:]**2))

        η[i] = np.sqrt(sscale / zscale)

        wi[:] = si / sscale
        wi[0] += zi[0] / zscale
        wi[1:] -= zi[1:] / zscale

        wscale = np.sqrt(wi[0]**2 - np.sum(wi[1:]**2))
        wi /= wscale

        w1sq = np.sum(wi[1:]**2)
        wi[0] = np.sqrt(1 + w1sq)

        γi = 0.5 * wscale
        λi[0] = γi

        coef = 1 / (si[0] / sscale + zi[0] / zscale + 2 * γi)
        c1 = (γi + zi[0] / zscale) / sscale
        c2 = (γi + si[0] / sscale) / zscale
        λi[1:] = coef * (c1 * si[1:] + c2 * zi[1:])
        λi *= np.sqrt(sscale * zscale)

@njit
def update_scaling_psd(L1, L2, z, s, workmat1, λpsd, Λisqrt, R, Rinv, Hspsd, rng_cones, n_shift, n_psd):
    for i in range(n_psd):
        shift_i = i + n_shift
        rng = rng_cones[shift_i]
        zi = z[rng]
        si = s[rng]
        L2[i] = np.reshape(zi, (int(np.sqrt(len(zi))), -1))
        L1[i] = np.reshape(si, (int(np.sqrt(len(si))), -1))

        _, infoz = np.linalg.cholesky(L2[i])
        _, infos = np.linalg.cholesky(L1[i])

        if not (np.all(infoz == 0) and np.all(infos == 0)):
            return False

        tmp = workmat1
        tmp[i] = np.dot(L2[i].T, L1[i])
        U, S, V = np.linalg.svd(tmp[i])

        λpsd[:, i] = S
        Λisqrt[:, i] = 1 / np.sqrt(λpsd[:, i])

        R[i] = np.dot(L1[i], V)
        R[i] = np.dot(R[i], np.diag(Λisqrt[:, i]))

        Rinv[i] = np.dot(U.T, L2[i].T)
        Rinv[i] = np.dot(np.diag(Λisqrt[:, i]), Rinv[i])

        RRt = workmat1
        RRt[i] = np.dot(R[i], R[i].T)

        Hspsd[i] = np.kron(RRt[i], RRt[i])

    return True

def update_scaling(cones, s, z, μ, scaling_strategy):
    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    αp = cones.αp
    grad = cones.grad
    Hs = cones.Hs
    H_dual = cones.H_dual
    rng_cones = cones.rng_cones
    idx_inq = cones.idx_inq
    w = cones.w
    λ = cones.λ
    η = cones.η

    update_scaling_nonnegative(s, z, w, λ, rng_cones, idx_inq)

    if n_soc > 0:
        n_shift = n_linear
        update_scaling_soc(s, z, w, λ, η, rng_cones, n_shift, n_soc)

    if n_exp > 0:
        n_shift = n_linear + n_soc
        update_scaling_exp(s, z, grad, Hs, H_dual, rng_cones, μ, scaling_strategy, n_shift, n_exp)

    if n_pow > 0:
        n_shift = n_linear + n_soc + n_exp
        update_scaling_pow(s, z, grad, Hs, H_dual, αp, rng_cones, μ, scaling_strategy, n_shift, n_exp, n_pow)

    if n_psd > 0:
        n_shift = n_linear + n_soc + n_exp + n_pow
        update_scaling_psd(cones.chol1, cones.chol2, z, s, cones.workmat1, cones.λpsd, cones.Λisqrt, cones.R, cones.Rinv, cones.Hspsd, rng_cones, n_shift, n_psd)

    return True

@njit
def get_Hs_zero(Hsblocks, rng_blocks, idx_eq):
    for i in idx_eq:
        rng = rng_blocks[i]
        Hsblocks[rng] = 0

@njit
def get_Hs_nonnegative(Hsblocks, w, rng_cones, rng_blocks, idx_inq):
    for i in idx_inq:
        rng_cone = rng_cones[i]
        rng_block = rng_blocks[i]
        Hsblocks[rng_block] = 2 * np.outer(w[rng_cone], w[rng_cone])
        Hsblocks[rng_block[0]] -= 1
        for j in range(1, len(rng_cone)):
            Hsblocks[rng_block[j * len(rng_cone) + j]] += 1

@njit
def get_Hs_soc(Hsblocks, w, η, rng_cones, rng_blocks, n_shift, n_soc):
    for i in range(n_soc):
        shift_i = i + n_shift
        rng_cone = rng_cones[shift_i]
        rng_block = rng_blocks[shift_i]
        size_i = len(rng_cone)
        wi = w[rng_cone]
        Hsblocki = Hsblocks[rng_block]

        hidx = 0
        for col in range(size_i):
            wcol = wi[col]
            for row in range(size_i):
                Hsblocki[hidx] = 2 * wi[row] * wcol
                hidx += 1
        Hsblocki[0] -= 1
        for ind in range(1, size_i):
            Hsblocki[(ind - 1) * size_i + ind] += 1
        Hsblocki *= η[i]**2

@njit
def get_Hs_exp(Hsblocks, Hs, rng_blocks, n_shift, n_exp):
    for i in range(n_exp):
        shift_i = i + n_shift
        rng_block = rng_blocks[shift_i]
        Hsblocks[rng_block] = Hs[i].flatten()

@njit
def get_Hs_pow(Hsblocks, Hs, rng_blocks, n_shift, n_exp, n_pow):
    for i in range(n_pow):
        shift_i = i + n_shift
        rng_block = rng_blocks[shift_i]
        Hsblocks[rng_block] = Hs[n_exp + i].flatten()

@njit
def get_Hs_psd(Hsblocks, Hspsd, rng_blocks, n_shift, n_psd):
    for i in range(n_psd):
        shift_i = i + n_shift
        rng_block = rng_blocks[shift_i]
        Hsblocks[rng_block] = Hspsd[i].flatten()

def get_Hs(cones, Hsblocks):
    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    Hs = cones.Hs
    Hspsd = cones.Hspsd
    rng_blocks = cones.rng_blocks
    rng_cones = cones.rng_cones
    idx_eq = cones.idx_eq
    idx_inq = cones.idx_inq
    w = cones.w
    η = cones.η

    get_Hs_zero(Hsblocks, rng_blocks, idx_eq)
    get_Hs_nonnegative(Hsblocks, w, rng_cones, rng_blocks, idx_inq)

    if n_soc > 0:
        n_shift = n_linear
        get_Hs_soc(Hsblocks, w, η, rng_cones, rng_blocks, n_shift, n_soc)

    if n_exp > 0:
        n_shift = n_linear + n_soc
        get_Hs_exp(Hsblocks, Hs, rng_blocks, n_shift, n_exp)

    if n_pow > 0:
        n_shift = n_linear + n_soc + n_exp
        get_Hs_pow(Hsblocks, Hs, rng_blocks, n_shift, n_exp, n_pow)

    if n_psd > 0:
        n_shift = n_linear + n_soc + n_exp + n_pow
        get_Hs_psd(Hsblocks, Hspsd, rng_blocks, n_shift, n_psd)

    return

@njit
def mul_Hs_zero(y, rng_cones, idx_eq):
    for i in idx_eq:
        rng = rng_cones[i]
        y[rng] = 0

@njit
def mul_Hs_nonnegative(y, x, w, rng_cones, idx_inq):
    for i in idx_inq:
        rng = rng_cones[i]
        y[rng] = 2 * w[rng] * np.dot(w[rng], x[rng]) - x[rng]

@njit
def mul_Hs_soc(y, x, w, η, rng_cones, n_shift, n_soc):
    for i in range(n_soc):
        shift_i = i + n_shift
        rng = rng_cones[shift_i]
        size_i = len(rng)
        xi = x[rng]
        wi = w[rng]
        yi = y[rng]

        c = 2 * np.dot(wi, xi)
        yi[0] = -xi[0] + c * wi[0]
        yi[1:] = xi[1:] + c * wi[1:]
        yi *= η[i]**2

@njit
def mul_Hs_nonsymmetric(y, Hs, x, rng_cones, n_shift, n_nonsymmetric):
    for i in range(n_nonsymmetric):
        shift_i = i + n_shift
        rng = rng_cones[shift_i]
        y[rng] = np.dot(Hs[i], x[rng])

@njit
def mul_Hs_psd(y, x, Hspsd, rng_cones, n_shift, n_psd, psd_dim):
    for i in range(n_psd):
        shift_i = i + n_shift
        rng = rng_cones[shift_i]
        xi = x[rng]
        yi = y[rng]
        Hspsdi = Hspsd[i]

        n_tri_dim = psd_dim * (psd_dim + 1) // 2
        X = xi.reshape((n_tri_dim, n_psd))
        Y = yi.reshape((n_tri_dim, n_psd))

        Y[:] = np.dot(Hspsdi, X)

def mul_Hs(cones, y, x, work):
    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    psd_dim = cones.psd_dim
    Hs = cones.Hs
    Hspsd = cones.Hspsd
    rng_cones = cones.rng_cones
    idx_eq = cones.idx_eq
    idx_inq = cones.idx_inq
    w = cones.w
    η = cones.η

    mul_Hs_zero(y, rng_cones, idx_eq)
    mul_Hs_nonnegative(y, x, w, rng_cones, idx_inq)

    if n_soc > 0:
        n_shift = n_linear
        mul_Hs_soc(y, x, w, η, rng_cones, n_shift, n_soc)

    n_nonsymmetric = n_exp + n_pow
    if n_nonsymmetric > 0:
        n_shift = n_linear + n_soc
        mul_Hs_nonsymmetric(y, Hs, x, rng_cones, n_shift, n_nonsymmetric)

    if n_psd > 0:
        n_shift = n_linear + n_soc + n_exp + n_pow
        mul_Hs_psd(y, x, Hspsd, rng_cones, n_shift, n_psd, psd_dim)

    return

@njit
def affine_ds_zero(ds, rng_cones, idx_eq):
    for i in idx_eq:
        rng = rng_cones[i]
        ds[rng] = 0

@njit
def affine_ds_nonnegative(ds, λ, rng_cones, idx_inq):
    for i in idx_inq:
        rng = rng_cones[i]
        ds[rng] = λ[rng]**2

@njit
def affine_ds_soc(ds, λ, rng_cones, n_shift, n_soc):
    for i in range(n_soc):
        shift_i = i + n_shift
        rng = rng_cones[shift_i]
        size_i = len(rng)
        λi = λ[rng]
        dsi = ds[rng]

        dsi[0] = np.sum(λi**2)
        λi0 = λi[0]
        dsi[1:] = 2 * λi0 * λi[1:]

@njit
def affine_ds_nonsymmetric(ds, s, rng_cones, n_shift, n_nonsymmetric):
    for i in range(n_nonsymmetric):
        shift_i = i + n_shift
        rng = rng_cones[shift_i]
        ds[rng] = s[rng]

@njit
def affine_ds_psd(ds, λpsd, rng_cones, psd_dim, n_shift, n_psd):
    for i in range(n_psd):
        shift_i = i + n_shift
        rng = rng_cones[shift_i]
        dsi = ds[rng]
        λpsdi = λpsd[:, i]

        for k in range(psd_dim):
            dsi[k * (k + 1) // 2 + k] = λpsdi[k]**2

def affine_ds(cones, ds, s):
    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    rng_cones = cones.rng_cones
    idx_eq = cones.idx_eq
    idx_inq = cones.idx_inq
    λ = cones.λ
    psd_dim = cones.psd_dim
    λpsd = cones.λpsd

    affine_ds_zero(ds, rng_cones, idx_eq)
    affine_ds_nonnegative(ds, λ, rng_cones, idx_inq)

    if n_soc > 0:
        n_shift = n_linear
        affine_ds_soc(ds, λ, rng_cones, n_shift, n_soc)

    n_nonsymmetric = n_exp + n_pow
    if n_nonsymmetric > 0:
        n_shift = n_linear + n_soc
        affine_ds_nonsymmetric(ds, s, rng_cones, n_shift, n_nonsymmetric)

    if n_psd > 0:
        n_shift = n_linear + n_soc + n_exp + n_pow
        affine_ds_psd(ds, λpsd, rng_cones, psd_dim, n_shift, n_psd)

    return

@njit
def combined_ds_shift_zero(shift, rng_cones, idx_eq):
    for i in idx_eq:
        rng = rng_cones[i]
        shift[rng] = 0

@njit
def combined_ds_shift_nonnegative(shift, step_z, step_s, w, σμ, rng_cones, idx_inq):
    for i in idx_inq:
        rng = rng_cones[i]
        shift[rng] = w[rng] * (step_z[rng] + step_s[rng]) - σμ

@njit
def combined_ds_shift_soc(shift, step_z, step_s, w, η, rng_cones, n_shift, n_soc, σμ):
    for i in range(n_soc):
        shift_i = i + n_shift
        rng = rng_cones[shift_i]
        size_i = len(rng)
        step_zi = step_z[rng]
        step_si = step_s[rng]
        wi = w[rng]
        shifti = shift[rng]

        tmp = step_zi.copy()
        ζ = np.dot(wi[1:], tmp[1:])
        c = tmp[0] + ζ / (1 + wi[0])
        step_zi[0] = η[i] * (wi[0] * tmp[0] + ζ)
        step_zi[1:] = η[i] * (tmp[1:] + c * wi[1:])

        tmp = step_si.copy()
        ζ = np.dot(wi[1:], tmp[1:])
        c = -tmp[0] + ζ / (1 + wi[0])
        step_si[0] = (1 / η[i]) * (wi[0] * tmp[0] - ζ)
        step_si[1:] = (1 / η[i]) * (tmp[1:] + c * wi[1:])

        val = np.dot(step_si, step_zi)
        shifti[0] = val - σμ
        s0 = step_si[0]
        z0 = step_zi[0]
        shifti[1:] = s0 * step_zi[1:] + z0 * step_si[1:]

@njit
def combined_ds_shift_exp(shift, step_z, step_s, z, grad, H_dual, rng_cones, σμ, n_shift, n_exp):
    for i in range(n_exp):
        shift_i = i + n_shift
        rng = rng_cones[shift_i]
        step_zi = step_z[rng]
        step_si = step_s[rng]
        zi = z[rng]
        gradi = grad[i]
        Hi = H_dual[i]
        shifti = shift[rng]

        η = np.zeros(3)
        higher_correction_exp(Hi, zi, η, step_si, step_zi)

        shifti[:] = gradi * σμ - η

@njit
def combined_ds_shift_pow(shift, step_z, step_s, z, grad, H_dual, αp, rng_cones, σμ, n_shift, n_exp, n_pow):
    for i in range(n_pow):
        shift_i = i + n_shift
        rng = rng_cones[shift_i]
        step_zi = step_z[rng]
        step_si = step_s[rng]
        zi = z[rng]
        gradi = grad[n_exp + i]
        Hi = H_dual[n_exp + i]
        shifti = shift[rng]

        η = np.zeros(3)
        higher_correction_pow(Hi, zi, η, step_si, step_zi, αp[i])

        shifti[:] = gradi * σμ - η

@njit
def combined_ds_shift_psd(cones, shift, step_z, step_s, n_shift, n_psd, σμ):
    tmp = shift.copy()
    R = cones.R
    Rinv = cones.Rinv
    rng_cones = cones.rng_cones
    workmat1 = cones.workmat1
    workmat2 = cones.workmat2
    workmat3 = cones.workmat3
    psd_dim = cones.psd_dim

    rng = np.concatenate([rng_cones[i] for i in range(n_shift + 1, n_shift + n_psd + 1)])

    tmp[rng] = step_z[rng]
    mul_Wx_psd(step_z, tmp, R, rng_cones, workmat1, workmat2, workmat3, n_shift, n_psd, False)

    tmp[rng] = step_s[rng]
    mul_WTx_psd(step_s, tmp, Rinv, rng_cones, workmat1, workmat2, workmat3, n_shift, n_psd, False)

    svec_to_mat(workmat1, step_z, rng_cones, n_shift, n_psd)
    svec_to_mat(workmat2, step_s, rng_cones, n_shift, n_psd)
    workmat3 = np.dot(workmat1, workmat2)
    workmat3 = (workmat3 + workmat3.T) / 2

    mat_to_svec(shift, workmat3, rng_cones, n_shift, n_psd)
    scaled_unit_shift_psd(shift, -σμ, rng_cones, psd_dim, n_shift, n_psd)

def combined_ds_shift(cones, shift, step_z, step_s, z, σμ):
    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    grad = cones.grad
    H_dual = cones.H_dual
    αp = cones.αp
    rng_cones = cones.rng_cones
    idx_eq = cones.idx_eq
    idx_inq = cones.idx_inq
    w = cones.w
    η = cones.η

    combined_ds_shift_zero(shift, rng_cones, idx_eq)
    combined_ds_shift_nonnegative(shift, step_z, step_s, w, σμ, rng_cones, idx_inq)

    if n_soc > 0:
        n_shift = n_linear
        combined_ds_shift_soc(shift, step_z, step_s, w, η, rng_cones, n_shift, n_soc, σμ)

    if n_exp > 0:
        n_shift = n_linear + n_soc
        combined_ds_shift_exp(shift, step_z, step_s, z, grad, H_dual, rng_cones, σμ, n_shift, n_exp)

    if n_pow > 0:
        n_shift = n_linear + n_soc + n_exp
        combined_ds_shift_pow(shift, step_z, step_s, z, grad, H_dual, αp, rng_cones, σμ, n_shift, n_exp, n_pow)

    if n_psd > 0:
        n_shift = n_linear + n_soc + n_exp + n_pow
        combined_ds_shift_psd(cones, shift, step_z, step_s, n_shift, n_psd, σμ)

    return

@njit
def Δs_from_Δz_offset_zero(out, rng_cones, idx_eq):
    for i in idx_eq:
        rng = rng_cones[i]
        out[rng] = 0

@njit
def Δs_from_Δz_offset_nonnegative(out, ds, z, rng_cones, idx_inq):
    for i in idx_inq:
        rng = rng_cones[i]
        out[rng] = ds[rng] - z[rng]

@njit
def Δs_from_Δz_offset_soc(out, ds, z, w, λ, η, rng_cones, n_shift, n_soc):
    for i in range(n_soc):
        shift_i = i + n_shift
        rng = rng_cones[shift_i]
        size_i = len(rng)
        dsi = ds[rng]
        zi = z[rng]
        wi = w[rng]
        λi = λ[rng]
        outi = out[rng]

        reszi = np.sqrt(zi[0]**2 - np.sum(zi[1:]**2))
        λ1ds1 = np.dot(λi[1:], dsi[1:])
        w1ds1 = np.dot(wi[1:], dsi[1:])

        outi[:] = -zi
        outi[0] = zi[0]

        c = λi[0] * dsi[0] - λ1ds1
        outi *= c / reszi

        outi[0] += η[i] * w1ds1
        outi[1:] += η[i] * (dsi[1:] + w1ds1 / (1 + wi[0]) * wi[1:])

        outi /= λi[0]

@njit
def Δs_from_Δz_offset_nonsymmetric(out, ds, rng_cones, n_shift, n_nonsymmetric):
    for i in range(n_nonsymmetric):
        shift_i = i + n_shift
        rng = rng_cones[shift_i]
        out[rng] = ds[rng]

@njit
def Δs_from_Δz_offset_psd(cones, out, ds, work, n_shift, n_psd):
    R = cones.R
    λpsd = cones.λpsd
    rng_cones = cones.rng_cones
    workmat1 = cones.workmat1
    workmat2 = cones.workmat2
    workmat3 = cones.workmat3
    psd_dim = cones.psd_dim

    svec_to_mat(workmat2, ds, rng_cones, n_shift, n_psd)
    op_λ(workmat1, workmat2, λpsd, psd_dim, n_psd)
    mat_to_svec(work, workmat1, rng_cones, n_shift, n_psd)

    mul_WTx_psd(out, work, R, rng_cones, workmat1, workmat2, workmat3, n_shift, n_psd, False)

def Δs_from_Δz_offset(cones, out, ds, work, z):
    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    rng_cones = cones.rng_cones
    idx_eq = cones.idx_eq
    idx_inq = cones.idx_inq
    w = cones.w
    λ = cones.λ
    η = cones.η

    Δs_from_Δz_offset_zero(out, rng_cones, idx_eq)
    Δs_from_Δz_offset_nonnegative(out, ds, z, rng_cones, idx_inq)

    if n_soc > 0:
        n_shift = n_linear
        Δs_from_Δz_offset_soc(out, ds, z, w, λ, η, rng_cones, n_shift, n_soc)

    n_nonsymmetric = n_exp + n_pow
    if n_nonsymmetric > 0:
        n_shift = n_linear + n_soc
        Δs_from_Δz_offset_nonsymmetric(out, ds, rng_cones, n_shift, n_nonsymmetric)

    if n_psd > 0:
        n_shift = n_linear + n_soc + n_exp + n_pow
        Δs_from_Δz_offset_psd(cones, out, ds, work, n_shift, n_psd)

    return

@njit
def step_length_nonnegative(dz, ds, z, s, α, αmax, rng_cones, idx_inq):
    for i in idx_inq:
        rng = rng_cones[i]
        α[rng] = np.minimum(αmax, np.minimum(-z[rng] / dz[rng], -s[rng] / ds[rng]))
        αmax = np.minimum(αmax, np.min(α[rng]))
    return αmax

@njit
def step_length_soc(dz, ds, z, s, α, αmax, rng_cones, n_shift, n_soc):
    for i in range(n_soc):
        shift_i = i + n_shift
        rng = rng_cones[shift_i]
        α[shift_i] = np.minimum(αmax, np.minimum(-z[rng[0]] / dz[rng[0]], -s[rng[0]] / ds[rng[0]]))
        αmax = np.minimum(αmax, α[shift_i])
    return αmax

@njit
def step_length_psd(dz, ds, Λisqrt, d, Rx, Rinv, workmat1, workmat2, workmat3, αmax, rng_cones, n_shift, n_psd):
    for i in range(n_psd):
        shift_i = i + n_shift
        rng = rng_cones[shift_i]
        d[rng] = dz[rng]
        αmax = np.minimum(αmax, step_length_psd_component(workmat1, d, Λisqrt, n_psd, αmax))
        d[rng] = ds[rng]
        αmax = np.minimum(αmax, step_length_psd_component(workmat1, d, Λisqrt, n_psd, αmax))
    return αmax

@njit
def step_length_psd_component(workΔ, d, Λisqrt, n_psd, αmax):
    workΔ = np.dot(Λisqrt, d)
    e = np.linalg.eigvalsh(workΔ)
    γ = np.min(e)
    if γ < 0:
        return np.minimum(1 / -γ, αmax)
    else:
        return αmax

def step_length(cones, dz, ds, z, s, α, αmax):
    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_psd = cones.n_psd
    rng_cones = cones.rng_cones
    idx_inq = cones.idx_inq
    Λisqrt = cones.Λisqrt
    d = cones.d
    Rx = cones.R
    Rinv = cones.Rinv
    workmat1 = cones.workmat1
    workmat2 = cones.workmat2
    workmat3 = cones.workmat3

    αmax = step_length_nonnegative(dz, ds, z, s, α, αmax, rng_cones, idx_inq)

    if n_soc > 0:
        n_shift = n_linear
        αmax = step_length_soc(dz, ds, z, s, α, αmax, rng_cones, n_shift, n_soc)

    if n_psd > 0:
        n_shift = n_linear + n_soc
        αmax = step_length_psd(dz, ds, Λisqrt, d, Rx, Rinv, workmat1, workmat2, workmat3, αmax, rng_cones, n_shift, n_psd)

    return αmax

@njit
def compute_barrier_nonnegative(barrier, z, s, dz, ds, α, rng_cones, idx_inq):
    for i in idx_inq:
        rng = rng_cones[i]
        barrier[rng] = -np.log(z[rng] + α * dz[rng]) - np.log(s[rng] + α * ds[rng])
    return np.sum(barrier)

@njit
def compute_barrier_soc(barrier, z, s, dz, ds, α, rng_cones, n_shift, n_soc):
    for i in range(n_soc):
        shift_i = i + n_shift
        rng = rng_cones[shift_i]
        zi = z[rng]
        si = s[rng]
        dzi = dz[rng]
        dsi = ds[rng]
        cur_z = zi + α * dzi
        cur_s = si + α * dsi
        barrier_d = -np.log(np.sqrt(cur_z[0]**2 - np.sum(cur_z[1:]**2)))
        barrier_p = -np.log(np.sqrt(cur_s[0]**2 - np.sum(cur_s[1:]**2)))
        barrier[shift_i] = barrier_d + barrier_p
    return np.sum(barrier)

@njit
def compute_barrier_psd(barrier, z, s, dz, ds, α, workmat1, workvec, rng_cones, psd_dim, n_shift, n_psd):
    for i in range(n_psd):
        shift_i = i + n_shift
        rng = rng_cones[shift_i]
        dsi = ds[rng]
        dzi = dz[rng]
        zi = z[rng]
        si = s[rng]
        cur_z = zi + α * dzi
        cur_s = si + α * dsi
        Q = workmat1
        q = workvec
        q[rng] = cur_z
        Q[i] = np.reshape(q[rng], (int(np.sqrt(len(q[rng]))), -1))
        _, info = np.linalg.cholesky(Q[i])
        if np.all(info == 0):
            barrier_d = 2 * np.sum(np.log(np.diag(Q[i])))
        else:
            barrier_d = np.inf
        q[rng] = cur_s
        Q[i] = np.reshape(q[rng], (int(np.sqrt(len(q[rng]))), -1))
        _, info = np.linalg.cholesky(Q[i])
        if np.all(info == 0):
            barrier_p = 2 * np.sum(np.log(np.diag(Q[i])))
        else:
            barrier_p = np.inf
        barrier[shift_i] = -barrier_d - barrier_p
    return np.sum(barrier)

def compute_barrier(cones, barrier, z, s, dz, ds, α):
    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_psd = cones.n_psd
    rng_cones = cones.rng_cones
    idx_inq = cones.idx_inq
    workmat1 = cones.workmat1
    workvec = cones.workvec
    psd_dim = cones.psd_dim

    barrier[:] = 0

    barrier_sum = compute_barrier_nonnegative(barrier, z, s, dz, ds, α, rng_cones, idx_inq)

    if n_soc > 0:
        n_shift = n_linear
        barrier_sum += compute_barrier_soc(barrier, z, s, dz, ds, α, rng_cones, n_shift, n_soc)

    if n_psd > 0:
        n_shift = n_linear + n_soc
        barrier_sum += compute_barrier_psd(barrier, z, s, dz, ds, α, workmat1, workvec, rng_cones, psd_dim, n_shift, n_psd)

    return barrier_sum
