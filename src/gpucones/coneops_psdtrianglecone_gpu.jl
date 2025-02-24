import cupy as cp
import numpy as np
from numba import cuda, njit

@njit
def margins_psd(Z, z, rng_cones, n_shift, n_psd, αmin):
    svec_to_mat_gpu(Z, z, rng_cones, n_shift, n_psd)
    e = np.linalg.eigvalsh(Z)
    αmin = min(αmin, np.min(e))
    e = np.maximum(e, 0)
    return αmin, np.sum(e)

@njit
def scaled_unit_shift_psd(z, α, rng_cones, psd_dim, n_shift, n_psd):
    for i in range(n_psd):
        shift_i = i + n_shift
        rng_cone_i = rng_cones[shift_i]
        for k in range(psd_dim):
            z[rng_cone_i[k * (k + 1) // 2 + k]] += α

@njit
def unit_initialization_psd(z, s, rng_cones, psd_dim, n_shift, n_psd):
    for i in range(n_psd):
        shift_i = i + n_shift
        rng_cone_i = rng_cones[shift_i]
        s[rng_cone_i] = 0
        z[rng_cone_i] = 0
    α = 1
    scaled_unit_shift_psd(z, α, rng_cones, psd_dim, n_shift, n_psd)
    scaled_unit_shift_psd(s, α, rng_cones, psd_dim, n_shift, n_psd)

@njit
def set_identity_scaling_psd(R, Rinv, Hspsd, psd_dim, n_psd):
    for i in range(n_psd):
        for k in range(psd_dim):
            R[k, k, i] = 1
            Rinv[k, k, i] = 1
        for k in range(psd_dim * (psd_dim + 1) // 2):
            Hspsd[k, k, i] = 1

def update_scaling_psd(L1, L2, z, s, workmat1, λpsd, Λisqrt, R, Rinv, Hspsd, rng_cones, n_shift, n_psd):
    svec_to_mat_gpu(L2, z, rng_cones, n_shift, n_psd)
    svec_to_mat_gpu(L1, s, rng_cones, n_shift, n_psd)

    _, infoz = np.linalg.cholesky(L2)
    _, infos = np.linalg.cholesky(L1)

    if not (np.all(infoz == 0) and np.all(infos == 0)):
        return False

    tmp = workmat1
    tmp = np.dot(L2.T, L1)
    U, S, V = np.linalg.svd(tmp)

    λpsd[:] = S
    Λisqrt[:] = 1 / np.sqrt(λpsd)

    R[:] = np.dot(L1, V)
    R[:] = np.dot(R, np.diag(Λisqrt))

    Rinv[:] = np.dot(U.T, L2.T)
    Rinv[:] = np.dot(np.diag(Λisqrt), Rinv)

    RRt = workmat1
    RRt[:] = np.dot(R, R.T)

    Hspsd[:] = np.kron(RRt, RRt)

    return True

@njit
def get_Hs_psd(Hsblocks, Hs, rng_blocks, n_shift, n_psd):
    for i in range(n_psd):
        shift_i = i + n_shift
        rng_i = rng_blocks[shift_i]
        Hsblocks[rng_i] = Hs[i].flatten()

def mul_Hs_psd(y, x, Hspsd, rng_cones, n_shift, n_psd, psd_dim):
    rng = np.concatenate([rng_cones[i] for i in range(n_shift + 1, n_shift + n_psd + 1)])
    tmpx = x[rng]
    tmpy = y[rng]

    n_tri_dim = psd_dim * (psd_dim + 1) // 2
    n_psd_int64 = int(n_psd)

    X = tmpx.reshape((n_tri_dim, n_psd_int64))
    Y = tmpy.reshape((n_tri_dim, n_psd_int64))

    Y[:] = np.dot(Hspsd, X)

@njit
def affine_ds_psd(ds, λpsd, rng_cones, psd_dim, n_shift, n_psd):
    for i in range(n_psd):
        shift_idx = rng_cones[n_shift + i].start - 1
        for k in range(psd_dim):
            ds[shift_idx + k * (k + 1) // 2 + k] = λpsd[k, i] ** 2

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

    svec_to_mat_gpu(workmat1, step_z, rng_cones, n_shift, n_psd)
    svec_to_mat_gpu(workmat2, step_s, rng_cones, n_shift, n_psd)
    workmat3 = np.dot(workmat1, workmat2)
    workmat3 = (workmat3 + workmat3.T) / 2

    mat_to_svec_gpu(shift, workmat3, rng_cones, n_shift, n_psd)
    scaled_unit_shift_psd(shift, -σμ, rng_cones, psd_dim, n_shift, n_psd)

@njit
def op_λ(X, Z, λpsd, psd_dim, n_psd):
    for i in range(n_psd):
        Xi = X[:, :, i]
        Zi = Z[:, :, i]
        λi = λpsd[:, i]
        for k in range(psd_dim):
            for j in range(psd_dim):
                Xi[k, j] = 2 * Zi[k, j] / (λi[k] + λi[j])

def Δs_from_Δz_offset_psd(cones, out, ds, work, n_shift, n_psd):
    R = cones.R
    λpsd = cones.λpsd
    rng_cones = cones.rng_cones
    workmat1 = cones.workmat1
    workmat2 = cones.workmat2
    workmat3 = cones.workmat3
    psd_dim = cones.psd_dim

    svec_to_mat_gpu(workmat2, ds, rng_cones, n_shift, n_psd)
    op_λ(workmat1, workmat2, λpsd, psd_dim, n_psd)
    mat_to_svec_gpu(work, workmat1, rng_cones, n_shift, n_psd)

    mul_WTx_psd(out, work, R, rng_cones, workmat1, workmat2, workmat3, n_shift, n_psd, False)

def step_length_psd(dz, ds, Λisqrt, d, Rx, Rinv, workmat1, workmat2, workmat3, αmax, rng_cones, n_shift, n_psd):
    workΔ = workmat1
    mul_Wx_psd(d, dz, Rx, rng_cones, workmat1, workmat2, workmat3, n_shift, n_psd, True)
    αz = step_length_psd_component(workΔ, d, Λisqrt, n_psd, αmax)

    mul_WTx_psd(d, ds, Rinv, rng_cones, workmat1, workmat2, workmat3, n_shift, n_psd, True)
    αs = step_length_psd_component(workΔ, d, Λisqrt, n_psd, αmax)

    αmax = min(αmax, αz, αs)

    return αmax

@njit
def logdet_barrier_psd(barrier, fact, psd_dim, n_psd):
    for i in range(n_psd):
        val = 0
        for k in range(psd_dim):
            val += np.log(fact[k, k, i])
        barrier[i] = 2 * val

def compute_barrier_psd(barrier, z, s, dz, ds, α, workmat1, workvec, rng_cones, psd_dim, n_shift, n_psd):
    rng = np.concatenate([rng_cones[i] for i in range(n_shift + 1, n_shift + n_psd + 1)])

    barrier_d = logdet_barrier_psd(barrier, z + α * dz, psd_dim, n_psd)
    barrier_p = logdet_barrier_psd(barrier, s + α * ds, psd_dim, n_psd)
    return -barrier_d - barrier_p

def mul_Wx_psd(y, x, Rx, rng_cones, workmat1, workmat2, workmat3, n_shift, n_psd, step_search):
    X, Y, tmp = workmat1, workmat2, workmat3

    svec_to_mat_gpu(X, x, rng_cones, n_shift, n_psd)

    tmp = np.dot(Rx.T, X)
    Y = np.dot(tmp, Rx)

    if step_search:
        mat_to_svec_no_shift_gpu(y, Y, n_psd)
    else:
        mat_to_svec_gpu(y, Y, rng_cones, n_shift, n_psd)

def mul_WTx_psd(y, x, Rx, rng_cones, workmat1, workmat2, workmat3, n_shift, n_psd, step_search):
    X, Y, tmp = workmat1, workmat2, workmat3

    svec_to_mat_gpu(X, x, rng_cones, n_shift, n_psd)

    tmp = np.dot(X, Rx.T)
    Y = np.dot(Rx, tmp)

    if step_search:
        mat_to_svec_no_shift_gpu(y, Y, n_psd)
    else:
        mat_to_svec_gpu(y, Y, rng_cones, n_shift, n_psd)

@njit
def step_length_psd_component(workΔ, d, Λisqrt, n_psd, αmax):
    svec_to_mat_no_shift_gpu(workΔ, d, n_psd)
    workΔ = np.dot(Λisqrt, workΔ)
    e = np.linalg.eigvalsh(workΔ)
    γ = np.min(e)
    if γ < 0:
        return min(1 / -γ, αmax)
    else:
        return αmax

@njit
def svec_to_mat(Z, z):
    dim = Z.shape[0]
    idx = 0
    for j in range(dim):
        for i in range(j + 1):
            Z[i, j] = z[idx]
            Z[j, i] = z[idx]
            idx += 1

def svec_to_mat_gpu(Z, z, rng_blocks, n_shift, n_psd):
    for i in range(n_psd):
        shift_i = i + n_shift
        rng_i = rng_blocks[shift_i]
        Zi = Z[:, :, i]
        zi = z[rng_i]
        svec_to_mat(Zi, zi)

def svec_to_mat_no_shift_gpu(Z, z, n_psd):
    dim = Z.shape[0]
    for i in range(n_psd):
        Zi = Z[:, :, i]
        rng_i = slice(i * (dim * (dim + 1)) // 2, (i + 1) * (dim * (dim + 1)) // 2)
        zi = z[rng_i]
        svec_to_mat(Zi, zi)

@njit
def mat_to_svec(z, Z):
    dim = Z.shape[0]
    idx = 0
    for j in range(dim):
        for i in range(j + 1):
            z[idx] = Z[i, j]
            idx += 1

def mat_to_svec_gpu(z, Z, rng_blocks, n_shift, n_psd):
    for i in range(n_psd):
        shift_i = i + n_shift
        rng_i = rng_blocks[shift_i]
        Zi = Z[:, :, i]
        zi = z[rng_i]
        mat_to_svec(zi, Zi)

def mat_to_svec_no_shift_gpu(z, Z, n_psd):
    dim = Z.shape[0]
    for i in range(n_psd):
        Zi = Z[:, :, i]
        rng_i = slice(i * (dim * (dim + 1)) // 2, (i + 1) * (dim * (dim + 1)) // 2)
        zi = z[rng_i]
        mat_to_svec(zi, Zi)

def skron_batched(out, A):
    n = out.shape[2]
    for i in range(n):
        outi = out[:, :, i]
        Ai = A[:, :, i]
        skron_full(outi, Ai)

def skron_full(out, A):
    sqrt2 = np.sqrt(2)
    n = A.shape[0]

    col = 0
    for l in range(n):
        for k in range(l + 1):
            row = 0
            kl_eq = k == l

            for j in range(n):
                Ajl = A[j, l]
                Ajk = A[j, k]

                for i in range(j + 1):
                    if row > col:
                        break
                    ij_eq = i == j

                    if not ij_eq and not kl_eq:
                        out[row, col] = A[i, k] * Ajl + A[i, l] * Ajk
                    elif ij_eq and not kl_eq:
                        out[row, col] = sqrt2 * Ajl * Ajk
                    elif not ij_eq and kl_eq:
                        out[row, col] = sqrt2 * A[i, l] * Ajk
                    else:
                        out[row, col] = Ajl * Ajl

                    out[col, row] = out[row, col]

                    row += 1
            col += 1

def right_mul_batched(A, B, C):
    n2 = A.shape[1]
    n = n2 * A.shape[2]
    for i in range(n):
        k, j = divmod(i, n2)
        val = B[j, k]
        for l in range(A.shape[0]):
            C[l, j, k] = val * A[l, j, k]

def left_mul_batched(A, B, C):
    n2 = B.shape[1]
    n = n2 * B.shape[2]
    for i in range(n):
        k, j = divmod(i, n2)
        val = A[j, k]
        for l in range(A.shape[0]):
            C[j, l, k] = val * B[j, l, k]
