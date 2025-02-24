import cupy as cp
import numpy as np
from numba import cuda, njit

@cuda.jit
def _kernel_margins_soc(z, α, rng_cones, n_shift, n_soc):
    i = cuda.grid(1)
    if i < n_soc:
        shift_i = i + n_shift
        rng_cone_i = rng_cones[shift_i]
        size_i = len(rng_cone_i)
        zi = z[rng_cone_i]
        val = 0.0
        for j in range(1, size_i):
            val += zi[j] * zi[j]
        α[i] = zi[0] - np.sqrt(val)

def margins_soc(z, α, rng_cones, n_shift, n_soc, αmin):
    threads_per_block = 128
    blocks_per_grid = (n_soc + (threads_per_block - 1)) // threads_per_block
    _kernel_margins_soc[blocks_per_grid, threads_per_block](z, α, rng_cones, n_shift, n_soc)
    αsoc = α[:n_soc]
    αmin = min(αmin, np.min(αsoc))
    αsoc = cp.maximum(0, αsoc)
    return αmin, cp.sum(αsoc)

@cuda.jit
def _kernel_scaled_unit_shift_soc(z, α, rng_cones, n_shift, n_soc):
    i = cuda.grid(1)
    if i < n_soc:
        shift_i = i + n_shift
        rng_cone_i = rng_cones[shift_i]
        zi = z[rng_cone_i]
        zi[0] += α

def scaled_unit_shift_soc(z, rng_cones, α, n_shift, n_soc):
    threads_per_block = 128
    blocks_per_grid = (n_soc + (threads_per_block - 1)) // threads_per_block
    _kernel_scaled_unit_shift_soc[blocks_per_grid, threads_per_block](z, α, rng_cones, n_shift, n_soc)

@cuda.jit
def _kernel_unit_initialization_soc(z, s, rng_cones, n_linear, n_soc):
    i = cuda.grid(1)
    if i < n_soc:
        shift_i = i + n_linear
        rng_cone_i = rng_cones[shift_i]
        zi = z[rng_cone_i]
        si = s[rng_cone_i]
        zi[0] = 1.0
        for j in range(1, len(zi)):
            zi[j] = 0.0
        si[0] = 1.0
        for j in range(1, len(si)):
            si[j] = 0.0

def unit_initialization_soc(z, s, rng_cones, n_shift, n_soc):
    threads_per_block = 128
    blocks_per_grid = (n_soc + (threads_per_block - 1)) // threads_per_block
    _kernel_unit_initialization_soc[blocks_per_grid, threads_per_block](z, s, rng_cones, n_shift, n_soc)

@cuda.jit
def _kernel_set_identity_scaling_soc(w, η, rng_cones, n_linear, n_soc):
    i = cuda.grid(1)
    if i < n_soc:
        shift_i = i + n_linear
        rng_cone_i = rng_cones[shift_i]
        size_i = len(rng_cone_i)
        wi = w[rng_cone_i]
        wi[0] = 1.0
        for j in range(1, size_i):
            wi[j] = 0.0
        η[i] = 1.0

def set_identity_scaling_soc(w, η, rng_cones, n_shift, n_soc):
    threads_per_block = 128
    blocks_per_grid = (n_soc + (threads_per_block - 1)) // threads_per_block
    _kernel_set_identity_scaling_soc[blocks_per_grid, threads_per_block](w, η, rng_cones, n_shift, n_soc)

@cuda.jit
def _kernel_update_scaling_soc(s, z, w, λ, η, rng_cones, n_shift, n_soc):
    i = cuda.grid(1)
    if i < n_soc:
        shift_i = i + n_shift
        rng_i = rng_cones[shift_i]
        zi = z[rng_i]
        si = s[rng_i]
        wi = w[rng_i]
        λi = λ[rng_i]
        zscale = _sqrt_soc_residual_gpu(zi)
        sscale = _sqrt_soc_residual_gpu(si)
        η[i] = np.sqrt(sscale / zscale)
        for k in range(len(rng_i)):
            w[k] = si[k] / sscale
        wi[0] += zi[0] / zscale
        for j in range(1, len(wi)):
            wi[j] -= zi[j] / zscale
        wscale = _sqrt_soc_residual_gpu(wi)
        wi /= wscale
        w1sq = 0.0
        for j in range(1, len(wi)):
            w1sq += wi[j] * wi[j]
        wi[0] = np.sqrt(1 + w1sq)
        γi = 0.5 * wscale
        λi[0] = γi
        coef = 1 / (si[0] / sscale + zi[0] / zscale + 2 * γi)
        c1 = (γi + zi[0] / zscale) / sscale
        c2 = (γi + si[0] / sscale) / zscale
        for j in range(1, len(λi)):
            λi[j] = coef * (c1 * si[j] + c2 * zi[j])
        λi *= np.sqrt(sscale * zscale)

def update_scaling_soc(s, z, w, λ, η, rng_cones, n_shift, n_soc):
    threads_per_block = 128
    blocks_per_grid = (n_soc + (threads_per_block - 1)) // threads_per_block
    _kernel_update_scaling_soc[blocks_per_grid, threads_per_block](s, z, w, λ, η, rng_cones, n_shift, n_soc)

@cuda.jit
def _kernel_get_Hs_soc(Hsblocks, w, η, rng_cones, rng_blocks, n_linear, n_soc):
    i = cuda.grid(1)
    if i < n_soc:
        shift_i = i + n_linear
        rng_cone_i = rng_cones[shift_i]
        rng_block_i = rng_blocks[shift_i]
        size_i = len(rng_cone_i)
        wi = w[rng_cone_i]
        Hsblocki = Hsblocks[rng_block_i]
        hidx = 0
        for col in range(size_i):
            wcol = wi[col]
            for row in range(size_i):
                Hsblocki[hidx] = 2 * wi[row] * wcol
                hidx += 1
        Hsblocki[0] -= 1.0
        for ind in range(1, size_i):
            Hsblocki[(ind - 1) * size_i + ind] += 1.0
        Hsblocki *= η[i] ** 2

def get_Hs_soc(Hsblocks, w, η, rng_cones, rng_blocks, n_shift, n_soc):
    threads_per_block = 128
    blocks_per_grid = (n_soc + (threads_per_block - 1)) // threads_per_block
    _kernel_get_Hs_soc[blocks_per_grid, threads_per_block](Hsblocks, w, η, rng_cones, rng_blocks, n_shift, n_soc)

@cuda.jit
def _kernel_mul_Hs_soc(y, x, w, η, rng_cones, n_linear, n_soc):
    i = cuda.grid(1)
    if i < n_soc:
        shift_i = i + n_linear
        rng_cone_i = rng_cones[shift_i]
        size_i = len(rng_cone_i)
        xi = x[rng_cone_i]
        yi = y[rng_cone_i]
        wi = w[rng_cone_i]
        c = 2 * _dot_xy_gpu(wi, xi, size_i)
        yi[0] = -xi[0] + c * wi[0]
        for j in range(1, size_i):
            yi[j] = xi[j] + c * wi[j]
        yi *= η[i] ** 2

def mul_Hs_soc(y, x, w, η, rng_cones, n_shift, n_soc):
    threads_per_block = 128
    blocks_per_grid = (n_soc + (threads_per_block - 1)) // threads_per_block
    _kernel_mul_Hs_soc[blocks_per_grid, threads_per_block](y, x, w, η, rng_cones, n_shift, n_soc)

@cuda.jit
def _kernel_affine_ds_soc(ds, λ, rng_cones, n_linear, n_soc):
    i = cuda.grid(1)
    if i < n_soc:
        shift_i = i + n_linear
        rng_cone_i = rng_cones[shift_i]
        size_i = len(rng_cone_i)
        dsi = ds[rng_cone_i]
        λi = λ[rng_cone_i]
        dsi[0] = 0.0
        for j in range(size_i):
            dsi[0] += λi[j] * λi[j]
        λi0 = λi[0]
        for j in range(1, size_i):
            dsi[j] = 2 * λi0 * λi[j]

def affine_ds_soc(ds, λ, rng_cones, n_shift, n_soc):
    threads_per_block = 128
    blocks_per_grid = (n_soc + (threads_per_block - 1)) // threads_per_block
    _kernel_affine_ds_soc[blocks_per_grid, threads_per_block](ds, λ, rng_cones, n_shift, n_soc)

@cuda.jit
def _kernel_combined_ds_shift_soc(shift, step_z, step_s, w, η, rng_cones, n_linear, n_soc, σμ):
    i = cuda.grid(1)
    if i < n_soc:
        shift_i = i + n_linear
        rng_cone_i = rng_cones[shift_i]
        size_i = len(rng_cone_i)
        step_zi = step_z[rng_cone_i]
        step_si = step_s[rng_cone_i]
        wi = w[rng_cone_i]
        shifti = shift[rng_cone_i]
        tmp = shifti.copy()
        for j in range(size_i):
            tmp[j] = step_zi[j]
        ζ = 0.0
        for j in range(1, size_i):
            ζ += wi[j] * tmp[j]
        c = tmp[0] + ζ / (1 + wi[0])
        step_zi[0] = η[i] * (wi[0] * tmp[0] + ζ)
        for j in range(1, size_i):
            step_zi[j] = η[i] * (tmp[j] + c * wi[j])
        for j in range(size_i):
            tmp[j] = step_si[j]
        ζ = 0.0
        for j in range(1, size_i):
            ζ += wi[j] * tmp[j]
        c = -tmp[0] + ζ / (1 + wi[0])
        step_si[0] = (1 / η[i]) * (wi[0] * tmp[0] - ζ)
        for j in range(1, size_i):
            step_si[j] = (1 / η[i]) * (tmp[j] + c * wi[j])
        val = 0.0
        for j in range(size_i):
            val += step_si[j] * step_zi[j]
        shifti[0] = val - σμ
        s0 = step_si[0]
        z0 = step_zi[0]
        for j in range(1, size_i):
            shifti[j] = s0 * step_zi[j] + z0 * step_si[j]

def combined_ds_shift_soc(shift, step_z, step_s, w, η, rng_cones, n_shift, n_soc, σμ):
    threads_per_block = 128
    blocks_per_grid = (n_soc + (threads_per_block - 1)) // threads_per_block
    _kernel_combined_ds_shift_soc[blocks_per_grid, threads_per_block](shift, step_z, step_s, w, η, rng_cones, n_shift, n_soc, σμ)

@cuda.jit
def _kernel_Δs_from_Δz_offset_soc(out, ds, z, w, λ, η, rng_cones, n_shift, n_soc):
    i = cuda.grid(1)
    if i < n_soc:
        shift_i = i + n_shift
        rng_cone_i = rng_cones[shift_i]
        size_i = len(rng_cone_i)
        outi = out[rng_cone_i]
        dsi = ds[rng_cone_i]
        zi = z[rng_cone_i]
        wi = w[rng_cone_i]
        λi = λ[rng_cone_i]
        reszi = _soc_residual_gpu(zi)
        λ1ds1 = _dot_xy_gpu(λi, dsi, size_i)
        w1ds1 = _dot_xy_gpu(wi, dsi, size_i)
        _minus_vec_gpu(outi, zi)
        outi[0] = zi[0]
        c = λi[0] * dsi[0] - λ1ds1
        _multiply_gpu(outi, c / reszi)
        outi[0] += η[i] * w1ds1
        for j in range(1, size_i):
            outi[j] += η[i] * (dsi[j] + w1ds1 / (1 + wi[0]) * wi[j])
        _multiply_gpu(outi, 1 / λi[0])

def Δs_from_Δz_offset_soc(out, ds, z, w, λ, η, rng_cones, n_shift, n_soc):
    threads_per_block = 128
    blocks_per_grid = (n_soc + (threads_per_block - 1)) // threads_per_block
    _kernel_Δs_from_Δz_offset_soc[blocks_per_grid, threads_per_block](out, ds, z, w, λ, η, rng_cones, n_shift, n_soc)

@cuda.jit
def _kernel_step_length_soc(dz, ds, z, s, α, rng_cones, n_linear, n_soc):
    i = cuda.grid(1)
    if i < n_soc:
        shift_i = i + n_linear
        rng_cone_i = rng_cones[shift_i]
        si = s[rng_cone_i]
        dsi = ds[rng_cone_i]
        zi = z[rng_cone_i]
        dzi = dz[rng_cone_i]
        αz = _step_length_soc_component_gpu(zi, dzi, α[i])
        αs = _step_length_soc_component_gpu(si, dsi, α[i])
        α[i] = min(αz, αs)

def step_length_soc(dz, ds, z, s, α, αmax, rng_cones, n_shift, n_soc):
    threads_per_block = 128
    blocks_per_grid = (n_soc + (threads_per_block - 1)) // threads_per_block
    _kernel_step_length_soc[blocks_per_grid, threads_per_block](dz, ds, z, s, α, rng_cones, n_shift, n_soc)
    αmax = min(αmax, np.min(α[:n_soc]))
    return αmax

@cuda.jit
def _kernel_compute_barrier_soc(barrier, z, s, dz, ds, α, rng_cones, n_linear, n_soc):
    i = cuda.grid(1)
    if i < n_soc:
        shift_i = i + n_linear
        rng_cone_i = rng_cones[shift_i]
        si = s[rng_cone_i]
        dsi = ds[rng_cone_i]
        zi = z[rng_cone_i]
        dzi = dz[rng_cone_i]
        res_si = _soc_residual_shifted(si, dsi, α)
        res_zi = _soc_residual_shifted(zi, dzi, α)
        barrier[i] = -np.log(res_si * res_zi) / 2 if res_si > 0 and res_zi > 0 else np.inf

def compute_barrier_soc(barrier, z, s, dz, ds, α, rng_cones, n_linear, n_soc):
    threads_per_block = 128
    blocks_per_grid = (n_soc + (threads_per_block - 1)) // threads_per_block
    _kernel_compute_barrier_soc[blocks_per_grid, threads_per_block](barrier, z, s, dz, ds, α, rng_cones, n_linear, n_soc)
    return np.sum(barrier[:n_soc])

@njit
def _soc_residual_gpu(z):
    res = z[0] * z[0]
    for j in range(1, len(z)):
        res -= z[j] * z[j]
    return res

@njit
def _sqrt_soc_residual_gpu(z):
    res = _soc_residual_gpu(z)
    return np.sqrt(res) if res > 0 else 0.0

@njit
def _dot_xy_gpu(x, y, size):
    val = 0.0
    for j in range(size):
        val += x[j] * y[j]
    return val

@njit
def _minus_vec_gpu(y, x):
    for j in range(len(x)):
        y[j] = -x[j]

@njit
def _multiply_gpu(x, a):
    for j in range(len(x)):
        x[j] *= a

@njit
def _step_length_soc_component_gpu(x, y, αmax):
    a = _soc_residual_gpu(y)
    b = 2 * (x[0] * y[0] - _dot_xy_gpu(x, y, len(x)))
    c = max(0.0, _soc_residual_gpu(x))
    d = b * b - 4 * a * c
    if c < 0:
        return -np.inf
    if (a > 0 and b > 0) or d < 0:
        return αmax
    if a == 0:
        return αmax
    if c == 0:
        return αmax if a >= 0 else 0.0
    t = (-b - np.sqrt(d)) if b >= 0 else (-b + np.sqrt(d))
    r1 = (2 * c) / t
    r2 = t / (2 * a)
    r1 = np.finfo(np.float64).max if r1 < 0 else r1
    r2 = np.finfo(np.float64).max if r2 < 0 else r2
    return min(αmax, r1, r2)
