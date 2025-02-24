import cupy as cp
import numpy as np
from numba import cuda, njit

@njit
def unit_initialization_exp(z, s, rng_cones, n_shift, n_exp):
    for i in range(n_exp):
        shift_i = i + n_shift
        rng_cone_i = rng_cones[shift_i]
        s[rng_cone_i[0]] = -1.051383945322714
        s[rng_cone_i[1]] = 0.556409619469370
        s[rng_cone_i[2]] = 1.258967884768947
        z[rng_cone_i] = s[rng_cone_i]

@njit
def update_Hs_exp(s, z, grad, Hs, H_dual, μ, scaling_strategy):
    if scaling_strategy == "Dual":
        Hs[:] = μ * H_dual
    else:
        use_primal_dual_scaling_exp(s, z, grad, Hs, H_dual)

@njit
def update_scaling_exp(s, z, grad, Hs, H_dual, rng_cones, μ, scaling_strategy, n_shift, n_exp):
    for i in range(n_exp):
        shift_i = i + n_shift
        rng_i = rng_cones[shift_i]
        zi = z[rng_i]
        si = s[rng_i]
        gradi = grad[i, :]
        Hsi = Hs[i, :, :]
        Hi = H_dual[i, :, :]
        update_dual_grad_H_exp(gradi, Hi, zi)
        update_Hs_exp(si, zi, gradi, Hsi, Hi, μ, scaling_strategy)

@njit
def get_Hs_exp(Hsblocks, Hs, rng_blocks, n_shift, n_exp):
    for i in range(n_exp):
        shift_i = i + n_shift
        rng_i = rng_blocks[shift_i]
        Hsblocks[rng_i] = Hs[i, :, :].flatten()

@njit
def combined_ds_shift_exp(shift, step_z, step_s, z, grad, H_dual, rng_cones, σμ, n_shift, n_exp):
    for i in range(n_exp):
        shift_i = i + n_shift
        rng_i = rng_cones[shift_i]
        Hi = H_dual[i, :, :]
        gradi = grad[i, :]
        zi = z[rng_i]
        step_si = step_s[rng_i]
        step_zi = step_z[rng_i]
        shifti = shift[rng_i]
        η = np.zeros(3)
        higher_correction_exp(Hi, zi, η, step_si, step_zi)
        shifti[:] = gradi * σμ - η

@njit
def step_length_exp(dz, ds, z, s, α, rng_cones, αmax, αmin, step, n_shift, n_exp):
    for i in range(n_exp):
        shift_i = i + n_shift
        rng_i = rng_cones[shift_i]
        dzi = dz[rng_i]
        dsi = ds[rng_i]
        zi = z[rng_i]
        si = s[rng_i]
        α[i] = backtrack_search_exp(dzi, zi, dsi, si, αmax, αmin, step)
    return min(αmax, np.min(α[:n_exp]))

@njit
def backtrack_search_exp(dzi, zi, dsi, si, αmax, αmin, step):
    α = αmax
    work = np.zeros(3)
    while True:
        work[:] = si + α * dsi
        if is_primal_feasible_exp(work):
            break
        if (α := α * step) < αmin:
            return 0.0
    while True:
        work[:] = zi + α * dzi
        if is_dual_feasible_exp(work):
            break
        if (α := α * step) < αmin:
            return 0.0
    return α

@njit
def compute_barrier_exp(barrier, z, s, dz, ds, α, rng_cones, n_shift, n_exp):
    for i in range(n_exp):
        shift_i = i + n_shift
        rng_i = rng_cones[shift_i]
        dzi = dz[rng_i]
        dsi = ds[rng_i]
        zi = z[rng_i]
        si = s[rng_i]
        cur_z = zi + α * dzi
        cur_s = si + α * dsi
        barrier_d = barrier_dual_exp(cur_z)
        barrier_p = barrier_primal_exp(cur_s)
        barrier[i] = barrier_d + barrier_p
    return np.sum(barrier[:n_exp])

@njit
def barrier_dual_exp(z):
    l = np.log(-z[2] / z[0])
    return -np.log(-z[2] * z[0]) - np.log(z[1] - z[0] - z[0] * l)

@njit
def barrier_primal_exp(s):
    ω = _wright_omega(1 - s[0] / s[1] - np.log(s[1] / s[2]))
    ω = (ω - 1) * (ω - 1) / ω
    return -np.log(ω) - 2 * np.log(s[1]) - np.log(s[2]) - 3

@njit
def is_primal_feasible_exp(s):
    if s[2] > 0 and s[1] > 0:
        res = s[1] * np.log(s[2] / s[1]) - s[0]
        if res > 0:
            return True
    return False

@njit
def is_dual_feasible_exp(z):
    if z[2] > 0 and z[0] < 0:
        res = z[1] - z[0] - z[0] * np.log(-z[2] / z[0])
        if res > 0:
            return True
    return False

@njit
def gradient_primal_exp(s):
    ω = _wright_omega(1 - s[0] / s[1] - np.log(s[1] / s[2]))
    g1 = 1 / ((ω - 1) * s[1])
    g2 = g1 + g1 * np.log(ω * s[1] / s[2]) - 1 / s[1]
    g3 = ω / ((1 - ω) * s[2])
    return np.array([g1, g2, g3])

@njit
def higher_correction_exp(H, z, η, ds, v):
    u = np.linalg.solve(H, ds)
    η[1] = 1
    η[2] = -z[0] / z[2]
    η[0] = np.log(η[2])
    ψ = z[0] * η[0] - z[0] + z[1]
    dotψu = np.dot(η, u)
    dotψv = np.dot(η, v)
    coef = ((u[0] * (v[0] / z[0] - v[2] / z[2]) + u[2] * (z[0] * v[2] / z[2] - v[0]) / z[2]) * ψ - 2 * dotψu * dotψv) / (ψ * ψ * ψ)
    η *= coef
    inv_ψ2 = 1 / (ψ * ψ)
    η[0] += (1 / ψ - 2 / z[0]) * u[0] * v[0] / (z[0] * z[0]) - u[2] * v[2] / (z[2] * z[2]) / ψ + dotψu * inv_ψ2 * (v[0] / z[0] - v[2] / z[2]) + dotψv * inv_ψ2 * (u[0] / z[0] - u[2] / z[2])
    η[2] += 2 * (z[0] / ψ - 1) * u[2] * v[2] / (z[2] * z[2] * z[2]) - (u[2] * v[0] + u[0] * v[2]) / (z[2] * z[2]) / ψ + dotψu * inv_ψ2 * (z[0] * v[2] / (z[2] * z[2]) - v[0] / z[2]) + dotψv * inv_ψ2 * (z[0] * u[2] / (z[2] * z[2]) - u[0] / z[2])
    η /= 2

@njit
def update_dual_grad_H_exp(grad, H, z):
    l = np.log(-z[2] / z[0])
    r = -z[0] * l - z[0] + z[1]
    c2 = 1 / r
    grad[0] = c2 * l - 1 / z[0]
    grad[1] = -c2
    grad[2] = (c2 * z[0] - 1) / z[2]
    H[0, 0] = ((r * r - z[0] * r + l * l * z[0] * z[0]) / (r * z[0] * z[0] * r))
    H[0, 1] = (-l / (r * r))
    H[1, 0] = H[0, 1]
    H[1, 1] = (1 / (r * r))
    H[0, 2] = ((z[1] - z[0]) / (r * r * z[2]))
    H[2, 0] = H[0, 2]
    H[1, 2] = (-z[0] / (r * r * z[2]))
    H[2, 1] = H[1, 2]
    H[2, 2] = ((r * r - z[0] * r + z[0] * z[0]) / (r * r * z[2] * z[2]))

@njit
def _wright_omega(z):
    if z < 0:
        return np.inf
    if z < 1 + np.pi:
        zm1 = z - 1
        p = zm1
        w = 1 + 0.5 * p
        p *= zm1
        w += (1 / 16.0) * p
        p *= zm1
        w -= (1 / 192.0) * p
        p *= zm1
        w -= (1 / 3072.0) * p
        p *= zm1
        w += (13 / 61440.0) * p
    else:
        logz = np.log(z)
        zinv = 1 / z
        w = z - logz
        q = logz * zinv
        w += q
        q *= zinv
        w += q * (logz / 2 - 1)
        q *= zinv
        w += q * (logz * logz / 3.0 - (3 / 2.0) * logz + 1)
    r = z - w - np.log(w)
    for _ in range(2):
        wp1 = w + 1
        t = wp1 * (wp1 + (2 * r) / 3.0)
        w *= 1 + (r / wp1) * (t - 0.5 * r) / (t - r)
        r = (2 * w * w - 8 * w - 1) / (72.0 * (wp1 * wp1 * wp1 * wp1 * wp1 * wp1)) * r * r * r * r
    return w
