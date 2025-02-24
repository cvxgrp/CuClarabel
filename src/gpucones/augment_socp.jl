import cupy as cp
import numpy as np
from numba import cuda, njit
from scipy.sparse import csr_matrix, vstack

@njit
def count_soc(cone, size_soc):
    numel_cone = cone.shape[0]
    assert numel_cone > size_soc

    num_socs = 1
    numel_cone -= size_soc - 1

    while numel_cone > size_soc - 1:
        numel_cone -= size_soc - 2
        num_socs += 1

    num_socs += 1

    return num_socs, numel_cone + 1

@njit
def augment_data(At0, b0, rng_row, size_soc, num_soc, last_size, augx_idx):
    At = At0[:, rng_row]
    b = b0[rng_row]
    n, m = At.shape
    reduce_soc = size_soc - 2
    assert reduce_soc > 0

    bnew = np.empty(m + 2 * (num_soc - 1), dtype=At0.dtype)
    conenew = []

    Atnew = At[:, :1]
    bnew[0] = b[0]
    idx = 1  # complete index
    for i in range(num_soc):
        if i == num_soc - 1:
            rng = slice(idx + 1, idx + last_size)
            Atnew = np.hstack((Atnew, At[:, rng]))
            bnew[idx + 1:idx + last_size] = b[rng]
            conenew.append(last_size)
        else:
            rng = slice(idx + 1, idx + reduce_soc + 1)
            Atnew = np.hstack((Atnew, At[:, rng]))
            bnew[idx + 1:idx + reduce_soc + 1] = b[rng]
            conenew.append(size_soc)

            idx += reduce_soc
            augx_idx += 1
            Atnew = np.hstack((Atnew, csr_matrix(([-1, -1], ([augx_idx, augx_idx], [0, 1])), shape=(n, 2)).toarray()))
            bnew[idx + 1:idx + 3] = [0, 0]

    return Atnew, bnew, conenew, augx_idx

@njit
def augment_A_b_soc(cones, P, q, A, b, size_soc, num_socs, last_sizes, soc_indices, soc_starts):
    m, n = A.shape

    extra_dim = np.sum(num_socs) - len(num_socs)  # Additional dimensionality for x

    At = vstack([csr_matrix(A.T), csr_matrix((extra_dim, m))])  # May be costly, but more efficient to add rows to a SparseCSR matrix
    bnew = np.empty(m + 2 * extra_dim, dtype=A.dtype)
    conesnew = []

    Atnew = csr_matrix((n + extra_dim, 0))

    start_idx = 0
    end_idx = 0
    cone_idx = 0
    augx_idx = n  # the pointer to the auxiliary x used so far

    for i, ind in enumerate(soc_indices):
        conesnew.extend(cones[cone_idx:ind])

        numel_cone = cones[ind]

        end_idx = soc_starts[i]

        rng = slice(start_idx, end_idx)
        Atnew = vstack([Atnew, At[:, rng]])
        bnew[start_idx:end_idx] = b[rng]

        start_idx = end_idx
        end_idx += numel_cone
        rng_cone = slice(start_idx, end_idx)

        Ati, bi, conesi, augx_idx = augment_data(At, b, rng_cone, size_soc, num_socs[i], last_sizes[i], augx_idx)  # augment the current large soc

        Atnew = vstack([Atnew, csr_matrix(Ati)])
        bnew[start_idx:end_idx] = bi
        conesnew.extend(conesi)

        start_idx = end_idx
        cone_idx = ind

    if cone_idx < len(cones):
        Atnew = vstack([Atnew, At[:, start_idx:]])
        bnew[start_idx:] = b[start_idx:]
        conesnew.extend(cones[cone_idx:])

    Pnew = vstack([csr_matrix(P), csr_matrix((extra_dim, n))])
    Pnew = vstack([Pnew, csr_matrix((n + extra_dim, extra_dim))])
    return Pnew.toarray(), np.hstack([q, np.zeros(extra_dim)]), Atnew.toarray().T, bnew, conesnew

@njit
def expand_soc(cones, size_soc):
    n_large_soc = 0
    soc_indices = []
    soc_starts = []
    num_socs = []
    last_sizes = []

    cones_dim = 0
    for i, cone in enumerate(cones):
        numel_cone = cone
        if numel_cone > size_soc:
            soc_indices.append(i)
            soc_starts.append(cones_dim)

            num_soc, last_size = count_soc(cone, size_soc)
            num_socs.append(num_soc)
            last_sizes.append(last_size)
            n_large_soc += 1

        cones_dim += numel_cone

    return num_socs, last_sizes, soc_indices, soc_starts
