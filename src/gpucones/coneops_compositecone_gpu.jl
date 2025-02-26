
degree(cones::CompositeConeGPU{T}) where {T} = cones.degree
numel(cones::CompositeConeGPU{T}) where {T}  = cones.numel

# # -----------------------------------------------------
# # dispatch operators for multiple cones
# # -----------------------------------------------------

function is_symmetric(cones::CompositeConeGPU{T}) where {T}
    #true if all pieces are symmetric.  
    #determined during obj construction
    return cones._is_symmetric
end

function allows_primal_dual_scaling(cones::CompositeConeGPU{T}) where {T}
    # always true for the GPU settings
    return true                         #
end


function rectify_equilibration!(
    cones::CompositeConeGPU{T},
     δ::AbstractVector{T},
     e::AbstractVector{T}
) where{T}

    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd

    rng_cones = cones.rng_cones

    n_rec = n_soc + n_exp + n_pow + n_psd
    any_changed = (n_rec > 0) ? true : false

    #we will update e <- δ .* e using return values
    #from this function.  default is to do nothing at all
    CUDA.@sync @. δ = one(T)

    if any_changed
        n_shift = n_linear
        rectify_equilibration_gpu!(δ, e, rng_cones, n_shift, n_rec)
    end

    return any_changed
end

function margins(
    cones::CompositeConeGPU{T},
    z::AbstractVector{T},
    pd::PrimalOrDualCone,
) where {T}
    αmin = typemax(T)
    β = zero(T)

    n_linear = cones.n_linear
    n_nn  = cones.n_nn
    n_soc = cones.n_soc
    n_psd = cones.n_psd
    rng_cones = cones.rng_cones
    idx_inq = cones.idx_inq
    Z = cones.workmat1

    α = cones.α
    @. α = αmin
    
    if n_nn > 0
        (αmin,val) = margins_nonnegative(z, α, rng_cones, idx_inq, αmin)
        β += val
    end

    if n_soc > 0
        n_shift = n_linear
        (αmin,val) = margins_soc(z, α, rng_cones, n_shift, n_soc, αmin)
        β += val
    end

    if n_psd > 0
        n_shift = n_linear + n_soc
        (αmin,val) = margins_psd(Z, z, rng_cones, n_shift, n_psd, αmin)
        β += val
    end

    return (αmin,β)
end

function scaled_unit_shift!(
    cones::CompositeConeGPU{T},
    z::AbstractVector{T},
    α::T,
    pd::PrimalOrDualCone
) where {T}

    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_psd = cones.n_psd
    psd_dim = cones.psd_dim
    rng_cones = cones.rng_cones
    idx_eq = cones.idx_eq
    idx_inq = cones.idx_inq

    scaled_unit_shift_zero!(z, rng_cones, idx_eq, pd)

    scaled_unit_shift_nonnegative!(z, rng_cones, idx_inq, α)

    if n_soc > 0
        n_shift = n_linear
        scaled_unit_shift_soc!(z, rng_cones, α, n_shift, n_soc)
    end

    if n_psd > 0
        n_shift = n_linear + n_soc
        scaled_unit_shift_psd!(z, α, rng_cones, psd_dim, n_shift, n_psd)
    end

    return nothing

end

# unit initialization for asymmetric solves
function unit_initialization!(
    cones::CompositeConeGPU{T},
    z::AbstractVector{T},
    s::AbstractVector{T}
) where {T}

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

    unit_initialization_zero!(z, s, rng_cones, idx_eq)

    unit_initialization_nonnegative!(z, s, rng_cones, idx_inq)
    
    if n_soc > 0
        n_shift = n_linear
        unit_initialization_soc!(z, s, rng_cones, n_shift, n_soc)
    end

    if n_exp > 0
        n_shift = n_linear+n_soc
        unit_initialization_exp!(z, s, rng_cones, n_shift, n_exp)
    end

    if n_pow > 0
        n_shift = n_linear+n_soc+n_exp
        unit_initialization_pow!(z, s, αp, rng_cones, n_shift, n_pow)
    end

    if n_psd > 0
        n_shift = n_linear+n_soc+n_exp+n_pow
        unit_initialization_psd_gpu!(z,s,rng_cones,psd_dim,n_shift,n_psd)
    end

    return nothing

end

function set_identity_scaling!(
    cones::CompositeConeGPU{T}
) where {T}

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
    
    set_identity_scaling_nonnegative!(w, rng_cones, idx_inq)

    if n_soc > 0
        set_identity_scaling_soc!(w, η, rng_cones, n_linear, n_soc)
    end

    if n_psd > 0
        set_identity_scaling_psd!(R, Rinv, Hspsd, psd_dim, n_psd)
    end

    return nothing
end

function update_scaling!(
    cones::CompositeConeGPU{T},
    s::AbstractVector{T},
    z::AbstractVector{T},
	μ::T,
    scaling_strategy::ScalingStrategy
) where {T}

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
    
    update_scaling_nonnegative!(s, z, w, λ, rng_cones, idx_inq)

    if n_soc > 0
        n_shift = n_linear
        update_scaling_soc!(s, z, w, λ, η, rng_cones, n_shift, n_soc)
    end

    if n_exp > 0
        n_shift = n_linear+n_soc
        update_scaling_exp!(s, z, grad, Hs, H_dual, rng_cones, μ, scaling_strategy, n_shift, n_exp)
    end

    if n_pow > 0
        n_shift = n_linear+n_soc+n_exp
        update_scaling_pow!(s, z, grad, Hs, H_dual, αp, rng_cones, μ, scaling_strategy, n_shift, n_exp, n_pow)
    end

    if n_psd > 0
        n_shift = n_linear+n_soc+n_exp+n_pow
        update_scaling_psd!(cones.chol1, cones.chol2, z, s, cones.workmat1, cones.λpsd, cones.Λisqrt, cones.R, cones.Rinv, cones.Hspsd, rng_cones, n_shift, n_psd)
    end

    return is_scaling_success = true
end

# Update Hs block for each cone.
function get_Hs!(
    cones::CompositeConeGPU{T},
    Hsblocks::AbstractVector{T}
) where {T}

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

    get_Hs_zero!(Hsblocks, rng_blocks, idx_eq)

    get_Hs_nonnegative!(Hsblocks, w, rng_cones, rng_blocks, idx_inq)

    if n_soc > 0
        n_shift = n_linear
        get_Hs_soc!(Hsblocks, w, η, rng_cones, rng_blocks, n_shift, n_soc)
    end

    if n_exp > 0
        n_shift = n_linear+n_soc
        get_Hs_exp!(Hsblocks, Hs, rng_blocks, n_shift, n_exp)
    end

    if n_pow > 0
        n_shift = n_linear+n_soc+n_exp
        get_Hs_pow!(Hsblocks, Hs, rng_blocks, n_shift, n_exp, n_pow)
    end

    if n_psd > 0
        n_shift = n_linear+n_soc+n_exp+n_pow
        get_Hs_psd!(Hsblocks, Hspsd, rng_blocks, n_shift, n_psd)
    end

    return nothing
end

# compute the generalized product :
# WᵀWx for symmetric cones 
# μH(s)x for symmetric cones

function mul_Hs!(
    cones::CompositeConeGPU{T},
    y::AbstractVector{T},
    x::AbstractVector{T},
    work::AbstractVector{T}
) where {T}

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

    mul_Hs_zero!(y, rng_cones, idx_eq)
    
    mul_Hs_nonnegative!(y, x, w, rng_cones, idx_inq)

    if n_soc > 0
        n_shift = n_linear
        mul_Hs_soc!(y, x, w, η, rng_cones, n_shift, n_soc)
    end

    n_nonsymmetric = n_exp + n_pow
    if n_nonsymmetric > 0
        n_shift = n_linear+n_soc
        mul_Hs_nonsymmetric!(y, Hs, x, rng_cones, n_shift, n_nonsymmetric)
    end

    if n_psd > 0
        n_shift = n_linear+n_soc+n_exp+n_pow
        mul_Hs_psd!(y, x, Hspsd, rng_cones, n_shift, n_psd, psd_dim)
    end

    return nothing
end

# x = λ ∘ λ for symmetric cone and x = s for asymmetric cones
function affine_ds!(
    cones::CompositeConeGPU{T},
    ds::AbstractVector{T},
    s::AbstractVector{T}
) where {T}

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

    affine_ds_zero!(ds, rng_cones, idx_eq)

    affine_ds_nonnegative!(ds, λ, rng_cones, idx_inq)

    if n_soc > 0
        n_shift = n_linear
        affine_ds_soc!(ds, λ, rng_cones, n_shift, n_soc)
    end

    #update nonsymmetric cones
    n_nonsymmetric = n_exp + n_pow
    if n_nonsymmetric > 0
        n_shift = n_linear + n_soc
        affine_ds_nonsymmetric!(ds, s, rng_cones, n_shift, n_nonsymmetric)
    end

    if n_psd > 0
        n_shift = n_linear + n_soc + n_exp + n_pow
        affine_ds_psd_gpu!(ds,λpsd,rng_cones,psd_dim,n_shift,n_psd)
    end

    return nothing
end

function combined_ds_shift!(
    cones::CompositeConeGPU{T},
    shift::AbstractVector{T},
    step_z::AbstractVector{T},
    step_s::AbstractVector{T},
    z::AbstractVector{T},
    σμ::T
) where {T}

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

    CUDA.synchronize()
    
    combined_ds_shift_zero!(shift, rng_cones, idx_eq)
    
    combined_ds_shift_nonnegative!(shift, step_z, step_s, w, σμ, rng_cones, idx_inq)

    if n_soc > 0
        n_shift = n_linear
        combined_ds_shift_soc!(shift, step_z, step_s, w, η, rng_cones, n_shift, n_soc, σμ)
    end

    if n_exp > 0
        n_shift = n_linear+n_soc
        combined_ds_shift_exp!(shift, step_z, step_s, z, grad, H_dual, rng_cones, σμ, n_shift, n_exp)
    end

    if n_pow > 0
        n_shift = n_linear+n_soc+n_exp
        combined_ds_shift_pow!(shift, step_z, step_s, z, grad, H_dual, αp, rng_cones, σμ, n_shift, n_exp, n_pow)
    end
    
    if n_psd > 0 
        n_shift = n_linear+n_soc+n_exp+n_pow
        combined_ds_shift_psd!(cones,shift,step_z,step_s,n_shift,n_psd,σμ)
    end

    return nothing

end

function Δs_from_Δz_offset!(
    cones::CompositeConeGPU{T},
    out::AbstractVector{T},
    ds::AbstractVector{T},
    work::AbstractVector{T},
    z::AbstractVector{T}
) where {T}

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

    Δs_from_Δz_offset_zero!(out, rng_cones, idx_eq)
    
    Δs_from_Δz_offset_nonnegative!(out, ds, z, rng_cones, idx_inq)

    if n_soc > 0
        n_shift = n_linear
        Δs_from_Δz_offset_soc!(out, ds, z, w, λ, η, rng_cones, n_shift, n_soc)
    end

    n_nonsymmetric = n_exp+n_pow
    if n_nonsymmetric > 0
        n_shift = n_linear + n_soc
        Δs_from_Δz_offset_nonsymmetric!(out, ds, rng_cones, n_shift, n_nonsymmetric)
    end

    if n_psd > 0
        n_shift = n_linear+n_soc+n_exp+n_pow
        Δs_from_Δz_offset_psd!(cones, out, ds, work, n_shift, n_psd)
    end

    return nothing
end

# maximum allowed step length over all cones
function step_length(
     cones::CompositeConeGPU{T},
        dz::AbstractVector{T},
        ds::AbstractVector{T},
         z::AbstractVector{T},
         s::AbstractVector{T},
  settings::Settings{T},
      αmax::T,
) where {T}

    n_linear = cones.n_linear
    n_soc = cones.n_soc
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    rng_cones       = cones.rng_cones
    idx_inq = cones.idx_inq
    α               = cones.α
    αp              = cones.αp
    Λisqrt = cones.Λisqrt
    d      = cones.workvec
    Rx     = cones.R
    Rinv   = cones.Rinv
    (workmat1, workmat2, workmat3) = (cones.workmat1, cones.workmat2, cones.workmat3)

    CUDA.@sync @. α = αmax          #Initialize step size

    αmax = step_length_nonnegative(dz, ds, z, s, α, αmax, rng_cones, idx_inq)

    if n_soc > 0
        n_shift = n_linear
        αmax = step_length_soc(dz, ds, z, s, α, αmax, rng_cones, n_shift, n_soc)
        if αmax < 0
            throw(DomainError("starting point of line search not in SOC"))
        end
    end

    if n_psd > 0
        n_shift = n_linear+n_soc+n_exp+n_pow
        αmax = step_length_psd(dz, ds, Λisqrt, d, Rx, Rinv, workmat1, workmat2, workmat3, αmax, rng_cones, n_shift, n_psd)
        if αmax < 0
            throw(DomainError("starting point of line search not in positive semidefinite cones"))
        end
    end

    step = settings.linesearch_backtrack_step
    αmin = settings.min_terminate_step_length
    #if we have any nonsymmetric cones, then back off from full steps slightly
    #so that centrality checks and logarithms don't fail right at the boundaries
    if(n_exp + n_pow > 0)
        αmax = min(αmax,settings.max_step_fraction)
    end
  
    if n_exp > 0
        n_shift = n_linear+n_soc
        αmax = step_length_exp(dz, ds, z, s, α, rng_cones, αmax, αmin, step, n_shift, n_exp)
        if αmax < 0
            throw(DomainError("starting point of line search not in expotential cones"))
        end
    end

    if n_pow > 0
        n_shift = n_linear+n_soc+n_exp
        αmax = step_length_pow(dz,ds,z,s,α,αp,rng_cones,αmax,αmin,step,n_shift,n_pow)
        if αmax < 0
            throw(DomainError("starting point of line search not in power cones"))
        end
    end

    return (αmax,αmax)
end

# compute the total barrier function at the point (z + α⋅dz, s + α⋅ds)
function compute_barrier(
    cones::CompositeConeGPU{T},
    z::AbstractVector{T},
    s::AbstractVector{T},
    dz::AbstractVector{T},
    ds::AbstractVector{T},
    α::T
) where {T}

    n_linear = cones.n_linear
    n_nn = cones.n_nn
    n_soc = cones.n_soc
    n_exp = cones.n_exp
    n_pow = cones.n_pow
    n_psd = cones.n_psd
    psd_dim = cones.psd_dim
    rng_cones = cones.rng_cones
    αp    = cones.αp
    workmat1 = cones.workmat1
    workvec  = cones.workvec

    barrier = zero(T)
    work = cones.α
    
    if n_nn > 0
        val = compute_barrier_nonnegative(work, z, s, dz, ds, α, rng_cones, cones.idx_inq, n_nn)
        barrier += val
    end

    if n_soc > 0
        val = compute_barrier_soc(work, z, s, dz, ds, α, rng_cones, n_linear, n_soc)
        barrier += val
    end

    if n_exp > 0
        n_shift = n_linear+n_soc
        val = compute_barrier_exp(work, z, s, dz, ds, α, rng_cones, n_shift, n_exp)
        barrier += val
    end

    if n_pow > 0
        n_shift = n_linear+n_soc+n_exp
        val = compute_barrier_pow(work, z, s, dz, ds, α, αp, rng_cones, n_shift, n_pow)
        barrier += val
    end

    if n_psd > 0
        n_shift = n_linear+n_soc+n_exp+n_pow
        val = compute_barrier_psd(work, z, s, dz, ds, α, workmat1, workvec, rng_cones, psd_dim, n_shift, n_psd)
        barrier += val
    end

    return barrier
end

