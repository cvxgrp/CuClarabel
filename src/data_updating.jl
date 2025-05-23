
# Supported matrix and vector updating input types
# NB: this a trait in Rust, not a type. 
const MatrixProblemDataUpdate{T} = Union{
    Nothing,
    SparseMatrixCSC{T},
    AbstractVector{T},
    Iterators.Zip{Tuple{Vector{DefaultInt}, Vector{T}}}
} where {T} 

const VectorProblemDataUpdate{T} = Union{
    Nothing,
    AbstractVector{T},
    Iterators.Zip{Tuple{Vector{DefaultInt}, Vector{T}}}
} where {T} 

"""
	update_data!(solver,P,q,A,b)

Overwrites internal problem data structures in a solver object with new data, avoiding new memory 
allocations.   See [`update_P!`](@ref), [`update_q!`](@ref), [`update_A!`](@ref), [`update_b!`](@ref) 
for allowable input types.

"""

function update_data!(
    s::Solver{T},
    P::VectorProblemDataUpdate{T} ,
    q::Option{Vector{T}},
    A::MatrixProblemDataUpdate{T},
    b::VectorProblemDataUpdate{T} 
) where{T}

    update_P!(s,P)
    update_q!(s,q)
    update_A!(s,A)
    update_b!(s,b)

    return nothing
end 

function update_data!(
    s::Solver{T},
    P::CuSparseMatrix{T} ,
    q::CuVector{T},
    A::CuSparseMatrix{T},
    b::CuVector{T} 
) where{T}

    update_P!(s,P)
    update_q!(s,q)
    update_A!(s,A)
    update_b!(s,b)

    return nothing
end 

"""
	update_P!(solver,P)

Overwrites the `P` matrix data in an existing solver object.  The input `P` can be:

    - a nonempty Vector, in which case the nonzero values of the original `P` are overwritten, preserving the sparsity pattern, or

    - a SparseMatrixCSC, in which case the input must match the sparsity pattern of the upper triangular part of the original `P`.   

    - an empty vector or `nothing`, in which case no action is taken.

"""

function update_P!(
    s::Solver{T},
    data::MatrixProblemDataUpdate{T} 
) where{T}

    isnothing(data) && return
    check_data_update_allowed(s)
    d = s.data.equilibration.d
    c = s.data.equilibration.c[]
    _update_matrix(data,s.data.P,d,d,c)
    # overwrite KKT data 
    kkt_update_P!(s.kktsystem,s.data.P)

    return nothing
end 

function update_P!(
    s::Solver{T},
    P::CuSparseMatrix{T} 
) where{T}

    #YC: Internal issymmetric() function returns a wrong solution for CuMatrices.
    #    We rely on users to guarantee that the input P matrix is positive semidefinite.
    isnothing(P) && return
    !(length(P.nzVal) == length(s.data.P.nzVal)) && error("The dimension of P is incorrect! It should be a full sparse matrix.")

    check_data_update_allowed(s)
    d = s.data.equilibration.d
    _update_matrix(P,s.data.P,d,d)
    # overwrite KKT data 
    kkt_update_P!(s.kktsystem,s.data.P)

    return nothing
end 

"""
	update_A!(solver,A)

Overwrites the `A` matrix data in an existing solver object.  The input `A` can be:

    - a nonempty Vector, in which case the nonzero values of the original `A` are overwritten, preserving the sparsity pattern, or

    - a SparseMatrixCSC, in which case the input must match the sparsity pattern of the original `A`.   

    - an empty vector or `nothing`, in which case no action is taken.

"""


function update_A!(
    s::Solver{T},
    data::MatrixProblemDataUpdate{T} 
) where{T}

    isnothing(data) && return
    check_data_update_allowed(s)
    d = s.data.equilibration.d
    e = s.data.equilibration.e 
    _update_matrix(data,s.data.A,e,d,nothing)
    # overwrite KKT data 
    kkt_update_A!(s.kktsystem,s.data.A)

    return nothing
end 

function update_A!(
    s::Solver{T},
    A::CuSparseMatrix{T} 
) where{T}

    isnothing(A) && return
    check_data_update_allowed(s)
    d = s.data.equilibration.d
    e = s.data.equilibration.e 
    _update_matrix(A,s.data.A,e,d)
    At = CuSparseMatrixCSR(s.data.A')
    CUDA.@sync @. s.data.At.nzVal = At.nzVal

    # overwrite KKT data 
    kkt_update_A!(s.kktsystem,s.data.A)
    kkt_update_At!(s.kktsystem,s.data.At)
    CUDA.unsafe_free!(At)

    return nothing
end 

"""
	update_q!(solver,q)

Overwrites the `q` vector data in an existing solver object.  No action is taken if 'q' is an empty vector or `nothing`.

"""

function update_q!(
    s::Solver{T},
    data::VectorProblemDataUpdate{T} 
) where{T}

    isnothing(data) && return
    check_data_update_allowed(s)
    d = s.data.equilibration.d
    c = s.data.equilibration.c[] 
    _update_vector(data,s.data.q,d,c)

    # flush unscaled norm.   Will be recalculated during solve
    data_clear_normq!(s.data)

    return nothing
end 

"""
	update_b!(solver,b)

Overwrites the `b` vector data in an existing solver object.  No action is taken if 'b' is an empty vector or `nothing`.

"""

function update_b!(
    s::Solver{T},
    data::VectorProblemDataUpdate{T} 
) where{T}

    isnothing(data) && return
    check_data_update_allowed(s)
    e = s.data.equilibration.e     
    _update_vector(data,s.data.b,e,nothing)

    # flush unscaled norm.   Will be recalculated during solve
    data_clear_normb!(s.data)

    return nothing
end 

function check_data_update_allowed(s)
    # Fail if presolve / chordal decomp have reduced problem  
    if data_is_presolved(s.data)
        error("Data updates not allowed if presolver is active.")
    elseif data_is_chordal_decomposed(s.data)
        error("Data updates not allowed if chordal decomposition is active.")
    end
end 

function is_data_update_allowed(s)
    try
        check_data_update_allowed(s)
        return true
    catch
        return false
    end
end 

function _update_matrix(
    data::SparseMatrixCSC{T},
    M::SparseMatrixCSC{T},
    lscale::AbstractVector{T},
    rscale::AbstractVector{T},
    cscale::Union{Nothing,T},
) where{T}
    
    isequal_sparsity(data,M) || throw(DimensionMismatch("Input must match sparsity pattern of original data."))
    _update_matrix(data.nzval,M,lscale,rscale,cscale)
end

function _update_matrix(
    data::AbstractVector{T},
    M::SparseMatrixCSC{T},
    lscale::AbstractVector{T},
    rscale::AbstractVector{T},
    cscale::Union{Nothing,T},
) where{T}
    
    length(data) == 0 && return
    length(data) == nnz(M) || throw(DimensionMismatch("Input must match length of original data."))
    M.nzval .= data
    lrscale!(lscale,M,rscale)
    isnothing(cscale) || (M.nzval .*= cscale)
end

function _update_matrix(
    data::Iterators.Zip{Tuple{Vector{DefaultInt}, Vector{T}}},
    M::SparseMatrixCSC{T},
    lscale::AbstractVector{T},
    rscale::AbstractVector{T},
    cscale::Union{Nothing,T},
) where{T}
    
    for (idx,value) in data
        idx ∈ 0:nnz(M) || throw(DimensionMismatch("Input must match sparsity pattern of original data."))
        (row,col) = index_to_coord(M,idx)
        if isnothing(cscale)
            M.nzval[idx] = lscale[row] * rscale[col] * value
        else
            M.nzval[idx] = lscale[row] * rscale[col] * cscale * value
        end
    end
end

function _update_matrix(
    data::CuSparseMatrix{T},
    M::CuSparseMatrix{T},
    lscale::CuVector{T},
    rscale::CuVector{T}
) where{T}
    
    isequal_sparsity(data,M) || throw(DimensionMismatch("Input must match sparsity pattern of original data."))
    _update_matrix(data.nzVal,M,lscale,rscale)
end

function _update_matrix(
    data::CuVector{T},
    M::CuSparseMatrix{T},
    lscale::CuVector{T},
    rscale::CuVector{T}
) where{T}
    
    length(data) == 0 && return
    length(data) == nnz(M) || throw(DimensionMismatch("Input must match length of original data."))
    CUDA.@sync @. M.nzVal = data
    lrscale_gpu!(lscale,M,rscale)
end

function _update_vector(
    data::AbstractVector{T},
    v::AbstractVector{T},
    vscale::AbstractVector{T},
    cscale::Union{Nothing,T},
) where{T}
    
    length(data) == 0 && return
    length(data) == length(v) || throw(DimensionMismatch("Input must match length of original data."))
    
    if isnothing(cscale)
        v .= data.*vscale
    else
        v .= data.*vscale.*cscale
    end
end


function _update_vector(
    data::Base.Iterators.Zip{Tuple{Vector{DefaultInt}, Vector{T}}},
    v::AbstractVector{T},
    vscale::AbstractVector{T},
    cscale::Union{Nothing,T},
) where{T}
    for (idx,value) in data
        if isnothing(cscale)
            v[idx] = value*vscale[idx]
        else
            v[idx] = value*vscale[idx]*cscale
        end
    end
end

#Update for GPU vectors
function _update_vector(
    data::CuVector{T},
    v::CuVector{T},
    scale::CuVector{T}
) where{T}

    length(data) == 0 && return
    length(data) == length(v) || throw(DimensionMismatch("Input must match length of original data."))
    
    CUDA.@sync @. v= data*scale
end


