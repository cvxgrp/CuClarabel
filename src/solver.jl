# -------------------------------------
# utility constructor that includes
# both object creation and setup
#--------------------------------------
function Solver(
    P::AbstractMatrix{T},
    c::AbstractVector{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    cones::Vector{<:SupportedCone},
    kwargs...
) where{T <: AbstractFloat}

    s = Solver{T}()
    setup!(s,P,c,A,b,cones,kwargs...)
    return s
end

function Solver(
    P::AbstractMatrix{T},
    c::AbstractVector{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    cones::Dict,
    kwargs...
) where{T <: AbstractFloat}
    cones = cones_from_dict(cones)

    s = Solver{T}()
    setup!(s,P,c,A,b,cones,kwargs...)
    return s
end

# given the cone-dict in scs format create an array of type Vector{SupportedCone}
function cones_from_dict(cone::Dict)
    # cones  = sizehint!(Clarabel.SupportedCone[],length(cone_specs))   #Efficient preallocation later on
	cones = Vector{Clarabel.SupportedCone}(undef, 0)

	haskey(cone, "f") && push!(cones, Clarabel.ZeroConeT(cone["f"]))
	haskey(cone, "l") && push!(cones, Clarabel.NonnegativeConeT(cone["l"]))

	# second-order cones
	if haskey(cone, "q")
		socp_dim = cone["q"]
		for dim in socp_dim
			push!(cones, Clarabel.SecondOrderConeT(dim))
		end
	end
	# sdp triangle cones
	if haskey(cone, "s")
		sdp_dim = cone["s"]
		for dim in sdp_dim
			push!(cones, Clarabel.PSDTriangleConeT(dim))
		end
	end
	# primal exponential cones
	if haskey(cone, "ep")
		for k = 1:cone["ep"]
			push!(cones, Clarabel.ExponentialConeT())
		end
	end
	# power cones
	if haskey(cone, "p")
		pow_exponents = cone["p"]
		for exponent in pow_exponents
            push!(cones, Clarabel.PowerConeT(exponent))
		end
	end
	return cones
end

# -------------------------------------
# setup!
# -------------------------------------


"""
	setup!(solver, P, q, A, b, cones, [settings])

Populates a [`Solver`](@ref) with a cost function defined by `P` and `q`, and one or more conic constraints defined by `A`, `b` and a description of a conic constraint composed of cones whose types and dimensions are specified by `cones.`

The solver will be configured to solve the following optimization problem:

```
min   1/2 x'Px + q'x
s.t.  Ax + s = b, s ∈ K
```

All data matrices must be sparse.   The matrix `P` is assumed to be symmetric and positive semidefinite, and only the upper triangular part is used.

The cone `K` is a composite cone.   To define the cone the user should provide a vector of cone specifications along
with the appropriate dimensional information.   For example, to generate a cone in the nonnegative orthant followed by
a second order cone, use:

```
cones = [Clarabel.NonnegativeConeT(dim_1),
         Clarabel.SecondOrderConeT(dim_2)]
```

If the argument 'cones' is constructed incrementally, the should should initialize it as an empty array of the supertype for all allowable cones, e.g.

```
cones = Clarabel.SupportedCone[]
push!(cones,Clarabel.NonnegativeConeT(dim_1))
...
```

The optional argument `settings` can be used to pass custom solver settings:
```julia
settings = Clarabel.Settings(verbose = true)
setup!(model, P, q, A, b, cones, settings)
```

To solve the problem, you must make a subsequent call to [`solve!`](@ref)
"""
function setup!(s,P,c,A,b,cones,settings::Settings)
    #this allows total override of settings during setup
    s.settings = settings
    setup!(s,P,c,A,b,cones)
end

function setup!(s,P,c,A,b,cones; kwargs...)
    #this allows override of individual settings during setup
    settings_populate!(s.settings, Dict(kwargs))
    setup!(s,P,c,A,b,cones)
end

# main setup function
function setup!(
    s::Solver{T},
    P::AbstractMatrix{T},
    q::AbstractVector{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    cones::Vector{<:SupportedCone},
) where{T}

    # project against cones with overly specific type, e.g. 
    # when all of the cones are NonnegativeConeT
    cones = convert(Vector{SupportedCone},cones)

    #sanity check problem dimensions
    _check_dimensions(P,q,A,b,cones)

    #make this first to create the timers
    s.info    = DefaultInfo{T}()

    @timeit s.timers "setup!" begin

        #GPU preprocess
        use_gpu = gpu_preprocess(s.settings)

        # user facing results go here  
        (m,n) = size(A)
        s.solution = DefaultSolution{T}(n,m,use_gpu)

        if use_gpu
            P,q,A,b = gpu_data_copy!(P,q,A,b)
        end
        # presolve / chordal decomposition if needed,
        # then take an internal copy of the problem data
        @timeit s.timers "presolve" begin
            s.data = use_gpu ? DefaultProblemDataGPU{T}(P,q,A,b,cones,s.settings) : DefaultProblemData{T}(P,q,A,b,cones,s.settings)
        end 
        
        s.cones  = use_gpu ? CompositeConeGPU{T}(s.data.cones, s.settings.soc_threshold) : CompositeCone{T}(s.data.cones)

        s.data.m == s.cones.numel || throw(DimensionMismatch())

        s.variables = DefaultVariables{T}(s.data.n,s.data.m,use_gpu)
        s.residuals = DefaultResiduals{T}(s.data.n,s.data.m,use_gpu)

        #equilibrate problem data immediately on setup.
        #this prevents multiple equlibrations if solve!
        #is called more than once.
        @timeit s.timers "equilibration" begin
            data_equilibrate!(s.data,s.cones,s.settings)
        end
      
        @timeit s.timers "kkt init" begin
            DefaultKKTsys = use_gpu ? DefaultKKTSystemGPU : DefaultKKTSystem
            s.kktsystem = DefaultKKTsys{T}(s.data,s.cones,s.settings)
        end

        # work variables for assembling step direction LHS/RHS
        s.step_rhs  = DefaultVariables{T}(s.data.n,s.data.m,use_gpu)
        s.step_lhs  = DefaultVariables{T}(s.data.n,s.data.m,use_gpu)

        # a saved copy of the previous iterate
        s.prev_vars = DefaultVariables{T}(s.data.n,s.data.m,use_gpu)

    end

    return s
end

# sanity check problem dimensions passed by user

function _check_dimensions(P,q,A,b,cones)

    n = length(q)
    m = length(b)
    p = sum(cone -> nvars(cone), cones; init = 0)

    m == size(A)[1] || throw(DimensionMismatch("A and b incompatible dimensions."))
    p == m          || throw(DimensionMismatch("Constraint dimensions inconsistent with size of cones."))
    n == size(A)[2] || throw(DimensionMismatch("A and q incompatible dimensions."))
    n == size(P)[1] || throw(DimensionMismatch("P and q incompatible dimensions."))
    size(P)[1] == size(P)[2] || throw(DimensionMismatch("P not square."))

end


# an enum for reporting strategy checkpointing
@enum StrategyCheckpoint begin 
    Update = 0   # Checkpoint is suggesting a new ScalingStrategy
    NoUpdate     # Checkpoint recommends no change to ScalingStrategy
    Fail         # Checkpoint found a problem but no more ScalingStrategies to try
end


# -------------------------------------
# solve!
# -------------------------------------

"""
	solve!(solver)

Computes the solution to the problem in a `Clarabel.Solver` previously defined in [`setup!`](@ref).
"""
function solve!(
    s::Solver{T}
) where{T}

    # initialization needed for first loop pass 
    iter   = 0
    σ = one(T) 
    α = zero(T)
    μ = typemax(T)

    # solver release info, solver config
    # problem dimensions, cone type etc
    @notimeit begin
        print_banner(s.settings.verbose)
        info_print_configuration(s.info,s.settings,s.data,s.cones)
        info_print_status_header(s.info,s.settings)
    end

    info_reset!(s.info,s.timers)

    @timeit s.timers "solve!" begin

        # initialize variables to some reasonable starting point
        @timeit s.timers "default start" solver_default_start!(s)

        @timeit s.timers "IP iteration" begin

        # ----------
        #  main loop
        # ----------

        # Scaling strategy
        scaling = allows_primal_dual_scaling(s.cones) ? PrimalDual::ScalingStrategy : Dual::ScalingStrategy;

        while true

            #update the residuals
            #--------------
            @timeit s.timers "residual update" residuals_update!(s.residuals,s.variables,s.data)

            #calculate duality gap (scaled)
            #--------------
            @timeit s.timers "update μ" μ = variables_calc_mu(s.variables, s.residuals, s.cones)

            # record scalar values from most recent iteration.
            # This captures μ at iteration zero.  
            info_save_scalars(s.info, μ, α, σ, iter)

            #convergence check and printing
            #--------------

            @timeit s.timers "info update" info_update!(
                s.info,s.data,s.variables,
                s.residuals,s.kktsystem,s.settings,s.timers
            )
            @notimeit info_print_status(s.info,s.settings)
            isdone = info_check_termination!(s.info,s.residuals,s.settings,iter)

            # check for termination due to slow progress and update strategy
            if isdone
                (action,scaling) = _strategy_checkpoint_insufficient_progress(s,scaling) 
                if     action ∈ [NoUpdate,Fail]; break;
                elseif action === Update; continue; 
                end
            end # allows continuation if new strategy provided

            
            #update the scalings
            #--------------
            @timeit s.timers "scale cones" begin
            is_scaling_success = variables_scale_cones!(s.variables,s.cones,μ,scaling)
            end

            # check whether variables are interior points
            (action,scaling) = _strategy_checkpoint_is_scaling_success(s,is_scaling_success,scaling)
            if action === Fail;  break;
            else ();  # we only expect NoUpdate or Fail here
            end
                

            #increment counter here because we only count
            #iterations that produce a KKT update 
            iter += 1


            #Update the KKT system and the constant parts of its solution.
            #Keep track of the success of each step that calls KKT
            #--------------

            @timeit s.timers "kkt update" begin
            is_kkt_solve_success = kkt_update!(s.kktsystem,s.data,s.cones)
            end

            #calculate the affine step
            #--------------
            @timeit s.timers "update affine step" variables_affine_step_rhs!(
                s.step_rhs, s.residuals,
                s.variables, s.cones
            )

            @timeit s.timers "kkt solve" begin
            is_kkt_solve_success = is_kkt_solve_success && 
                kkt_solve!(
                    s.kktsystem, s.step_lhs, s.step_rhs,
                    s.data, s.variables, s.cones, :affine
                )
            end

            # combined step only on affine step success 
            if is_kkt_solve_success

                #calculate step length and centering parameter
                #--------------
                @timeit s.timers "affine step size" α = solver_get_step_length(s,:affine,scaling)
                σ = _calc_centering_parameter(α)

                #make a reduced Mehrotra correction in the first iteration
                #to accommodate badly centred starting points
                m = iter > 1 ? one(T) : α;

                #calculate the combined step and length
                #--------------
                @timeit s.timers "update combined step" variables_combined_step_rhs!(
                    s.step_rhs, s.residuals,
                    s.variables, s.cones,
                    s.step_lhs, σ, μ, m
                )

                @timeit s.timers "kkt solve" begin
                is_kkt_solve_success =
                    kkt_solve!(
                        s.kktsystem, s.step_lhs, s.step_rhs,
                        s.data, s.variables, s.cones, :combined
                    )
                end

            end

            # check for numerical failure and update strategy
            (action,scaling) = _strategy_checkpoint_numerical_error(s, is_kkt_solve_success, scaling) 
            if     action === NoUpdate; ();  #just keep going 
            elseif action === Update; α = zero(T); continue; 
            elseif action === Fail;   α = zero(T); break; 
            end
    

            #compute final step length and update the current iterate
            #--------------
            @timeit s.timers "combined step size" α = solver_get_step_length(s,:combined,scaling)

            # check for undersized step and update strategy
            (action,scaling) = _strategy_checkpoint_small_step(s, α, scaling)
            if     action === NoUpdate; ();  #just keep going 
            elseif action === Update; α = zero(T); continue; 
            elseif action === Fail;   α = zero(T); break; 
            end 

            # Copy previous iterate in case the next one is a dud
            info_save_prev_iterate(s.info,s.variables,s.prev_vars)

            @timeit s.timers "final update" variables_add_step!(s.variables,s.step_lhs,α)

        end  #end while
        #----------
        #----------

        end #end IP iteration timer

    end #end solve! timer

    # Check we if actually took a final step.  If not, we need 
    # to recapture the scalars and print one last line 
    if(α == zero(T))
        info_save_scalars(s.info, μ, α, σ, iter)
        @notimeit info_print_status(s.info,s.settings)
    end 

    @timeit s.timers "post-process" begin
        #check for "almost" convergence checks and then extract solution
        info_post_process!(s.info,s.residuals,s.settings) 
        solution_post_process!(s.solution,s.data,s.variables,s.info,s.settings)
    end 
    
    #halt timers
    info_finalize!(s.info,s.timers) 
    solution_finalize!(s.solution,s.info)
    

    @notimeit info_print_footer(s.info,s.settings)

    return s.solution
end


function solver_default_start!(s::Solver{T}) where {T}

    # If there are only symmetric cones, use CVXOPT style initilization
    # Otherwise, initialize along central rays

    if (is_symmetric(s.cones))
        #set all scalings to identity (or zero for the zero cone)
        set_identity_scaling!(s.cones)
        #Refactor
        kkt_update!(s.kktsystem,s.data,s.cones)
        #solve for primal/dual initial points via KKT
        kkt_solve_initial_point!(s.kktsystem,s.variables,s.data)
        #fix up (z,s) so that they are in the cone
        variables_symmetric_initialization!(s.variables, s.cones)

    else
        #Assigns unit (z,s) and zeros the primal variables 
        variables_unit_initialization!(s.variables, s.cones)
    end

    return nothing
end


function solver_get_step_length(s::Solver{T},steptype::Symbol,scaling::ScalingStrategy) where{T}

    # step length to stay within the cones
    α = variables_calc_step_length(
        s.variables, s.step_lhs,
        s.cones, s.settings, steptype
    )

    # additional barrier function limits for asymmetric cones
    if (!is_symmetric(s.cones) && steptype == :combined && scaling == Dual::ScalingStrategy)
        αinit = α
        α = solver_backtrack_step_to_barrier(s, αinit)
        # α = solver_backtrack_step_to_centrality(s,αinit)
    end
    return α
end


# check the distance to the boundary for asymmetric cones
function solver_backtrack_step_to_barrier(
    s::Solver{T}, αinit::T
) where {T}

    step = s.settings.linesearch_backtrack_step
    α = αinit

    for j = 1:50
        barrier = variables_barrier(s.variables,s.step_lhs,α,s.cones)
        if barrier < one(T)
            return α
        else
            α = step*α   #backtrack line search
        end
    end

    return α
end

# check the distance to the boundary for asymmetric cones
function solver_backtrack_step_to_centrality(
    s::Solver{T}, αinit::T
) where {T}

    step = s.settings.linesearch_backtrack_step
    α_min = s.settings.min_terminate_step_length

    α = αinit

    for j = 1:50
        centrality = shadow_centrality_check(s.variables,s.step_lhs,α,s.cones,s.settings.neighborhood) 

        if centrality
            return α
        else
            if (α *= step) < α_min
                α = zero(T)
                break
            end
        end
    end

    return α
end

# Mehrotra heuristic
function _calc_centering_parameter(α::T) where{T}

    return σ = (1-α)^3
end



function _strategy_checkpoint_insufficient_progress(s::Solver{T},scaling::ScalingStrategy) where {T} 

    if s.info.status != INSUFFICIENT_PROGRESS
        # there is no problem, so nothing to do
        return (NoUpdate::StrategyCheckpoint, scaling)
    else 
        #recover old iterate since "insufficient progress" often 
        #involves actual degradation of results 
        info_reset_to_prev_iterate(s.info,s.variables,s.prev_vars)

        # If problem is asymmetric, we can try to continue with the dual-only strategy
        if !is_symmetric(s.cones) && (scaling == PrimalDual::ScalingStrategy)
            s.info.status = UNSOLVED
            return (Update::StrategyCheckpoint, Dual::ScalingStrategy)
        else
            return (Fail::StrategyCheckpoint, scaling)
        end
    end

end 


function _strategy_checkpoint_numerical_error(s::Solver{T}, is_kkt_solve_success::Bool, scaling::ScalingStrategy) where {T}

    # if kkt was successful, then there is nothing to do 
    if is_kkt_solve_success
        return (NoUpdate::StrategyCheckpoint, scaling)
    end
    # If problem is asymmetric, we can try to continue with the dual-only strategy
    if !is_symmetric(s.cones) && (scaling == PrimalDual::ScalingStrategy)
        return (Update::StrategyCheckpoint, Dual::ScalingStrategy)
    else
        #out of tricks.  Bail out with an error
        s.info.status = NUMERICAL_ERROR
        return (Fail::StrategyCheckpoint,scaling)
    end
end 


function _strategy_checkpoint_small_step(s::Solver{T}, α::T, scaling::ScalingStrategy) where {T}

    if !is_symmetric(s.cones) &&
        scaling == PrimalDual::ScalingStrategy && α < s.settings.min_switch_step_length
        return (Update::StrategyCheckpoint, Dual::ScalingStrategy)

    elseif α <= max(zero(T), s.settings.min_terminate_step_length)
        s.info.status = INSUFFICIENT_PROGRESS
        return (Fail::StrategyCheckpoint,scaling)

    else
        return (NoUpdate::StrategyCheckpoint,scaling)
    end 
end 

function _strategy_checkpoint_is_scaling_success(s::Solver{T}, is_scaling_success::Bool, scaling::ScalingStrategy) where {T}
    if is_scaling_success
        return (NoUpdate::StrategyCheckpoint,scaling)
    else
        s.info.status = NUMERICAL_ERROR
        return (Fail::StrategyCheckpoint,scaling)
    end
end

# printing 

function Base.show(io::IO, solver::Clarabel.Solver{T}) where {T}
    println(io, "Clarabel model with Float precision: $(T)")
end



# -------------------------------------
# getters accessing individual fields via function calls.
# this is necessary because it allows us to use multiple
# dispatch through these calls to access internal solver
# data within the MOI interface, which must also support
# the ClarabelRs wrappers
# -------------------------------------

get_solution(s::Solver{T}) where {T} = s.solution
get_info(s::Solver{T}) where {T} = s.info

#gpu preprocess, reset the static regularization
function gpu_preprocess(
    settings::Settings{T}
) where {T}

    #Reset the value of static regularization
    if settings.direct_solve_method == :cudssmixed
        @assert(T === Float64)           #
        settings.static_regularization_constant = sqrt(eps(Float32))
    end

    return (settings.direct_kkt_solver && (settings.direct_solve_method in gpu_solver_list))
end

#copy data from CPU to GPU
function gpu_data_copy!(
    P::AbstractMatrix{T},
    q::AbstractVector{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T}
) where{T}
    # If P matrix is on CPU, assumed to be upper triangular
	if isa(P, SparseMatrixCSC)
		P = SparseMatrixCSC(Symmetric(P))
	end 
    # Allow P matrix input on GPU, but should be in the full form
    return (CuSparseMatrixCSR(P), CuVector(q), CuSparseMatrixCSR(A), CuVector(b))
end
