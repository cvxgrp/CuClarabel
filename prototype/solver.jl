
# -------------------------------------
# setup!
# -------------------------------------
function setup!(
    s::Solver{T},
    P::AbstractMatrix{T},
    c::Vector{T},
    A::AbstractMatrix{T},
    b::Vector{T},
    cone_types::Vector{SupportedCones},
    cone_dims::Vector{Int},
    settings::Settings{T} = Settings()
) where{T}

    cone_info   = ConeInfo(cone_types,cone_dims)

    s.settings  = settings
    s.data      = DefaultProblemData(P,c,A,b,cone_info)
    s.scalings  = DefaultScalings(cone_info)
    s.variables = DefaultVariables(s.data.n,cone_info)
    s.residuals = DefaultResiduals(s.data.n,s.data.m)
    if(settings.direct_kkt_solver == true)
        s.kktsolver = DefaultKKTSolverDirect(s.data,s.scalings)
    else
        s.kktsolver = DefaultKKTSolverIndirect(s.data,s.scalings)
    end
    s.status    = DefaultStatus()

    # work variables for assembling step direction LHS/RHS
    s.step_rhs  = DefaultVariables(s.data.n,s.scalings.cone_info)
    s.step_lhs  = DefaultVariables(s.data.n,s.scalings.cone_info)

    return nothing
end


# -------------------------------------
# solve!
# -------------------------------------
function solve!(
    s::Solver{T}
) where{T}

    #various initializations
    status_reset!(s.status)
    iter   = 0
    isdone = false

    #initial residuals and duality gap
    gap       = T(0)
    sigma     = T(0)

    #solver release info, solver config
    #problem dimensions, cone type etc
    print_header(s.status,s.settings,s.data)

    #initialize variables to some reasonable starting point
    solver_default_start!(s)

    #----------
    # main loop
    #----------
    while iter <= s.settings.max_iter

        iter += 1

        #update the residuals
        #--------------
        residuals_update!(s.residuals,s.variables,s.data)

        #calculate duality gap (scaled)
        #--------------
        μ = calc_mu(s.variables, s.residuals, s.scalings)

        #convergence check and printing
        #--------------
        isdone = check_termination!(
            s.status,s.data,s.variables,
            s.residuals,s.scalings,s.settings
        )
        print_status(s.status,s.settings)

        isdone && break

        #update the scalings
        #--------------
        scaling_update!(s.scalings,s.variables)

        #update the KKT system
        #--------------
        kkt_update!(s.kktsolver,s.scalings)

        #calculate the affine step
        #--------------
        calc_affine_step_rhs!(
            s.step_rhs, s.residuals,
            s.data, s.variables, s.scalings
        )
        kkt_solve!(
            s.kktsolver, s.step_lhs, s.step_rhs,
            s.variables, s.scalings, s.data
        )

        #calculate step length and centering parameter
        #--------------
        #PJG: Debug
        #print("\nMaking AFFINE length\n")
        α = calc_step_length(s.variables,s.step_lhs,s.scalings)
        σ = calc_centering_parameter(α)

        #calculate the combined step and length
        #--------------
        calc_combined_step_rhs!(
            s.step_rhs, s.residuals,
            s.data, s.variables, s.scalings,
            s.step_lhs, σ, μ
        )
        kkt_solve!(
            s.kktsolver, s.step_lhs, s.step_rhs,
            s.variables, s.scalings, s.data
        )

        #compute final step length and update the current iterate
        #--------------
        α = 0.99*calc_step_length(s.variables,s.step_lhs,s.scalings) #PJG: make tunable
        variables_add_step!(s.variables,s.step_lhs,α)

        #record scalar values from this iteration
        status_save_scalars(s.status,μ,α,σ,iter)

        #PJG: debug. Rescale homogenous variables
        #debug_rescale(s)

    end

    status_finalize!(s.status)
    print_footer(s.status,s.settings)

    return nothing
end


# Mehrotra heuristic
function calc_centering_parameter(α::T) where{T}

    return σ = (1-α)^3
end


function solver_default_start!(s::Solver{T}) where {T}

    #set all scalings to identity (or zero for the zero cone)
    scaling_identity!(s.scalings)
    #Refactor
    kkt_update!(s.kktsolver,s.scalings)
    #solve for primal/dual initial points via KKT
    kkt_solve_initial_point!(s.kktsolver,s.variables,s.data)
    #fix up (z,s) so that they are in the cone
    variables_shift_to_cone!(s.variables, s.scalings)

    return nothing
end
