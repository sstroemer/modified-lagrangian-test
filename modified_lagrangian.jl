using JuMP, Random, Distributions, Intervals
import Gurobi

Solver = Gurobi


max_iter = 200
sens = 0.1  
max_step = 1000             # use this to control "overshooting"
max_step_constr = 1000      # use this to control "overshooting"

T = 1:8760
C = 1:2


function main_loop()
    # Construct "random" scenario
    Random.seed!(1234)
    wind_d = truncated(Normal(10, 3), 0, Inf)
    demand_d = truncated(Normal(30, 5), 0, Inf)

    wind_avail = [
        rand(wind_d, T[end]),
        rand(wind_d, T[end])
    ]
    demand = [
        rand(demand_d, T[end]),
        rand(demand_d, T[end])
    ]
    capacity = [
        rand([0., 3, 5, 7, 10, 25, 25, 25, 50, 50, 50, 50, 50, 50], T[end]),
        rand([0., 3, 5, 7, 10, 25, 25, 25, 50, 50, 50, 50, 50, 50], T[end])
    ]
    prices = [
        rand([40., 80, 80, 100, 100, 100, 100, 100, 100], T[end]),
        rand([40., 40, 40, 80, 80, 80, 80, 100, 100], T[end])
    ]

    # Make sure prices differ
    for t in T
        !isapprox(prices[1][t], prices[2][t]; atol=0.5) && continue
        prices[1 + Int(rand() >= 0.5)][t] += 0.5 + rand()
    end

    # ====================================================================

    # Solve original problem
    sm = setup_single_model(T, C, wind_avail, prices, capacity, demand)
    optimize!(sm)
    target_obj_value = objective_value(sm)
    println("TARGET: $(round(target_obj_value))")
    println("SOLVE TIME: $(round(solve_time(sm) * 1000., digits=2))ms")

    # ====================================================================

    total_solve_time = 0
    total_lpsens_time = 0

    models = setup_models(T, C, wind_avail, prices, capacity, demand)
    λ = ones(length(T))
    
    add_constr = Dict()
    enable_constr = false
    
    conv = [0 for t in T]
    li = [Inf]
    lo = [Inf]
    
    for i in 1:max_iter
        if length(li) >= 3 && (li[end] ≈ li[end-2] && lo[end] ≈ lo[end-2])
            if abs(lo[end] - li[end]) / lo[end] < 0.005
                println("Early stop due to oscillation.")
                break
            elseif !enable_constr
                enable_constr = true
                println("Enabling constraints due to oscillation.")
            end
        end
        
        optimize!(models[1])
        optimize!(models[2])

        total_solve_time += (solve_time(models[1]) + solve_time(models[2])) / 2.

        # Check (and update) current objective values
        objs = [objective_value(m) for m in models]
        if isapprox(sum(objs), li[end]; atol=1.0)
            if enable_constr && isapprox(li[end], lo[end]; atol=1.0) 
                println("$(i):: Done.")
                break
            elseif !enable_constr
                enable_constr = true
                println("$(i):: Enabling constraints.")
            end
        end
        push!(li, sum(objs))
        push!(lo, sum(sum(prices[c][t] * value(models[c][:p][t]) for t in T) for c in C))
    
        # Calculate exchange values
        e_1 = value.(models[1][:e])
        e_2 = value.(models[2][:e])
            
        for t in T
            # Get overlapping interval WITHOUT lpsens
            i1 = if prices[1][t] > λ[t] Interval(0, prices[1][t]) else Interval(prices[1][t], Inf) end
            i2 = if prices[2][t] > λ[t] Interval(0, prices[2][t]) else Interval(prices[2][t], Inf) end
            int = intersect(i1, i2)

            # Check distance to interval to determine convergence
            dist = min(abs(λ[t] - int.first), abs(λ[t] - int.last))
            if (isinf(int.first) && isinf(int.last)) || (dist < 2*sens)
                conv[t] += 1
            else
                conv[t] = 0
            end

            # Check if we need to delete a constraint that is not binding anymore
            if haskey(add_constr, t) 
                !(abs(e_1[t]) ≈ add_constr[t]) || !(abs(e_2[t]) ≈ add_constr[t]) && delete!(add_constr, t)
            end

            # If this time step is converged, nothing to do
            (e_1[t] ≈ e_2[t]) && continue
        
            # Update λ
            if e_2[t] > e_1[t]
                # Import > Export
                # => Increasing λ => seek next border 
                border = if !isinf(int.last) int.last elseif !isinf(int.first) int.first else error(t, "  =>  ", int) end       # todo: this doesnt work for lambda < 0
                border += sens
                λ[t] = min(border, λ[t] + max_step)
            else
                # Export > Import
                
                # Check if we need to add a new constraint
                if conv[t] >= 3
                    if sign(e_1[t] * e_2[t]) > 0 && enable_constr
                        target = min(abs(e_1[t]), abs(e_2[t]))
                        start = max(abs(e_1[t]), abs(e_2[t]))
                        add_constr[t] = max(start - max_step_constr, target)
                    end
                end
                if conv[t] < 3 || sign(e_1[t] * e_2[t]) < 0
                    # Decreasing λ => seek next border
                    border = if !isinf(int.first) int.first elseif !isinf(int.last) int.last else error("...") end
                    border -= sens
                    λ[t] = max(border, λ[t] - max_step)
                end
            end
        end

        update_model!(models[1], -1, λ, capacity, add_constr, prices[1], collect(e_2))
        update_model!(models[2], +1, λ, capacity, add_constr, prices[2], collect(e_1))
        println("$(i) done: $(round(li[end]; digits=1)) ($(round(lo[end]; digits=1)))")
    end

    println("Deviation to target: $(round((lo[end]-target_obj_value) / target_obj_value * 100; digits=4))%.")
    println("Total solve time: $(round(total_solve_time*1000, digits=2))ms.")
    println("Total lpsens time: $(round(total_lpsens_time*1000, digits=2))ms.")
    println("After a total of $(length(li))) iterations.")

    return li, lo, [target_obj_value, solve_time(sm) * 1000., total_solve_time*1000]
end

function setup_models(T, C, wind_avail, prices, capacity, demand)
    dir = [-1, 1]
    η = 0.9

    model = [Model(Solver.Optimizer) for c in C]

    for c in C
        #set_optimizer_attribute(model[c], "Method", 1)
        set_silent(model[c])

        # The variable representing flow/exchange between the zones
        @variable(model[c], e[t in T])

        # Variables for thermal/wind generation, (dis-)charging, and storage
        @variable(model[c], 0 <= p[t in T] <= 100)
        @variable(model[c], 0 <= w[t in T] <= wind_avail[c][t])
        @variable(model[c], 0 <= b_charge[t in T] <= 5)
        @variable(model[c], 0 <= b_discharge[t in T] <= 5)
        @variable(model[c], 0 <= s[t in T] <= 10)

        # Upper/lower bound of exchange
        @constraint(model[c], ex_ub[t in T], model[c][:e][t] <= capacity[1][t])
        @constraint(model[c], ex_lb[t in T], -model[c][:e][t] <= capacity[2][t])

        # Nodal balance constraint for each time step
        @constraint(
            model[c], nodal_balance[t in T],
            model[c][:p][t] + model[c][:w][t] + model[c][:b_discharge][t] - model[c][:b_charge][t] - demand[c][t] + dir[c]*model[c][:e][t] == 0
        )
        # State constraint for each time step
        @constraint(
            model[c], state[t in T],
            model[c][:s][t == T[end] ? 1 : (t+1)] == model[c][:s][t] + η*model[c][:b_charge][t] - 1/η*model[c][:b_discharge][t]
        )

        @objective(
            model[c], Min,
            sum(
                prices[c][t] * model[c][:p][t] +         # cost of electricity generation
                1.0 * dir[c] * model[c][:e][t]           # multiplier (λ[t] = 1 ∀ t)
            for t in T)
        )
    end

    return model
end

function update_model!(
    m::JuMP.Model, dir::Int, λ::Vector{Float64},
    capacity::Vector{Vector{Float64}}, add_constr::Dict,
    prices::Vector{Float64}, other_e::Vector{Float64}
    )
    for t in T
        set_objective_coefficient(m, m[:e][t], λ[t] * dir)

        cap_ub = min(capacity[1][t], get(add_constr, t, Inf))
        cap_lb = min(capacity[2][t], get(add_constr, t, Inf))

        set_normalized_rhs(m[:ex_ub][t], cap_ub)
        set_normalized_rhs(m[:ex_lb][t], cap_lb)
    end

#     @objective(m, Min, sum(prices[t] * m[:p][t] + λ[t] * dir * m[:e][t] for t in T))
end

function setup_single_model(T, C, wind_avail, prices, capacity, demand)
    dir = [-1, 1]
    η = 0.9

    model = Model(Solver.Optimizer)

    @variable(model, -capacity[2][t] <= e[t in T] <= capacity[1][t])
    
    @variable(model, 0 <= p[c in C, t in T] <= 100)
    @variable(model, 0 <= w[c in C, t in T] <= wind_avail[c][t])
    @variable(model, 0 <= b_charge[c in C, t in T] <= 5)
    @variable(model, 0 <= b_discharge[c in C, t in T] <= 5)
    @variable(model, 0 <= s[c in C, t in T] <= 10)
    
    @constraint(
        model, nodal_balance[c in C, t in T],
        model[:p][c, t] + model[:w][c, t] + model[:b_discharge][c, t] - model[:b_charge][c, t] - demand[c][t] + dir[c]*model[:e][t] == 0
    )
    @constraint(
        model, state[c in C, t in T],
        model[:s][c, t == T[end] ? 1 : (t+1)] == model[:s][c, t] + η*model[:b_charge][c, t] - 1/η*model[:b_discharge][c, t]
    )
    
    @objective(model, Min, sum(prices[c][t] * model[:p][c, t] for t in T for c in C))

    return model
end
