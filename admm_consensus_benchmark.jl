using JuMP, Random, Distributions, Intervals
import Gurobi

Solver = Gurobi


max_iter = 100
ρ = 1

T = 1:8760
C = 1:2


function main_loop(; fair=false)
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

    # Make sure prices differ (this is not needed here, but is included to solve exactly the same problem)
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

    models = setup_models(T, C, wind_avail, prices, capacity, demand)
    λ = [ones(length(T)) for _ in models]
    z = zeros(length(T))

    if fair
        λ .*= sum(sum(prices)) / 2 / length(T)
    end

    li = [Inf]
    lo = [Inf]
    
    for i in 1:max_iter
        if length(li) >= 3 && abs(lo[end] - li[end]) / lo[end] < 0.005
            println("Early stop due to tolerance.")
            break
        end
        optimize!(models[1])
        optimize!(models[2])

        total_solve_time += (solve_time(models[1]) + solve_time(models[2])) / 2.

        # Check (and update) current objective values
        objs = [objective_value(m) for m in models]
        if isapprox(sum(objs), li[end]; atol=1.0)
            println("$(i):: Done.")
            break
        end
        push!(li, sum(objs))
        push!(lo, sum(sum(prices[c][t] * value(models[c][:p][t]) for t in T) for c in C))
    
        # Calculate exchange values
        e_1 = collect(value.(models[1][:e]))
        e_2 = collect(value.(models[2][:e]))

        z = 1/2. * ((e_1 + 1/ρ .* λ[1]) + (e_2 + 1/ρ .* λ[2]))
        λ[1] = λ[1] + ρ .* (e_1 - z)
        λ[2] = λ[2] + ρ .* (e_2 - z)

        update_model!(models[1], λ[1], capacity, prices[1], z)
        update_model!(models[2], λ[2], capacity, prices[2], z)
        println("$(i) done: $(round(li[end]; digits=1)) ($(round(lo[end]; digits=1)))")
    end

    println("Deviation to target: $(round((lo[end]-target_obj_value) / target_obj_value * 100; digits=4))%.")
    println("Total solve time: $(round(total_solve_time*1000, digits=2))ms.")
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
    m::JuMP.Model, λ::Vector{Float64},
    capacity::Vector{Vector{Float64}},
    prices::Vector{Float64}, z::Vector{Float64}
    )
    @objective(m, Min, sum(
        prices[t] * m[:p][t] +
        λ[t] * (m[:e][t] - z[t]) +
        ρ/2  * (m[:e][t] - z[t])^2
        for t in T
    ))
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
