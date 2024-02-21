cd(dirname(@__FILE__))
# Cooperative and Critical Phenomena.
# 2D Ising Model with null magnetic field and we take J = k = 1.

# Importing Packages
using Random, ProgressBars, StatsBase, LaTeXStrings, Plots, DataFrames, GLM, LsqFit
# Random: For generating random numbers.
# ProgressBars: To display progress meters for long-running operations.
# StatsBase: Provides statistical support functions.
# LaTeXStrings and Plots: For creating and managing plots with LaTeX labels.
# DataFrames: To handle, manipulate, and analyze data in tabular form.
# GLM: Generalized linear models for statistical analysis.
# LsqFit: For fitting models to data using least squares.

# Plots aesthetics
default(fontfamily="Computer Modern", linewidth=0.85, grid=false, xminorticks=10, yminorticks=10, dpi = 500, markersize = 3)

# -------------------------------------------------------
# General purpose functions: linear regression, plot, save and load data
# -------------------------------------------------------
"""
    linear_regression(X, Y, verbose=false)

Perform linear regression on the given data.

Parameters:
- `X`: Independent variable data.
- `Y`: Dependent variable data.
- `verbose`: A boolean indicating whether to print the regression results. Default is `false`.

Returns:
A dictionary containing the regression results including intercept, slope, their errors, and R-squared value.
"""
function linear_regression(X, Y, print_results = false)
    df = DataFrame(x = X, y = Y)
    # Perform linear regression
    linear_model = lm(@formula(y ~ x), df)

    # Extract coefficients and standard errors
    B, A = coef(linear_model)
    σ_B, σ_A = stderror(linear_model)

    # Calculate R-squared
    r_squared = r2(linear_model)

    # Return results as a dictionary
    results = Dict(
        "Intercept" => B,
        "Slope" => A,
        "Intercept_Error" => σ_B,
        "Slope_Error" => σ_A,
        "R_squared" => r_squared
    )
    print_results ? println("y = ($A pm $σ_A) x + ($B pm $σ_B); r^2 = $r_squared") : nothing

    return results
end

"""
    Plot_lr(X::Array, Y::Array, results::Dict, xlabel="x", ylabel="y", scatter_label::String="Data Points",
            fit_label::String="Linear Regression Fit", save=false, filename::String="Linear_regression.png")

Plot the linear regression fit along with the data points.

Parameters:
- `X::Array`: Array of x-values.
- `Y::Array`: Array of y-values.
- `results::Dict`: Dictionary containing the results of linear regression, including keys "Slope", "Intercept", "Slope_Error", "Intercept_Error", and "R_squared".
- `xlabel::String`: Label for the x-axis. Default is "x".
- `ylabel::String`: Label for the y-axis. Default is "y".
- `scatter_label::String`: Label for the data points in the scatter plot. Default is "Data Points".
- `fit_label::String`: Label for the linear regression fit line. Default is "Linear Regression Fit".
- `save::Bool`: A boolean indicating whether to save the plot. Default is `false`.
- `filename::String`: Name of the file to save the plot. Default is "Linear_regression.png".

Returns:
Nothing. Displays and optionally saves the plot.
"""
function Plot_lr(X::Array,Y::Array, results::Dict, xlabel="x", ylabel="y", scatter_label::String = "Data Points", fit_label::String = "Linear Regression Fit", save = false, filename::String = "Linear_regression.png")
    # Extract results
    A, B = results["Slope"], results["Intercept"]
    σ_A, σ_B = results["Slope_Error"], results["Intercept_Error"]
    r2 = results["R_squared"]

    # Plot results
    x_fit = range(start = minimum(X), stop = maximum(X), length = 1000)
    y_fit = A .* x_fit .+ B

    plot(x_fit, y_fit, label = fit_label)
    scatter!(X, Y, label = scatter_label, xlabel=LaTeXString(xlabel), ylabel=LaTeXString(ylabel))

    # Save figure
    save == true ? savefig(filename) : nothing

    # Display the plot
    display(current())
end

# -------------------------------------------------------
# Theoretical Results (Onsager)
# -------------------------------------------------------
# Thermodynamic functions
M_theo(T, T_c) = T <= T_c ? (1 - sinh(2 / T)^(-4))^(1 / 8) : 0 # Magnetization. For T < Tc, else 0
E_theo(T) = -coth(2/T) * (1 + 2 * coth(2/T)^2) # Energy
C_teor(T, T_c) = 2 / π * ((2 * (T_c - T) / T_c)^2 * (-log(abs(1 - T / T_c)) + log(T_c / 2) - (1 + π / 4))) # Specific heat. ONLY VALID NEAR Tc 

# Critical exponents
const Tc_theo = 2 / log(1 + sqrt(2)) # Critical temperature ≈ 2.269
const α_theo = 0.0 # Specific heat exponent
const β_theo = 1 / 8.0 # Magnetization exponent
const γ_theo = 7 / 4.0 # Susceptibility exponent
const ν_theo = 1.0 # Correlation length exponent

# -------------------------------------------------------
# Ising Functions
# -------------------------------------------------------
"""
    random_state(L)

Generate a random initial state for a lattice with spins.

Parameters:
- `L`: Size of the lattice.

Returns:
- `init_state`: Array representing the random initial state of spins.
"""
function random_state(L)
    N = L*L # Number of spins
    init_state = rand([-1, 1], N) # Initial state
    return init_state
end

"""
    calculate_neighbors(L::Int)

Calculate the indices of neighboring spins for each spin in a square lattice.

Parameters:
- `L::Int`: Size of the square lattice.

Returns:
- `right::Array{Int64}`: Indices of the spins to the right of each spin.
- `up::Array{Int64}`: Indices of the spins above each spin.
- `left::Array{Int64}`: Indices of the spins to the left of each spin.
- `down::Array{Int64}`: Indices of the spins below each spin.
"""
function calculate_neighbors(L::Int)
    N = L*L
    right = zeros(Int64, N); up = zeros(Int64, N); left = zeros(Int64, N); down = zeros(Int64, N)
    for ix in 1:L
        for iy in 1:L
            i = (iy - 1) * L + ix
            ix_right = ifelse(ix + 1 == L + 1, 1, ix + 1)
            right[i] = (iy - 1) * L + ix_right

            iy_up = ifelse(iy + 1 == L + 1, 1, iy + 1)
            up[i] = (iy_up - 1) * L + ix

            ix_left = ifelse(ix - 1 == 0, L, ix - 1)
            left[i] = (iy - 1) * L + ix_left

            iy_down = ifelse(iy - 1 == 0, L, iy - 1)
            down[i] = (iy_down - 1) * L + ix
        end
    end
    return right, up, left, down
end

"""
    Correlation_time(Observable)

Calculate the approximate correlation time of an observable.

Parameters:
- `Observable`: Array representing the observable.

Returns:
- `τeq::Float64`: Approximated correlation time.
"""
function Correlation_time(Observable)
    # Correlation function ρG(k) = <G(0)G(k)> - <G(0)>^2
    ρg = autocor(Observable, 1:1) # I only calculate the first value because I only need the first value for the approximated correlation time. I don't know how to calculate the actual correlation time generally. 
    
    # Approximated Correlation time (assuming exponential decay)
    τeq = ρg[1]/(1-ρg[1])
    return τeq
end

"""
    energy(state, neighbors)

Calculate the energy of the system.

Parameters:
- `state::Array{Int64}`: Array representing the state of the system.
- `neighbors::Tuple{Array{Int64}, Array{Int64}, Array{Int64}, Array{Int64}}`: Tuple containing arrays of neighboring spins.

Returns:
- `E::Float64`: Energy of the system.
"""
function energy(state, neighbors)
    N = length(state)
    right, up, left, down = neighbors
    E = -sum(state[i] * (state[right[i]] + state[up[i]] + state[left[i]] + state[down[i]]) for i in 1:N) / 2
    return E
end

"""
    metropolis(state, neighbors, MCS::Int, T::Float64)

Perform Metropolis Monte Carlo updates on the Ising model.

Parameters:
- `state::Array{Int64}`: Array representing the state of the system.
- `neighbors::Tuple{Array{Int64}, Array{Int64}, Array{Int64}, Array{Int64}}`: Tuple containing arrays of neighboring spins.
- `MCS::Int`: Number of Monte Carlo steps.
- `T::Float64`: Temperature of the system.

Returns:
- `state::Array{Int64}`: Updated state of the system.
"""
function metropolis(state, neighbors, MCS::Int, T::Float64)
    N = length(state) # Number of spins
    right, up, left, down = neighbors
    # In order to avoid computing exp(-2 * bi / T) for every bi, we compute the acceptance probability for the possible values of bi and store them in a list
    h = [exp(-2 * j / T) for j in [2,4]] # only for j = 2, 4 because for -4, -2, 0 we always accept the flip

    for _ in 1:(MCS * N) # Note: this is for MCS Monte Carlo Steps. For only one new proposal state as explained in 115 of the book, take MCS = 1.
        i = rand(1:N) # Random spin
        bi = state[i] * (state[right[i]] + state[up[i]] + state[left[i]] + state[down[i]]) # bi = si Σ_j^4 sj factor (5.57) Toral book
        if bi <= 0 # if bi <= 0, then exp(-2 * bi / T) >= 1, so min(1, exp(-2 * bi / T)) = 1; we always accept the flip
            state[i] = -state[i]
        else # h = min(1, exp(-2 * bi / T)) = exp(-2 * bi / T)
            if rand() < h[div(bi, 2)] # avoiding to calculate exp(-2 * bi / T) each time
                state[i] = -state[i]
            end
        end
    end
    return state
end

"""
    Tising(state, T::Float64, neighbors, steps::Int, MCS::Int = 1)

Perform Ising model simulation for a given temperature.

Parameters:
- `state::Array{Int64}`: Array representing the initial state of the system.
- `T::Float64`: Temperature of the system.
- `neighbors::Tuple{Array{Int64}, Array{Int64}, Array{Int64}, Array{Int64}}`: Tuple containing arrays of neighboring spins.
- `steps::Int`: Number of steps for the simulation.
- `MCS::Int`: Number of Monte Carlo steps for each temperature (default is 1).

Returns:
- `state::Array{Int64}`: Updated state of the system.
- `E::Array{Float64}`: Array containing the energy values for each step.
- `M::Array{Float64}`: Array containing the absolute magnetization values for each step.
"""
function Tising(state, T::Float64, neighbors, steps::Int, MCS::Int = 1)
    # Initialize variables
    E = zeros(steps); M = zeros(steps)

    # Calculate energy and magnetization for each spin configuration given by the Monte Carlo steps
    for i in 1:steps
        # Energy and magnetization
        E[i] = energy(state, neighbors)
        M[i] = abs(sum(state))

        # Update state
        state = metropolis(state, neighbors, MCS, T)
    end

    return state, E, M
end

"""
    Ising(init_state, Ts, therm_steps::Int, steps::Int, MCS::Int = 1, verbose = false)

Perform Ising model simulation for different temperatures.

Parameters:
- `init_state::Array{Int64}`: Array representing the initial state of the system.
- `Ts::Array{Float64}`: Array containing temperatures for the simulation.
- `therm_steps::Int`: Number of thermalization steps.
- `steps::Int`: Number of steps for the simulation.
- `MCS::Int`: Number of Monte Carlo steps for each temperature (default is 1).
- `verbose::Bool`: Verbosity flag (default is `false`).

Returns:
- `state::Array{Int64}`: Updated state of the system.
- `u::Array{Float64}`: Array containing the average energy per spin for each temperature.
- `m::Array{Float64}`: Array containing the average magnetization per spin for each temperature.
- `c::Array{Float64}`: Array containing the specific heat per spin for each temperature.
- `x::Array{Float64}`: Array containing the susceptibility per spin for each temperature.
- `δu::Array{Float64}`: Array containing the error in energy per spin for each temperature.
- `δm::Array{Float64}`: Array containing the error in magnetization per spin for each temperature.
- `δc::Array{Float64}`: Array containing the error in specific heat per spin for each temperature.
- `δx::Array{Float64}`: Array containing the error in susceptibility per spin for each temperature.
"""
function Ising(init_state, Ts, therm_steps::Int, steps::Int, MCS::Int = 1, verbose = false) # Calculation for different temperatures
    # Define system
    N = length(init_state) # Number of spins
    L = Int(sqrt(N)) # Lattice size
    neighbors = calculate_neighbors(L) 

    if verbose
        println("---------- ISING MODEL ----------")
        println("L = $L, therm_steps = $therm_steps, steps = $steps, MCS = $MCS")
    end

    # Initialize variables and errors: Energy, magnetization, specific heat and susceptibility
    ET = zeros(length(Ts)); MT = zeros(length(Ts)); CT = zeros(length(Ts)); XT = zeros(length(Ts)) 
    δET = zeros(length(Ts)); δMT = zeros(length(Ts)); δCT = zeros(length(Ts)); δXT = zeros(length(Ts)) 

    state = init_state
    # Calculate for each temperature
    for (i, T) in enumerate(Ts)

        verbose ? println("Starting Thermalization") : nothing

        # Thermalization. We do therm_steps MCS
        state = metropolis(state, neighbors, therm_steps, T)

        verbose ? println("Finish Thermalization") : nothing

        # Calculation
        verbose ? println("Starting Calculation") : nothing

        state, E, M = Tising(state, T, neighbors, steps, MCS)
        E2 = E .^ 2; M2 = M .^ 2

        # Average values and Errors
        verbose ? println("Calculate averages and errors") : nothing

        E_avg = mean(E); E_std = std(E) 
        τE = Correlation_time(E); δE = E_std * sqrt((2 * τE + 1) / steps)
        E2_std = std(E2) 
        τE2 = Correlation_time(E2); δE2 = E2_std * sqrt((2 * τE2 + 1) / steps)
        M_avg = mean(M); M_std = std(M)
        τM = Correlation_time(M); δM = M_std * sqrt((2 * τM + 1) / steps)
        M2_std = std(M2); τM2 = Correlation_time(M2) 
        δM2 = M2_std * sqrt((2 * τM2 + 1) / steps)
        C_avg = E_std^2 / T^2; δC = (δE + δE2) / T^2
        X_avg = M_std^2 / T; δX = (δM + δM2) / T

        # Save data
        ET[i] = E_avg; MT[i] = M_avg; CT[i] = C_avg; XT[i] = X_avg
        δET[i] = δE; δMT[i] = δM; δCT[i] = δC; δXT[i] = δX

        verbose ? println("Ising lattice for L = $L and T = $T is done!") : nothing
    end
    u, m, c, x = ET ./ N, MT ./ N, CT ./ N, XT ./ N # measures per spin
    δu, δm, δc, δx = δET ./ N, δMT ./ N, δCT ./ N, δXT ./ N # errors per spin

    return state, u, m, c, x, δu, δm, δc, δx
end

"""
    Ising_flex(init_state, T_values, therm_steps::Int, steps::Int, MCS1::Int = 1, MCS2::Int = 1, verbose = false)

Perform Ising model simulation for a flexible range of temperatures.

Parameters:
- `init_state::Array{Int64}`: Array representing the initial state of the system.
- `T_values::Tuple{Float64, Float64, Float64, Float64, Float64, Float64}`: Tuple containing temperature range and step values:
  - `Tmin::Float64`: Minimum temperature.
  - `Tcmin::Float64`: Minimum critical temperature.
  - `Tcmax::Float64`: Maximum critical temperature.
  - `Tmax::Float64`: Maximum temperature.
  - `ΔT1::Float64`: Step size for temperatures above and below the critical region.
  - `ΔT2::Float64`: Step size for temperatures within the critical region.
- `therm_steps::Int`: Number of thermalization steps.
- `steps::Int`: Number of steps for the simulation.
- `MCS1::Int`: Number of Monte Carlo steps for temperatures above and below the critical region (default is 1).
- `MCS2::Int`: Number of Monte Carlo steps for temperatures within the critical region (default is 1).
- `verbose::Bool`: Verbosity flag (default is `false`).

Returns:
- `u::Array{Float64}`: Array containing the average energy per spin for each temperature interval.
- `m::Array{Float64}`: Array containing the average magnetization per spin for each temperature interval.
- `c::Array{Float64}`: Array containing the specific heat per spin for each temperature interval.
- `x::Array{Float64}`: Array containing the susceptibility per spin for each temperature interval.
- `δu::Array{Float64}`: Array containing the error in energy per spin for each temperature interval.
- `δm::Array{Float64}`: Array containing the error in magnetization per spin for each temperature interval.
- `δc::Array{Float64}`: Array containing the error in specific heat per spin for each temperature interval.
- `δx::Array{Float64}`: Array containing the error in susceptibility per spin for each temperature interval.
"""
function Ising_flex(init_state, T_values, therm_steps::Int, steps::Int, MCS1::Int = 1, MCS2::Int = 1, verbose = false)
    Tmin, Tcmin, Tcmax, Tmax, ΔT1, ΔT2 = T_values

    verbose ? println("Starting Ising_flex for T in [$Tmin, $Tmax], taking the critical interval to be [Tcmin = $Tcmin, Tcmax = $Tcmax] with respective step values $ΔT1 and $ΔT2") : nothing

    # Define temperature intervals
    Ts1 = range(Tmax, stop=Tcmax + 0.001, step=-ΔT1)
    Ts2 = range(Tcmax, stop=Tcmin + 0.001, step=-ΔT2)
    Ts3 = range(Tcmin, stop=Tmin, step=-ΔT1)

    # Calculate for each temperature interval
    state1, u1, m1, c1, x1, δu1, δm1, δc1, δx1 = Ising(init_state, Ts1, therm_steps, steps, MCS1, verbose)
    state2, u2, m2, c2, x2, δu2, δm2, δc2, δx2 = Ising(state1, Ts2, therm_steps, steps, MCS2, verbose)
    _, u3, m3, c3, x3, δu3, δm3, δc3, δx3 = Ising(state2, Ts3, therm_steps, steps, MCS1, verbose)

    # Join results
    u = [u1; u2; u3]; m = [m1; m2; m3]; c = [c1; c2; c3]; x = [x1; x2; x3]
    δu = [δu1; δu2; δu3]; δm = [δm1; δm2; δm3]; δc = [δc1; δc2; δc3]; δx = [δx1; δx2; δx3]

    return u, m, c, x, δu, δm, δc, δx
end

# -------------------------------------------------------
# Calculations
# -------------------------------------------------------
# Define simulation parameters as constants
const Ls = [4, 8, 16, 32, 64]
const therm_steps = 2000
const steps = 1000 
const MCS1 = 50
const MCS2 = 500
const Tmin, Tcmin, Tcmax, Tmax, ΔT1, ΔT2 = 0.1, 2.1, 2.6, 5.0, 0.1, 0.05
const T_values = Tmin, Tcmin, Tcmax, Tmax, ΔT1, ΔT2

# For trials 
const Ls_try = [4, 5, 6, 7, 8]
const therm_steps_try = 1000
const steps_try = 500 
const MCS1_try = 1
const MCS2_try = 1
const Tmin_try, Tcmin_try, Tcmax_try, Tmax_try, ΔT1_try, ΔT2_try = 1.0, 2.1, 2.6, 5.0, 0.5, 0.1
const T_values_try = Tmin_try, Tcmin_try, Tcmax_try, Tmax_try, ΔT1_try, ΔT2_try

# Define Plots parameters
# Define a function to create a color gradient
# const viridis_colors = RGB[cgrad(:viridis)[z] for z in range(0, stop=1, length=length(Ls))]


# -------------------------------------------------------
# u, m, c, x
# -------------------------------------------------------
"""
    basic_measurements(L_values, T_values, therm_steps, steps, MCS1, MCS2)

Perform basic measurements for the Ising model.

Parameters:
- `L_values::Array{Int}`: Array of lattice sizes.
- `T_values::Tuple{Float64, Float64, Float64, Float64, Float64, Float64}`: Tuple containing temperature range and step values:
  - `Tmin::Float64`: Minimum temperature.
  - `Tcmin::Float64`: Minimum critical temperature.
  - `Tcmax::Float64`: Maximum critical temperature.
  - `Tmax::Float64`: Maximum temperature.
  - `ΔT1::Float64`: Step size for temperatures above and below the critical region.
  - `ΔT2::Float64`: Step size for temperatures within the critical region.
- `therm_steps::Int`: Number of thermalization steps.
- `steps::Int`: Number of steps for the simulation.
- `MCS1::Int`: Number of Monte Carlo steps for temperatures above and below the critical region.
- `MCS2::Int`: Number of Monte Carlo steps for temperatures within the critical region.

Returns:
- Plots the basic measurements (energy, magnetization, specific heat, susceptibility) for each lattice size and saves the plot as "umcx.png".
"""
function basic_measurements(L_values, T_values, therm_steps, steps, MCS1, MCS2)
    Tmin, Tcmin, Tcmax, Tmax, ΔT1, ΔT2 = T_values
    
    # Define temperature intervals
    Ts = [range(Tmax, stop=Tcmax + 0.001, step=-ΔT1); 
          range(Tcmax, stop=Tcmin + 0.001, step=-ΔT2); 
          range(Tcmin, stop=Tmin, step=-ΔT1)]

    # Create subplots
    plt = plot(layout=(2, 2), sharex=true)
     
    for (i, L) in enumerate(L_values)
        # Call Ising function
        init_state = random_state(L)
        u, m, c, x, δu, δm, δc, δx = Ising_flex(init_state, T_values, therm_steps, steps, MCS1, MCS2, true)

        # Energy plot
        scatter!(plt[1], Ts, u, yerror=δu)

        # Magnetization plot
        scatter!(plt[2], Ts, m, yerror=δm)
        
        # Specific heat plot
        scatter!(plt[3], Ts, c, yerror=δc)

        # Susceptibility plot
        scatter!(plt[4], Ts, x, yerror=δx, label="L = $(L)")
    end
    ylabel!(plt[1], L"u"); ylabel!(plt[2], L"m")
    ylabel!(plt[3], L"c_v"); ylabel!(plt[4], L"x")
    xlabel!(plt[3], L"T"); xlabel!(plt[4], L"T")

    vline!(plt[1], [Tc_theo], linestyle=:dash, color=:red, legend=:false)
    vline!(plt[2], [Tc_theo], linestyle=:dash, color=:red, legend=:false)
    vline!(plt[3], [Tc_theo], linestyle=:dash, color=:red, legend=:false)
    vline!(plt[4], [Tc_theo], linestyle=:dash, color=:red, label = L"T_c")

    # Display the plot
    display(plt)

    # Save the plot
    savefig(plt, "umcx.png")
end
# @time basic_measurements(Ls_try, T_values_try, therm_steps_try, steps_try, MCS1_try, MCS2_try)
# @time basic_measurements(Ls, T_values, therm_steps, steps, MCS1, MCS2)

# -------------------------------------------------------
# Critical temperature 
# -------------------------------------------------------
fourth_order_cumulant(M2_avg, M4_avg) = 1 - M4_avg / (3 * M2_avg ^ 2)

"""
    U4L(init_state, Ts, therm_steps, steps, MCS = 1, verbose = false)

Calculate the fourth-order cumulant for the Ising model.

Parameters:
- `init_state::Array{Int}`: Initial state of the system.
- `Ts::Array{Float64}`: Array of temperatures.
- `therm_steps::Int`: Number of thermalization steps.
- `steps::Int`: Number of steps for the simulation.
- `MCS::Int`: Number of Monte Carlo steps. Default is 1.
- `verbose::Bool`: Verbosity flag. Default is false.

Returns:
- `state::Array{Int}`: Final state of the system.
- `U4T::Array{Float64}`: Array of fourth-order cumulant values for each temperature.
- `δU4T::Array{Float64}`: Array of errors in the fourth-order cumulant values for each temperature.
"""
function U4L(init_state, Ts, therm_steps::Int, steps::Int, MCS::Int = 1, verbose = false)
    # Calculation for different temperatures at constant L
    # Define system
    N = length(init_state) # Number of spins
    L = Int(sqrt(N)) # Lattice size
    neighbors = calculate_neighbors(L) 

    if verbose
        println("---------- FOURTH ORDER CUMULANT ----------")
        println("L = $L, therm_steps = $therm_steps, steps = $steps, MCS = $MCS")
    end

    # Initialize variables and errors: fourth order cumulant
    U4T = zeros(length(Ts)); δU4T = zeros(length(Ts))

    state = init_state
    # Calculate for each temperature
    for (i, T) in enumerate(Ts)

        verbose ? println("Starting Thermalization") : nothing

        # Thermalization. We do therm_steps MCS
        state = metropolis(state, neighbors, therm_steps, T)

        verbose ? println("Finish Thermalization") : nothing

        # Calculation
        verbose ? println("Starting Calculation") : nothing

        state, _, M = Tising(state, T, neighbors, steps, MCS)
        M2 = M .^ 2; M4 = M .^ 4

        # Average values and Errors
        M2_avg = mean(M2); M2_std = std(M2)
        τM2 = Correlation_time(M2);  δM2 = M2_std * sqrt((2 * τM2 + 1) / steps)
        M4_avg = mean(M4); M4_std = std(M4)
        τM4 = Correlation_time(M4); δM4 = M4_std * sqrt((2 * τM4 + 1) / steps)

        U4 = fourth_order_cumulant(M2_avg, M4_avg)
        δU4 = (2/3) * (M4_avg/(M2_avg^3)) * δM2 + δM4/(3*M2_avg^2)

        # Save data
        U4T[i] = U4; δU4T[i] = δU4

        verbose ? println("Lattice for L = $L and T = $T is done!") : nothing
    end

    return state, U4T, δU4T
end

"""
    U4_flex(init_state, T_values, therm_steps, steps, MCS1 = 1, MCS2 = 1, verbose = false)

Calculate the fourth-order cumulant for the Ising model over flexible temperature intervals.

Parameters:
- `init_state::Array{Int}`: Initial state of the system.
- `T_values::Tuple{Float64, Float64, Float64, Float64, Float64, Float64}`: Tuple containing the minimum temperature (`Tmin`), the minimum critical temperature (`Tcmin`), the maximum critical temperature (`Tcmax`), the maximum temperature (`Tmax`), the step size for the first temperature interval (`ΔT1`), and the step size for the second temperature interval (`ΔT2`).
- `therm_steps::Int`: Number of thermalization steps.
- `steps::Int`: Number of steps for the simulation.
- `MCS1::Int`: Number of Monte Carlo steps for the first temperature interval. Default is 1.
- `MCS2::Int`: Number of Monte Carlo steps for the second temperature interval. Default is 1.
- `verbose::Bool`: Verbosity flag. Default is false.

Returns:
- `U4::Array{Float64}`: Array of fourth-order cumulant values for each temperature.
- `δU4::Array{Float64}`: Array of errors in the fourth-order cumulant values for each temperature.
"""
function U4_flex(init_state, T_values, therm_steps::Int, steps::Int, MCS1::Int = 1, MCS2::Int = 1, verbose = false)
    Tmin, Tcmin, Tcmax, Tmax, ΔT1, ΔT2 = T_values

    verbose ? println("Starting U4_flex for T in [$Tmin, $Tmax], taking the critical interval to be [Tcmin = $Tcmin, Tcmax = $Tcmax] with respective step values $ΔT1 and $ΔT2") : nothing

    # Define temperature intervals
    Ts1 = range(Tmax, stop=Tcmax + 0.001, step=-ΔT1)
    Ts2 = range(Tcmax, stop=Tcmin + 0.001, step=-ΔT2)
    Ts3 = range(Tcmin, stop=Tmin, step=-ΔT1)

    # Calculate for each temperature interval
    state1, U4T1, δU4T1 = U4L(init_state, Ts1, therm_steps, steps, MCS1, verbose)
    state2, U4T2, δU4T2 = U4L(state1, Ts2, therm_steps, steps, MCS2, verbose)
    _, U4T3, δU4T3 = U4L(state2, Ts3, therm_steps, steps, MCS1, verbose)

    # Join results
    U4 = [U4T1; U4T2; U4T3]; δU4 = [δU4T1; δU4T2; δU4T3]

    return U4, δU4
end

"""
Tc_measurement(L_values, T_values, therm_steps, steps, MCS1, MCS2, verbose = false)

Estimates the critical temperature T_c for the Ising model using the fourth-order cumulant.

Arguments:
- `L_values`: Array of lattice sizes.
- `T_values`: Tuple containing temperature range information (Tmin, Tcmin, Tcmax, Tmax, ΔT1, ΔT2).
- `therm_steps`: Number of thermalization steps.
- `steps`: Number of steps to calculate measurements.
- `MCS1`: Number of Monte Carlo steps for the first stage of measurements.
- `MCS2`: Number of Monte Carlo steps for the second stage of measurements.
- `verbose`: Whether to print verbose output (default: `false`).

Returns: Nothing. Displays plots of U_4 vs. T and saves the plot as an image file.
"""
function Tc_measurement(L_values, T_values, therm_steps, steps, MCS1, MCS2, verbose = false)
    Tmin, Tcmin, Tcmax, Tmax, ΔT1, ΔT2 = T_values
    Ts = [range(Tmax, stop=Tcmax + 0.001, step=-ΔT1); 
          range(Tcmax, stop=Tcmin + 0.001, step=-ΔT2); 
          range(Tcmin, stop=Tmin, step=-ΔT1)]

    Tmin_index = findfirst(Ts .<= Tcmin) # index of Tcmin
    Tmax_index = findfirst(Ts .<= Tcmax) # index of Tcmax

    Ts_new = Ts[Tmax_index:Tmin_index]

    # Check if Tmin and Tmax are found in Ts
    if Tmin_index === nothing || Tmax_index === nothing
        println("Error: Tmin or Tmax not found in Ts.")
        return
    end

    # Initialize matrices for ULS and δULS
    ULS = zeros(Float64, length(L_values), length(Ts_new))
    δULS = zeros(Float64, length(L_values), length(Ts_new))

    # Create a plot with two subplots
    plt = plot(layout=(1, 2), sharey=true)

    for (i, L) in enumerate(L_values)
        init_state = random_state(L)
        # Call U4L function
        U4, δU4 = U4_flex(init_state, T_values, therm_steps, steps, MCS1, MCS2, verbose)

        # Save data to matrices
        ULS[i, :] .= U4[Tmax_index:Tmin_index]
        δULS[i, :] .= δU4[Tmax_index:Tmin_index]

        # Plot U4L vs T with error bars in the first subplot
        scatter!(plt[1], Ts, U4, yerror=δU4, label="L = $L", legend=false)

        # Plot U4L vs T with error bars in the second subplot
        scatter!(plt[2], Ts, U4, yerror=δU4, label="L = $L")

        # Set limits for the zoomed subplot
        ylims!(plt[1], (-0.4, .8))
        xlims!(plt[2], (2, 2.6))
        ylims!(plt[2], (0.5, 0.675))
    end
    vline!(plt[1], [Tc_theo], linestyle=:dash, color=:red, label = L"T_c")
    vline!(plt[2], [Tc_theo], linestyle=:dash, color=:red, label = L"T_c")

    function find_critical_temperature(ULS, Ts, Tmin_index)
        # Find the maximum and minimum for each column
        max_values = maximum(ULS, dims=1)
        min_values = minimum(ULS, dims=1)
    
        # Create arrays of maximums and minimums
        max_array = max_values[1, :]
        min_array = min_values[1, :]
    
        # Calculate the maximum distance between arrays
        distances = abs.(max_array .- min_array)
    
        # Find the temperature corresponding to the minimum distance
        critical_temp_index = argmin(distances)
        critical_temp = Ts_new[critical_temp_index]
    
        return critical_temp
    end

    Tc = find_critical_temperature(ULS, Ts, Tmin_index)
    println("Critical temperature: $Tc")

    # Customize the plots
    xlabel!(plt[1], L"T")
    ylabel!(plt[1], L"U_{4}\,(L)")
    xlabel!(plt[2], L"T")

    # Display the plot
    display(plt)
    
    # Save fig
    savefig(plt, "Tc_measurement.png")
end

# Call the function to generate the plot
# @time Tc_measurement(Ls, (0.1, 2.07, 2.57, 5.0, 0.5, 0.1), therm_steps, steps, MCS1, MCS2, true) # we choose the temperatures to obtain Tc = 2.269


# -------------------------------------------------------
# Critical Exponents
# -------------------------------------------------------

# By definition approach
"""
critical_exp_by_def(L, Ts, therm_steps, steps, MCS)

Calculates critical exponents for the Ising model by definition using finite-size scaling.

Arguments:
- `L`: Lattice size.
- `Ts`: Array of temperatures.
- `therm_steps`: Number of thermalization steps.
- `steps`: Number of steps to calculate measurements.
- `MCS`: Number of Monte Carlo steps.

Returns: Nothing. Displays plots of logarithmic values and saves them as image files. Prints the calculated critical exponents.
"""
function critical_exp_by_def(L, Ts, therm_steps, steps, MCS)
    
    ts = abs.(Ts ./ Tc_theo .- 1) # Reduced temperature

    # Call Ising function
    init_state = random_state(L)
    _, _, m, c, x, _, δm, δc, δx = Ising(init_state, Ts, therm_steps, steps, MCS, true)

    # Logarithms
    logts = log.(ts); logm = log.(m); logc = log.(c); logx = log.(x) 

    # Calculate critical exponents
    α_results = linear_regression(logts, logc, true)
    Plot_lr(logts, logc, α_results, L"\ln|t|", L"\ln(c_v)", "Simulation Points", "Linear Regression Fit", true, "Critical_exponent_C.png")
    println("α = $(-α_results["Slope"]) pm $(α_results["Slope_Error"])")

    β_results = linear_regression(logts, logm, true)
    Plot_lr(logts, logm, β_results, L"\ln|t|", L"\ln(m)", "Simulation Points", "Linear Regression Fit", true, "Critical_exponent_M.png")
    println("β = $(β_results["Slope"]) pm $(β_results["Slope_Error"])")

    γ_results = linear_regression(logts, logx, true)
    Plot_lr(logts, logx, γ_results, L"\ln|t|", L"\ln(x)", "Simulation Points", "Linear Regression Fit", true, "Critical_exponent_X.png")
    println("γ = $(-γ_results["Slope"]) pm $(γ_results["Slope_Error"])")
end

# Call the function
# @time critical_exp_by_def(180, range(Tc_theo-0.01, stop=2.0, step=-0.01), therm_steps, steps, MCS2)

"""
nu_by_scaling(Ls, Ts, therm_steps, steps, MCS)

Calculates the correlation length critical exponent (ν) using finite-size scaling.

Arguments:
- `Ls`: Array of lattice sizes.
- `Ts`: Array of temperatures.
- `therm_steps`: Number of thermalization steps.
- `steps`: Number of steps to calculate measurements.
- `MCS`: Number of Monte Carlo steps.

Returns: Nothing. Displays a plot of logarithmic values and saves it as an image file. Prints the calculated correlation length critical exponent (ν) with its error.
"""
function nu_by_scaling(Ls, Ts, therm_steps, steps, MCS)

    xc = zeros(length(Ls)) # Critical point susceptibility
    # Call Ising function
    for (i, L) in enumerate(Ls)
        # Call Ising function
        init_state = random_state(L)
        _, _, _, _, x, _, _, _, δx = Ising(init_state, Ts, therm_steps, steps, MCS, true)
        xc[i] = maximum(x)
    end

    # Logarithms
    logLs = log.(Ls); logxc = log.(xc) 

    # Calculate critical exponents
    ν_results = linear_regression(logLs, logxc, true)
    Plot_lr(logLs, logxc, ν_results, L"\ln(L)", L"\ln(x)", "Simulation Points", "Linear Regression Fit", true, "Critical_exponent_nu.png")
    A = ν_results["Slope"]; δ_A = ν_results["Slope_Error"]
    ν = 1.75/A; δ_ν = abs(1.75/(A*A)) * δ_A
    println("ν = $(ν) pm $(δ_ν)")
end

# Call the function
# @time nu_by_scaling(Ls, range(2.5, stop=2.1, step=-0.1), therm_steps, steps, MCS2)

# -------------------------------------------------------
# Correlation Length & Time
# -------------------------------------------------------

"""
τvsT(Ls, Ts, therm_steps, steps, verbose = false)

Calculates the correlation time (τ_M) as a function of temperature (T) for different lattice sizes.

Arguments:
- `Ls`: Array of lattice sizes.
- `Ts`: Array of temperatures.
- `therm_steps`: Number of thermalization steps.
- `steps`: Number of steps to calculate measurements.
- `verbose`: A boolean indicating whether to display verbose output (default: false).

Returns: Nothing. Displays a plot of correlation time vs temperature.
"""
function τvsT(Ls, Ts, therm_steps, steps, verbose = false, save = false)

    plt = plot(xlabel = L"T", ylabel = L"τ_M")
    
    for (i, L) in enumerate(Ls)
        # Define system
        init_state = random_state(L)
        neighbors = calculate_neighbors(L) 

        if verbose
            println("---------- ISING MODEL ----------")
            println("L = $L, therm_steps = $therm_steps, steps = $steps")
        end

        # Initialize correlation time
        τT = zeros(length(Ts))

        state = init_state
        # Calculate for each temperature
        for (i, T) in enumerate(Ts)

            verbose ? println("Starting Thermalization") : nothing

            # Thermalization. We do therm_steps MCS
            state = metropolis(state, neighbors, therm_steps, T)

            verbose ? println("Finish Thermalization") : nothing

            # Calculation
            verbose ? println("Starting Calculation") : nothing

            _, _, M = Tising(state, T, neighbors, steps, 1)

            # Save data
            τT[i] = Correlation_time(M)

            verbose ? println("Ising lattice for L = $L and T = $T is done!") : nothing
        end
        scatter!(Ts, τT, label = L"L = %$L")
    end

    save ? savefig(plt, "tauT.png") : nothing
    display(plt)
end

# Call the function
# @time τvsT(Ls, range(4, stop=1, step=-0.01), therm_steps, steps, true)

"""
r1k(L)

Calculates the distance between each lattice point and the (1,1) lattice point for a square lattice with periodic boundary conditions (PBC).

Arguments:
- `L`: Size of the square lattice.

Returns:
- An array containing the distances between each lattice point and the (1,1) lattice point, considering PBC.
"""
function r1k(L)
    ds = []
    for j in 1:L
        for i in 1:L
            if i != 1 && j != 1 # If not in horizontal or vertical border, diagonal distance
                d = sqrt((i-1)^2 + (j-1)^2)
            else # If in horizontal or vertical border, minimum distance
                d_non_BC = max(abs(i-1),abs(j-1))
                d = min(d_non_BC, L-d_non_BC)
            end
            push!(ds, d)
        end
    end
    return ds
end

# @time corr_len(40, 3.0, therm_steps_try, steps_try, 1)
# L, T, therm_steps, steps, MCS

"""
corr_len(L, T, therm_steps, steps, MCS, verbose = false)

Calculates the correlation length ξ for a square lattice Ising model at a given temperature.

Arguments:
- `L`: Size of the square lattice.
- `T`: Temperature of the system.
- `therm_steps`: Number of thermalization steps.
- `steps`: Number of Monte Carlo steps.
- `MCS`: Number of Monte Carlo steps for each individual spin flip.
- `verbose`: (Optional) A boolean indicating whether to print verbose output. Defaults to `false`.

Returns:
- `ξ`: The correlation length.
- `σ_ξ`: The error associated with the correlation length estimation.
"""
function corr_len(L, T, therm_steps, steps, MCS, verbose = false)
    # Define system
    N = L*L # Number of spins
    neighbors = calculate_neighbors(L)
    init_state = random_state(L)

    # Thermalization. We do therm_steps MCS
    state = metropolis(init_state, neighbors, therm_steps, T)

    # Initialize variables
    M = zeros(steps); s1sj = zeros(steps, N)
    # Calculate energy and magnetization for each spin configuration given by the Monte Carlo steps
    for i in 1:steps
        # Energy and magnetization
        M[i] = abs(sum(state))
        s1sj[i,:] = [state[1] * state[j] for j in 1:N]

        # Update state
        state = metropolis(state, neighbors, MCS, T)
    end
    m_avg = mean(M)/N; s1sj_avg = vec(mean(s1sj, dims = 1))
    
    g_avg = s1sj_avg .- m_avg^2
    r = r1k(L)

    # Cleaning, let's average all g that have same r and create a new r vector with non repeated values in order. 

    # Create a DataFrame with r and g_avg
    df = DataFrame(r = r, g_avg = g_avg)

    # Group by r and calculate the average of g_avg for each group
    df_grouped = combine(groupby(df, :r), :g_avg => mean)

    # Extract the unique r values and the corresponding average g_avg values
    r_unique = df_grouped.r
    g_avg_mean = abs.(df_grouped.g_avg_mean) # because I get negative numbers due to numerical errors

    p = sortperm(r_unique)
    r_unique = r_unique[p]
    g_avg_mean = g_avg_mean[p]

    # Define the exponential fitting function
    exp_func(x, p) = exp.(-x / p[1])
    fit_result = curve_fit(exp_func, r_unique, g_avg_mean, [1.0])

    # Extract the fitted parameter ξ with error
    ξ = fit_result.param[1]; σ_ξ = estimate_errors(fit_result)[1]

    verbose ? println("ξ = $ξ ± $σ_ξ") : nothing

    return ξ, σ_ξ
end

"""
calculate_ξ_vs_T(L, Ts, therm_steps, steps, MCS)

Calculates the correlation length ξ for a range of temperatures and plots it against temperature.

Arguments:
- `L`: Size of the square lattice.
- `Ts`: Array of temperatures for which to calculate ξ.
- `therm_steps`: Number of thermalization steps.
- `steps`: Number of Monte Carlo steps.
- `MCS`: Number of Monte Carlo steps for each individual spin flip.

Returns: Nothing. Displays a plot of correlation xi vs T.
"""
function calculate_ξ_vs_T(L, Ts, therm_steps, steps, MCS)
    ξ_values = []
    σ_ξ_values = []

    for T in Ts
        ξ, σ_ξ = corr_len(L, T, therm_steps, steps, MCS)
        push!(ξ_values, ξ)
        push!(σ_ξ_values, σ_ξ)
    end

    plt = plot(Ts, ξ_values, yerror = σ_ξ_values, xlabel = L"T", ylabel = L"\xi", legend = false)

    # Display the plot
    display(plt)

    savefig("xi_vs_T.png")
end

# @time calculate_ξ_vs_T(32, range(3.0, stop=2.0, step=-0.05), therm_steps, steps, 1)

ξcs = zeros(Float64,length(Ls))
for (i, L) in enumerate(Ls)
    ξ_values = []
    σ_ξ_values = []
    for T in range(2.3, stop=2.1, step=-0.01)
        ξ, σ_ξ = corr_len(L, T, therm_steps, steps, 1)
        push!(ξ_values, ξ)
        push!(σ_ξ_values, σ_ξ)
    end
    ξcs[i] = maximum(ξ_values)
end
results = linear_regression(Ls, ξcs, true)
Plot_lr(Ls, ξcs, results, L"L", L"\xi_c", "Simulation Points", "Linear Regression Fit", true, "Correlation_length.png")
# scatter(Ls, ξcs, xlabel = L"L", ylabel = L"\xi_c", legend = false)


# -------------------------------------------------------
# Universal Scaling
# -------------------------------------------------------
"""
universal_scaling(Ls, Ts, therm_steps, steps, MCS)

Calculates and plots the universal scaling function for susceptibility.

Arguments:
- `Ls`: Array of lattice sizes.
- `Ts`: Array of temperatures.
- `therm_steps`: Number of thermalization steps.
- `steps`: Number of Monte Carlo steps.
- `MCS`: Number of Monte Carlo steps for each individual spin flip.

Returns: Nothing. Displays a plot of the universal scaling function.
"""
function universal_scaling(Ls, Ts, therm_steps, steps, MCS)
    # reduced temperature
    ts = (Ts ./ Tc_theo) .- 1
    χL = []; tL = []
    ν = 1.0; γ = 1.75

    # Create subplots
    plt = plot(xlabel = L"tL^{1/\nu}", ylabel = L"\chi L^{-\gamma/\nu}")

    for (i, L) in enumerate(Ls)
        # Call Ising function
        init_state = random_state(L)
        _, _, _, _, x, _, _, _, _ = Ising(init_state, Ts, therm_steps, steps, MCS, true)
        push!(χL, x .* L^(-γ/ν)); push!(tL, ts .* L^(1/ν))
    end

    # Plot
    for (i, L) in enumerate(Ls)
        scatter!(plt, tL[i], χL[i], label = L"L = %$L", markersize = 3)
    end

    # Display the plot
    display(plt)

    # Save the plot
    savefig(plt, "Universal_scaling.png")
end

# @time universal_scaling(Ls, range(4, stop=1, step=-0.01), therm_steps, steps, 500)