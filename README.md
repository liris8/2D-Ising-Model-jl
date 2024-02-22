# 2D Ising Model Simulations in Julia

## Abstract
This repository contains Julia code for simulating the 2D Ising Model, a classical model in statistical mechanics. The simulations are designed to study cooperative and critical phenomena in the absence of an external magnetic field, with a focus on understanding phase transitions and critical behavior.

## Features
- **Importing Packages**: Utilizes packages like Random, ProgressBars, StatsBase, LaTeXStrings, Plots, DataFrames, GLM, and LsqFit for robust simulation and analysis.
- **General Purpose Functions**: Includes functions for linear regression, plotting data, and saving/loading results.
- **Theoretical Models**: Implements Onsager's exact solution for the 2D Ising Model to calculate thermodynamic functions and critical exponents.
- **Ising Model Functions**: Provides functions for generating random lattice states, calculating energy, performing Metropolis Monte Carlo updates, and more.
- **Simulation Execution**: Features functions for running simulations at different temperatures, calculating average energy, magnetization, specific heat, susceptibility, and their errors.
- **Critical Temperature and Exponents Estimation**: Functions for estimating the critical temperature and critical exponents using finite-size scaling and cumulant methods.
