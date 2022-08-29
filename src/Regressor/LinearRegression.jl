using Random: seed!, TaskLocalRNG
include("utilities.jl")
mutable struct LinearRegression
	a::Matrix{Float64}
	b::Float64
	α::Float64
	seed::TaskLocalRNG
	function LinearRegression(n_feature::Int64 ;α::Float64=0.01, seed_value::Int64=3)
		seed_value = seed!(seed_value)
        a = randn(seed_value, (n_feature, 1))
        b = randn(seed_value)
        new(a, b, α, seed_value)
    end
end
function gradient!(regressor::LinearRegression, X_train, y_train, iter::Int64)::Array{Float32}
	loss_train::Array{Float32} = []
    accuracy = []
    println("Start of gradient")
	for i in 1:iter
		F = f(regressor.a, regressor.b, X_train)
		da, db = ∂a(X_train, y_train, F), ∂b(y_train, F)
		regressor.a, regressor.b = update(regressor.a, regressor.b, da, db, regressor.α)
		push!(loss_train, loss(y_train, X_train, regressor.a, regressor.b))
	end
	return loss_train
end
function fit!(regressor::LinearRegression, X_train, y_train; iter::Int64=100)
	loss_train = gradient!(regressor, X_train, y_train, 100)
	return loss_train
end