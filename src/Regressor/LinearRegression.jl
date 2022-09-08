using Random: seed!, TaskLocalRNG
using Statistics
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
function gradient!(regressor::LinearRegression, X_train, y_train, iter::Int64)::Tuple{Array{Float32}, Array{Float32}}
	loss_train::Array{Float32} = []
    accuracy_train::Array{Float32} = []
    println("Start of gradient")
	for i in 1:iter
		F = f(regressor.a, regressor.b, X_train)
		da, db = ∂a(X_train, y_train, F), ∂b(y_train, F)
		regressor.a, regressor.b = update(regressor.a, regressor.b, da, db, regressor.α)
		push!(loss_train, loss(y_train, X_train, regressor.a, regressor.b))
        push!(accuracy_train, accuracy(regressor, X_train, y_train))
	end
	return loss_train, accuracy_train
end
function fit!(regressor::LinearRegression, X_train, y_train; iter::Int64=100)::Tuple{Array, Array}
	loss_train, accuracy = gradient!(regressor, X_train, y_train, iter)
	return loss_train, accuracy
end
function predict(regressor::LinearRegression, X_test)::Array{Float32}
    return f(regressor.a, regressor.b, X_test)
end
function accuracy(regressor::LinearRegression, X, y)::Float64
    predict_sum = sum((y .- predict(regressor, X)).^2)
    mean_sum = sum((y .- mean(y)).^2)
    return 1 - (predict_sum / mean_sum)
end

