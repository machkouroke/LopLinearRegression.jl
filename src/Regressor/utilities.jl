function f(α::Array, β::Float64, X::Array)::Array
	return X*α .+ β
end
function ∂a(X::Array, Y::Array, F)::Matrix{Float64}
	n = size(Y)[1]
	return n^-1 * X' * (F - Y)
end
function ∂b(Y::Array, F)::Float64
	n = size(Y)[1]
	return n^-1 * sum(F - Y)
end
function update(a::Matrix{Float64}, b::Float64,da::Matrix{Float64}, db::Float64, α::Float64)::Tuple{Matrix{Float64}, Float64}
	a = a .- α .* da
	b = b - α * db
	return a, b
end
function loss(Y, X, α::Array, β::Float64,)::Float64
	n = size(Y)[1]
	return (2*n)^-1 * sum((Y -  f(α, β, X)).^2)
end