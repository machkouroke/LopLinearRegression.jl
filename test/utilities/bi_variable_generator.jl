function generate_bi(distribution, n::Int64; seed_value::Int64=100)
    x1::Array{Float32} = normalize(collect(1:0.2:n))
    x2::Array{Float32} = normalize(collect(1:0.2:n))
    x = [x1 x2]
    a_1 = [8
	5]
	b_1 = [rand(distribution) for i in 1:size(x)[1]]
	y = reshape(x * a_1 .+ b_1, (size(x)[1], 1))
    return x, y
end

function normalize(x::Array{Float64})::Array{Float64}
    x_min = minimum(x)
    x_max = maximum(x)
    x_norm = (x .- x_min) / (x_max - x_min)
    return x_norm
end
