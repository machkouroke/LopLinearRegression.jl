function generate_uni(size::Int64; seed_value::Int64=100)
    x::Array{Float32} = collect(1:0.2:size)
	d = Normal(0, 1)
	td = truncated(d, 0.0, Inf)
	Random.seed!(seed_value)
	a = [rand(td) for i in 1:size(x)[1]]
	b = [rand(td) for i in 1:size(x)[1]]
	y::Array{Float32} = a .* x .+ b
	x = reshape(x, (size(x)[1], 1))
	y = reshape(y, (size(y)[1], 1))
    return x, y
end