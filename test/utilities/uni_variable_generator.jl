function generate_uni(distribution ,n::Int64; seed_value::Int64=100)
    x::Array{Float32} = collect(1:0.2:n)

	Random.seed!(seed_value)
	a = rand(distribution)
	b = [rand(distribution) for i in 1:size(x)[1]]
	y::Array{Float32} = a .* x .+ b
	x = reshape(x, (size(x)[1], 1))
	y = reshape(y, (size(y)[1], 1))
    return x, y
end