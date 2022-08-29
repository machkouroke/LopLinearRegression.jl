using LopLinearRegression
using Test
using Random, Distributions
using CairoMakie

include("utilities/uni_variable_generator.jl")
include("utilities/bi_variable_generator.jl")

const td = truncated(Normal(0, 1), 0.0, Inf)

@testset "Test Uni variable" begin
    x, y = generate_uni(td, 20)

    n_feature = 1
    regressor = LinearRegression(1)
    @test size(regressor.a) == (n_feature, 1)
    loss_train, accuracy = fit!(regressor, x, y)
    y_pred::Array{Float32} = regressor.a[1] .* x .+ regressor.b

    # Affichage des résultats
    fig = Figure()
    lines(fig[1, 1], loss_train, color=:red, label="Perte")
    scatter(fig[1, 2], [x...], [y...], color=:green, label="Point")
    lines!(fig[1, 2], [x...], [y_pred...], color=:blue, label="Prédiction")
    lines(fig[2, 1], accuracy, color=:black, label="accuracy")
    save("test/test_uni_variable.png", fig)
end
@testset "Test bi variable" begin
    x, y = generate_bi(td, 100)
    n_feature = 2
    regressor = LinearRegression(n_feature)
    @test size(regressor.a) == (n_feature, 1)
	loss_train, accuracy = fit!(regressor, x, y)
    # Affichage des résultats
    fig = Figure()
    lines(fig[1, 1], loss_train, color=:red, label="Perte")
    lines(fig[1, 2], accuracy, color=:black, label="accuracy")
    save("test/test_bi_variable.png", fig)
end

"Done"