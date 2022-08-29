using LopLinearRegression
using Test
using Random, Distributions
using CairoMakie
include("utilities/uni_variable_generator.jl")
@testset "Test Uni variable" begin
    x, y = generate_uni(100)

    n_feature = 1
    regressor = LinearRegression(1)
    @test size(regressor.a) = (n_feature, 1)
    loss_train = fit!(regressor, x, y)
    y_pred::Array{Float32} = regressor.a[1] .* x .+ regressor.b

    # Affichage des résultats
    fig = Figure()
    lines(fig[1, 1], loss_train, color=:red, label="Perte")
    scatter(fig[1, 2], [x...], [y...], color=:red, label="Perte")
    lines!(fig[1, 2], [x...], [y_pred...], color=:blue, label="Perte")
    save("test/test_uni_variable.png", fig)
end
