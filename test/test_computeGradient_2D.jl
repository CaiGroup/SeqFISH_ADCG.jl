using LinearAlgebra
using SparseInverseProblems
#using Plots
using Test
using Random

Random.seed!(2)

sigma_lb = 1.0
sigma_ub = 3.0
width = 32

p_true = [16.0 3; 16.0 3; 2.0 1.0; 1.0 0.0]
w_true = [1.0, 0.0]

gblur = GaussBlur2D(sigma_lb, sigma_ub, width)

test_img = phi(gblur, p_true) #.+ rand(1024)./1000

function resids(img, ps)
    trial_output = phi(gblur, ps)
    residuals = img .- trial_output
    residuals
end

function trial_gradient(test_img, _trial_point, _trial_weight)
    #trial_point = Matrix{Float64}(undef, 3, 1)
    trial_point = Matrix{Float64}(undef, 4, 1)
    trial_point[1:3] = _trial_point
    trial_point[4] = _trial_weight
    residuals = resids(test_img, trial_point)
    gradient = SparseInverseProblems.computeGradient(gblur, trial_point, residuals)
    return gradient
end

function empPartialDeriv(img, trial_point, trial_weight, Δ)
    #residuals = resids(img, trial_point .- Δ, trial_weight)
    #Δresiduals = resids(img, trial_point .+ Δ, trial_weight)
    residuals = resids(img, trial_point .- Δ)
    Δresiduals = resids(img, trial_point .+ Δ)
    sum(Δresiduals.^2 .- residuals.^2)/(2*norm(Δ))
end

function empGrad(img, _trial_point, _trial_weight, Δ)
    trial_point = Matrix{Float64}(undef, 4, 1)
    trial_point[1:3] = _trial_point
    trial_point[4] = _trial_weight
    ΔlossΔx1 = empPartialDeriv(img, trial_point, [_trial_weight], [ Δ, 0 ,0, 0])
    ΔlossΔx2 = empPartialDeriv(img, trial_point, [_trial_weight], [0,  Δ, 0, 0])
    ΔlossΔσ = empPartialDeriv(img, trial_point, [_trial_weight], [0, 0,  Δ, 0])
    ΔlossΔw = empPartialDeriv(img, trial_point, [_trial_weight], [0, 0, 0, Δ])
    [ΔlossΔx1, ΔlossΔx2, ΔlossΔσ, ΔlossΔw]
end

Δ = 0.01

function test_point(tpoint)
    tg = trial_gradient(test_img, tpoint, 1.0)
    eg = empGrad(test_img, tpoint, 1.0, Δ)
    @test isapprox(tg, eg, atol = 0.001)
end

@testset "Compute Gradient 2D hand chosen points" begin
    test_point([16.0, 16.0, 1.0])
    test_point([14.0, 16.0, 1.0])
    test_point([16.0, 18.0, 1.0])
    test_point([14.0, 18.0, 1.0])
    test_point([16.0, 16.0, 4.0])
    test_point([16.0, 15.0, 2.0])
    test_point([15.0, 16.0, 2.0])
    test_point([16.0, 16.0, 2.0])
end

@testset "Compute Gradient 2D random points" begin
    for i = 1:1000
        p = rand(3) .* [2.0, 2.0, 1.0] .+ [14.5, 14.5, 1.5]
        test_point(p)
    end
end
