using LinearAlgebra
using SeqFISH_ADCG
using Test

sigma_xy_lb = 1.0
sigma_xy_ub = 3.0
sigma_z_lb = 0.5
sigma_z_ub = 2.0
width = 32
n_slices = 15

p_true = Matrix([16.0 16.0 7.0 1.4 1.1 1.0;]')

p_trial = [16.0 16.0 7.0 1.0 1.0 1.0]


gblur = SeqFISH_ADCG.GaussBlur3D(sigma_xy_lb, sigma_xy_ub, sigma_z_lb, sigma_z_ub, width, n_slices)


test_img = phi(gblur, p_true)

function resids(img, ps)
    trial_output = phi(gblur, ps)
    residuals = img .- trial_output
    residuals
end

function trial_gradient(test_img, _trial_point)
    trial_point = Matrix{Float64}(undef, 6, 1)
    trial_point[:] = _trial_point
    residuals = resids(test_img, trial_point)
    gradient = SeqFISH_ADCG.computeGradient(gblur, trial_point, residuals)
    return gradient
end

function empPartialDeriv(img, trial_point, Δ)
    residuals = resids(img, trial_point .- Δ)
    Δresiduals = resids(img, trial_point .+ Δ)
    sum(Δresiduals.^2 .- residuals.^2)/(2*norm(Δ))
end

function empGrad(img, _trial_point, Δ)
    trial_point = Matrix{Float64}(undef, 6, 1)
    trial_point[:] = _trial_point
    ΔlossΔx₁ = empPartialDeriv(img, trial_point, [ Δ, 0 ,0, 0, 0, 0])
    ΔlossΔx₂ = empPartialDeriv(img, trial_point, [0,  Δ, 0, 0, 0, 0])
    ΔlossΔx₃ = empPartialDeriv(img, trial_point, [0, 0,  Δ, 0, 0, 0])
    ΔlossΔσxy = empPartialDeriv(img, trial_point, [0, 0, 0, Δ, 0, 0])
    ΔlossΔσz = empPartialDeriv(img, trial_point, [0, 0, 0, 0, Δ, 0])
    ΔlossΔw = empPartialDeriv(img, trial_point, [0, 0, 0, 0, 0, Δ])
    [ΔlossΔx₁, ΔlossΔx₂, ΔlossΔx₃, ΔlossΔσxy, ΔlossΔσz, ΔlossΔw]
end

Δ = 0.001

function test_point(tpoint)
    tg = trial_gradient(test_img, tpoint)
    eg = empGrad(test_img, tpoint, Δ)
    @test isapprox(tg, eg, atol = 0.001)
end

@test all(trial_gradient(test_img, [16.0, 16.0, 7.0, 1.4, 1.1, 1.0]) .== 0)

test_point([14.0, 18.0, 7.0, 1.4, 1.1, 1.0])
test_point([16.0, 16.0, 7.0, 1.4, 4.0, 1.0])
test_point([16.0, 15.0, 7.0, 1.4, 1.1, 1.0])
test_point([15.0, 16.0, 7.0, 1.4, 1.1, 1.0])
test_point([16.0, 16.0, 7.0, 1.4, 1.1, 1.0])