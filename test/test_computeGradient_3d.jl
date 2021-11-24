using LinearAlgebra
using SeqFISH_ADCG

sigma_xy_lb = 1.0
sigma_xy_ub = 3.0
sigma_z_lb = 0.5
sigma_z_ub = 2.0
width = 32
n_slices = 15

#p_true = Matrix([16.0 16.0 7.0 1.4 1.1;]')
#w_true = [1.0]
p_true = Matrix([16.0 16.0 7.0 1.4 1.1 1.0;]')

#p_trial = [16.0 16.0 7.0 1.0 1.0]
#w_trial = [1.0]

p_trial = [16.0 16.0 7.0 1.0 1.0 1.0]


#gblur = GaussBlur3D(sigma_xy_lb, sigma_xy_ub, sigma_z_lb, sigma_z_ub, width, 2width, n_slices)
gblur = SeqFISH_ADCG.GaussBlur3D(sigma_xy_lb, sigma_xy_ub, sigma_z_lb, sigma_z_ub, width, n_slices)

#GaussBlur3D(sigma_xy_lb, sigma_xy_ub, width, width*2)

test_img = phi(gblur, p_true) #.+ rand(32,32,15)/1000#rand(1024)./1000

#SparseInverseProblems.

function resids(img, ps)#, ws)
    trial_output = phi(gblur, ps)#, ws)
    residuals = img .- trial_output #.- img
    residuals#./norm(residuals)
end

function trial_gradient(test_img, _trial_point)
    trial_point = Matrix{Float64}(undef, 6, 1)
    trial_point[:] = _trial_point
    residuals = resids(test_img, trial_point)
    gradient = SeqFISH_ADCG.computeGradient(gblur, trial_point, residuals)
    return gradient
end

function empPartialDeriv(img, trial_point, trial_weight, Δ)
    residuals = resids(img, trial_point .- Δ, trial_weight)
    #normed_residuals = normed_resids(img, trial_point, trial_weight)
    Δresiduals = resids(img, trial_point .+ Δ, trial_weight)
    #sum(Δnormed_residuals .-normed_residuals)/(2*norm(Δ))
    sum(Δresiduals .- residuals)/(2*norm(Δ))
end

function empGrad(img, _trial_point, _trial_weight, Δ)
    trial_point = Matrix{Float64}(undef, 5, 1)
    trial_point[:] = _trial_point
    ΔlossΔx1 = empPartialDeriv(img, trial_point, [_trial_weight], [ Δ, 0 ,0])
    ΔlossΔx2 = empPartialDeriv(img, trial_point, [_trial_weight], [0,  Δ, 0])
    ΔlossΔσ = empPartialDeriv(img, trial_point, [_trial_weight], [0, 0,  Δ])
    [ΔlossΔx1, ΔlossΔx2, ΔlossΔσ]
end

Δ = 0.00001

println("gradient: ", trial_gradient(test_img, [16.0, 16.0, 7.0, 1.4, 1.1, 1.0], 1.0))
#println("empGrad: ", empGrad(test_img, [16.0, 16.0, 1.0], 1.0, Δ))
println()

println("grad: ", trial_gradient(test_img, [14.0, 18.0, 7.0, 1.4, 1.1, 1.0], 1.0))
#println("empGrad: ", empGrad(test_img, [14.0, 18.0, 1.0], 1.0, Δ))
println()
println("grad: ", trial_gradient(test_img, [16.0, 16.0, 7.0, 1.4, 4.0, 1.0], 1.0))
#println("empGrad: ", empGrad(test_img, [16.0, 16.0, 4.0], 1.0, Δ))

println()
println("grad: ", trial_gradient(test_img, [16.0, 15.0, 7.0, 1.4, 1.1, 1.0], 1.0))
#println("empGrad: ", empGrad(test_img, [16.0, 15.0, 2.0], 1.0, Δ))

println()
println("grad: ", trial_gradient(test_img, [15.0, 16.0, 7.0, 1.4, 1.1, 1.0], 1.0))
#println("empGrad: ", empGrad(test_img, [15.0, 16.0, 2.0], 1.0, Δ))


println()
println("grad: ", trial_gradient(test_img, [16.0, 16.0, 7.0, 1.4, 1.1, 1.0], 1.0))
#println("empGrad: ", empGrad(test_img, [16.0, 16.0, 2.0], 1.0, Δ))
