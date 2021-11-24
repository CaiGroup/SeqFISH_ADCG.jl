using LinearAlgebra
using SeqFISH_ADCG
using Test


sigma_lb = 1.0
sigma_ub = 2.5
width = 32

sigma_xy_lb = 1.0
sigma_xy_ub = 3.0
sigma_z_lb = 0.5
sigma_z_ub = 2.0
width = 32
n_slices = 15

#p_true = Matrix([16.0 16.0 7.0 1.4 1.1;]')
#w_true = [1.0]
p_true = Matrix([16.0 16.0 7.0 1.4 1.1 1.0;]')

#p_trial = [15.0 15.0 6.0 1.0 1.0]
#w_trial = [1.0]
p_trial = [15.0 15.0 6.0 1.0 1.0 1.0]

gblur = GaussBlur3D(sigma_xy_lb, sigma_xy_ub, sigma_z_lb, sigma_z_ub, width, n_slices)
test_img_stack = phi(gblur, p_true)#, w_true)

#residuals = .- test_img

 #localDescent(s :: BoxConstrainedDifferentiableModel, lossFn :: Loss, thetas ::Matrix{Float64}, w :: Vector{Float64}, y :: Vector{Float64})
res, optf = SeqFISH_ADCG.localDescent(gblur, LSLoss(), p_trial, test_img_stack)

@test all(isapprox.(res,p_true,atol=0.2)) 