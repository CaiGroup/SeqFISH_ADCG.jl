using LinearAlgebra
using SeqFISH_ADCG

sigma_lb = 1.0
sigma_ub = 2.5
width = 32

sigma_xy_lb = 1.0
sigma_xy_ub = 3.0
sigma_z_lb = 0.5
sigma_z_ub = 2.0
width = 32
n_slices = 15

p_true = Matrix([16.0 16.0 7.0 1.4 1.1;]')
w_true = [1.0]

p_trial = [15.0 15.0 6.0 1.0 1.0]
w_trial = [1.0]

gblur = GaussBlur2D(sigma_lb, sigma_ub, width, width*2)
test_img = phi(gblur, p_true, w_true)
residuals = .- test_img

 #localDescent(s :: BoxConstrainedDifferentiableModel, lossFn :: Loss, thetas ::Matrix{Float64}, w :: Vector{Float64}, y :: Vector{Float64})
res = SparseInverseProblems.localDescent(gblur, LSLoss(), trial_point, w_trial, test_img)
