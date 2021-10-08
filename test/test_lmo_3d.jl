using LinearAlgebra
using SeqFISH_ADCG

sigma_xy_lb = 1.0
sigma_xy_ub = 3.0
sigma_z_lb = 0.5
sigma_z_ub = 2.0
width = 32
n_slices = 15

p_true = Matrix([16.3 16.3 7.6 1.4 1.1;]')
w_true = [1.0]

gblur = GaussBlur3D(sigma_xy_lb, sigma_xy_ub, sigma_z_lb, sigma_z_ub, width, 2width, n_slices)

test_img = phi(gblur, p_true, w_true)
residuals = test_img
v = residuals./norm(residuals)

params, objv = lmo(gblur, residuals)

println(params)
