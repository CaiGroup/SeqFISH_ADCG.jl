using LinearAlgebra
using SeqFISH_ADCG
using Test


sigma_xy_lb = 1.2
sigma_xy_ub = 2.5# 1.5
sigma_z_lb = 2.0
sigma_z_ub = 5.0
min_weight = 0.5

width = 20
n_slices = 10

p_true = [14.33 3; 16.3 3; 5.7 2.0; 2.0 1.0; 2.0 1.0; 1.0 0.0]
w_true = [1.0, 0.0]

#gblur = GaussBlur3D(sigma_lb, sigma_ub, width, width*2)
gblur = GaussBlur3D(sigma_xy_lb, sigma_xy_ub, sigma_z_lb, sigma_z_ub, width, n_slices)
test_img_stack = phi(gblur, p_true)#, w_true)

result = SeqFISH_ADCG.getNextPoints(gblur, reshape(test_img_stack, length(test_img_stack)), min_weight)

#@test all(result .== [14.5, 16.5, 5.5, 1.2, 2.0])
println(result)
@test all(abs.(result .- [14.5, 16.5, 5.5, 2.0, 2.0, 1.0]) .< 1.0)

