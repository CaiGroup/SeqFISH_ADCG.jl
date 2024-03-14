using LinearAlgebra
using SeqFISH_ADCG
using Test


sigma_lb = 1.2
sigma_ub = 2.5# 1.5

width = 20

p_true = [14.33 5.0; 16.3 4.0; 2.0 1.0; 1.0 1.0]
w_true = [1.0, 1.0]

#gblur = GaussBlur3D(sigma_lb, sigma_ub, width, width*2)
gblur = GaussBlur2D(sigma_lb, sigma_ub, width)
#gblur = GaussBlur3D(sigma_xy_lb, sigma_xy_ub, sigma_z_lb, sigma_z_ub, width, width*2, n_slices)
test_img_stack = phi(gblur, p_true)
min_weight = 0.5

result = SeqFISH_ADCG.getNextPoints(gblur, reshape(test_img_stack, length(test_img_stack)), min_weight)

@test all(abs.(result .-[5.0 14.5; 4.0  16.5; 1.2 1.2; 1.0 1.0]) .< 0.5)  #2.0210526315789474, 0.9858427030815827])) < 0.1
