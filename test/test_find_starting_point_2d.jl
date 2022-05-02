using LinearAlgebra
using SeqFISH_ADCG
using Test


sigma_lb = 1.2
sigma_ub = 2.5# 1.5

width = 20

p_true = [14.33 3; 16.3 3; 2.0 1.0; 1.0 0.0]
w_true = [1.0, 0.0]

#gblur = GaussBlur3D(sigma_lb, sigma_ub, width, width*2)
gblur = GaussBlur2D(sigma_lb, sigma_ub, width)
#gblur = GaussBlur3D(sigma_xy_lb, sigma_xy_ub, sigma_z_lb, sigma_z_ub, width, width*2, n_slices)
test_img_stack = phi(gblur, p_true)

result = SeqFISH_ADCG.getStartingPoint(gblur, reshape(test_img_stack, length(test_img_stack)))

@test sum(result .-[14.5, 16.5, 2.0210526315789474, 0.9858427030815827]) < 0.1
