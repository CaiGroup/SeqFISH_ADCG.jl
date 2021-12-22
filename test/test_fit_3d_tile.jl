using LinearAlgebra
using SeqFISH_ADCG
using Test


sigma_xy_lb = 1.6
sigma_xy_ub = 1.7# 1.5
sigma_z_lb = 0.9
sigma_z_ub = 1.1

width = 20
n_slices = 10

p_true = [14.33 8.6 4.2;
          16.3 10.5 9.5;
          6.7 7.2 3.0;
          1.7 1.6 1.65;
          1.1 1.0 0.9;
          1.0 2.0 1.4]

#w_true = [1.0, 2.0, 1.4]

gblur = GaussBlur3D(sigma_xy_lb, sigma_xy_ub, sigma_z_lb, sigma_z_ub, width, n_slices)

test_img = phi(gblur, p_true)

test_img .+= 0.1*rand(length(test_img))

test_stack = reshape(test_img, width, width, n_slices)

min_weight = 0.5
final_loss_improvement = 3.0
max_iters = 200
max_cd_iters = 100

inputs = (test_stack, sigma_xy_lb, sigma_xy_ub, sigma_z_lb, sigma_z_ub, final_loss_improvement, min_weight, max_iters, max_cd_iters)
results = SeqFISH_ADCG.fit_stack(inputs)

println(results)
