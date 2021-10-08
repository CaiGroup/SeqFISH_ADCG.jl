using SparseInverseProblems

sigma_lb = 1.5
sigma_ub = 2.0
width = 20
min_weight = 0.05
final_loss_improvement = 0.01
max_iters = 200
max_cd_iters = 200
using Plots

"""
ps = [14.33 8.6;
      16.3 10.5;
      1.7 1.6;
      1.0 2.0]
"""
ps = [11.33 9.2;
      12.3 10.5;
      1.7 1.6;
      1.0 3.0]

w_true = ps[4, :]


gblur = GaussBlur2D(sigma_lb, sigma_ub, width)

test_img = phi(gblur, ps)

#test_img .+= 0.1*rand(length(test_img))

test_img = reshape(test_img, width, width)

inputs = (test_img, sigma_lb, sigma_ub, 0.0, 0.0, final_loss_improvement, min_weight, max_iters, max_cd_iters)
points = SparseInverseProblems.fit_tile(inputs)


sorted_results = sortslices(points, dims=2)
sorted_ps = sortslices(ps, dims=2)

heatmap(test_img)
scatter!(sorted_results[1,:], sorted_results[2,:])
