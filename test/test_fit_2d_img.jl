# create test image
using SeqFISH_ADCG
using Test

sigma_lb = 1.2
sigma_ub = 2.5
width = 2048
tau = 10.0
final_loss_improvement = 1.0
min_weight = 0.5
max_iters = 200
max_cd_iters = 10
noise_mean = 0.0

gblur = GaussBlur2D(sigma_lb, sigma_ub, width)

p_true = [16.33 16.3 2.0 1.0]

p_true = reshape(p_true, 4, 1)

test_img = reshape(phi(gblur, p_true), 2048, 2048)

test_blank = zeros(2048, 2048)

res_final, res_records = fit_2048x2048_img_tiles(test_img,
                           sigma_lb :: Float64,
                           sigma_ub :: Float64,
                           tau :: Float64,
                           final_loss_improvement :: Float64,
                           min_weight :: Float64,
                           max_iters :: Int64,
                           max_cd_iters :: Int64,
                           noise_mean :: Float64
        )

blank_res_final, blank_res_records = fit_2048x2048_img_tiles(
                                        test_blank,
                                        sigma_lb :: Float64,
                                        sigma_ub :: Float64,
                                        tau :: Float64,
                                        final_loss_improvement :: Float64,
                                        min_weight :: Float64,
                                        max_iters :: Int64,
                                        max_cd_iters :: Int64,
                                        noise_mean :: Float64
)
println("blank_res_final: $blank_res_final")
@test all(Array(isapprox.(Matrix(res_final), p_true', atol= 0.1)))
@test (0, 4) == size(blank_res_final)
