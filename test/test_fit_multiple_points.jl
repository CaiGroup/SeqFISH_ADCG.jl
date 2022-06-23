using LinearAlgebra
using SeqFISH_ADCG
using Test
using DataFrames


sigma_lb = 1.5
sigma_ub = 2.0 # 1.5
min_weight = 0.2
final_loss_improvement = 3.0
max_iters = 200
max_cd_iters = 200


width = 20

p_true = [14.33 8.6;
          16.3 10.5;
          1.7 1.6;
          1.0 2.0]

"""
p_true = [14.33 8.6 4.2;
          16.3 10.5 7.5;
          1.7 1.6 1.65]


p_true = [14.33 8.6 4.2;
          16.3 10.5 9.5;
          1.7 1.6 1.65]
"""

w_true = [1.0, 2.0]
#w_true = [1.0, 2.0, 1.4]

function test_fit_mult_ps(ps :: Matrix, δw :: Float64 = 0.001)

    gblur = GaussBlur2D(sigma_lb, sigma_ub, width)

    test_img = phi(gblur, ps)

    #test_img .+= 0.1*rand(length(test_img))

    test_img = reshape(test_img, width, width)

    inputs = (test_img, sigma_lb, sigma_ub, 0.0, 0.0, final_loss_improvement, min_weight, max_iters, max_cd_iters)
    records = SeqFISH_ADCG.fit_tile(inputs)

    sorted_results = sortslices(Matrix(records.last_iteration[:,1:4])', dims=2)
    sorted_ps = sortslices(ps, dims=2)

    @test all(isapprox.(sorted_ps, sorted_results, atol = 0.05))

    #test records
    for w in records.records.w
        inputs = (test_img, sigma_lb, sigma_ub, 0.0, 0.0, final_loss_improvement, w - δw, max_iters, max_cd_iters)
        new_records = SeqFISH_ADCG.fit_tile(inputs)
        new_results = new_records.last_iteration[:, Not(:records_idxs)]
        @test all(isapprox.(Matrix(new_results[:,:]), Matrix(get_mw_dots(records.records, w)[:,:]),atol=0.05))
    end
end

@testset "Test Fit Multiple Points" begin
    p_true = [14.33 8.6;
              16.3 10.5;
              1.7 1.6;
              1.0 2.0]
    test_fit_mult_ps(p_true)
    p_true = [14.33 10.6;
              16.3 12.5;
              1.7 1.6;
              1.0 2.0]
    test_fit_mult_ps(p_true)
    p_true = [13.33 10.6;
              15.3 12.5;
              1.7 1.6;
              1.0 2.0]
    test_fit_mult_ps(p_true, 0.5)
    p_true = [14.33 8.6 17.2;
              16.3 10.5 4.3;
              1.7 1.6 1.8;
              1.0 2.0 1.2]
    test_fit_mult_ps(p_true)

end

#println(results[2])
