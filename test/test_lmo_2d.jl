using LinearAlgebra
using SeqFISH_ADCG
using Test
using Random

Random.seed!(1)

sigma_lb = 1.2
sigma_ub = 2.5
width = 32

function test_lmo(p_true)
    p_true = reshape(p_true, 4, 1)

    gblur = GaussBlur2D(sigma_lb, sigma_ub, width)

    test_img = phi(gblur, p_true)
    residuals = test_img

    params, objv = lmo(gblur, residuals)

    @test all(isapprox.(p_true, params, atol = 0.01))

end


@testset "Test LMO" begin
    p_true = [16.33 16.3 2.0 1.0]
    test_lmo(p_true)
    for i = 1:1000
        p = rand(4) .* [8.0, 8.0, 0.5, 1.0] .+ [12.0, 12.0, 1.6, 1.0]
        test_lmo(p)
    end
end
