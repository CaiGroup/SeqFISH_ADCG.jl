using SeqFISH_ADCG
using Test

# write your own tests here
include("test_computeGradient_2D.jl")
include("test_lmo_2d.jl")
include("test_fit_multiple_points.jl")

@testset "misc. 2d tests" begin
    include("test_find_starting_point_2d.jl")
    include("test_fit_2d_img.jl")
end

"""
@testset "misc. 3d tests" begin
    include("test_coord_localDescent_3d.jl")
    include("test_fistarting_point_3d.jl")
end
"""