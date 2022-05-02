using SeqFISH_ADCG
using Test

@testset "2d tests" begin
    include("test_computeGradient_2D.jl")
    include("test_lmo_2d.jl")
    include("test_fit_multiple_points.jl")
    include("test_find_starting_point_2d.jl")
    include("test_fit_2d_img.jl")
end

@testset "3d tests" begin
    include("test_find_starting_point_3d.jl")
    include("test_coord_localDescent_3d.jl")
    include("test_lmo_3d.jl")
    include("test_fit_3d_tile.jl")
end
