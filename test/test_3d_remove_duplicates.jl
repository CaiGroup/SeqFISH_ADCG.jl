using LinearAlgebra
using SeqFISH_ADCG
using Test
using DataFrames


sigma_xy_lb = 1.5
sigma_xy_ub = 1.7
sigma_z_lb = 0.9
sigma_z_ub = 1.1

width = 20
n_slices = 10

points_w_dup = [8.3 8.6 8.9;
          10.2 10.5 10.8;
          0.0 7.2 7.2;
          1.7 1.6 1.65;
          1.1 1.0 0.9;
          1.0 2.0 1.4]

points_w_dup_df = DataFrame(points_w_dup', ["x","y","z","sxy","sz","w"])

points1 = remove_duplicates3d(points_w_dup_df, sigma_xy_lb, sigma_xy_ub, 2.0)
points2 = remove_duplicates_ignore_z(points_w_dup_df, sigma_xy_lb, sigma_xy_ub, 2.0)

@test points1 == points_w_dup_df[1:2,:]
@test points2 == DataFrame(points_w_dup_df[2,:])
