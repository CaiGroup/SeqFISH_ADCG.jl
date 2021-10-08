# Author: Jonathan A. White
# Date Created: February 10, 2021

export fit_img_tiles, fit_2048x2048_img_tiles, fit_tile, remove_duplicates

#using Distributed
using DataFrames
using Statistics
using NearestNeighbors


function fit_tile(inputs)
    #println("fitting tile ... ")
    tile, sigma_lb, sigma_ub, noise_mean, tau, final_loss_improvement, min_weight, max_iters, max_cd_iters = inputs

    n_pixels = maximum(size(tile))
    #min_weight = 200.0
    if any(size(tile) .< n_pixels)
        orig_tile = copy(tile)
        tile = zeros(n_pixels, n_pixels)
        tile[1:size(orig_tile)[1], 1:size(orig_tile)[2]] = orig_tile
    end
    grid = n_pixels*2

    gb_sim = GaussBlur2D(sigma_lb, sigma_ub, n_pixels)#, grid)

    target = vec(tile).-noise_mean#*1.5
    target[target .< 0] .= 0

    sum_target = sum(target)
    if sum_target == 0
        return []
    end

    #tau = sum(target[target .> 0])/final_loss_improvement + 1.0#/5.0 + 1.0
    #tau = sum_target
    tau = sum_target/15 + 1.0 #final_loss_improvement + 1.0
    #println("tau: ", tau)
    function callback(old_thetas,thetas,output,old_obj_val)
      #evalute current OV
      new_obj_val,t = loss(LSLoss(), output - target)
      #println("new_obj_val: ", new_obj_val)
      #println("old_obj_val: ", old_obj_val)
      if old_obj_val - new_obj_val < final_loss_improvement
        #println("old_obj: $old_obj_val")
        #println("new_obj: $new_obj_val")
        return true
      end
      return false
    end
    points = ADCG(gb_sim, LSLoss(), target, tau, min_weight, max_iters=max_iters, callback=callback, max_cd_iters=max_cd_iters)
    return points
end

function trim_tile_fit!(tile_fit, width)
    #ps, ws = tile_fit
    ps = tile_fit
    #if length(ws) == 0
    if length(ps) == 0
        return tile_fit
    end
    xs = ps[1,:]
    ys = ps[2,:]

    to_keep = (xs .> 2) .| (ys .> 2) .| (xs .< width-2) .| (ys .< width-2)

    ps_trim = ps[:, to_keep]
    #ws_trim = ws[to_keep]
    return ps_trim #[ps_trim, ws_trim]
end


function fit_2048x2048_img_tiles(img,
                           sigma_lb :: Float64,
                           sigma_ub :: Float64,
                           tau :: Float64,
                           final_loss_improvement :: Float64,
                           min_weight :: Float64,
                           max_iters :: Int64,
                           max_cd_iters :: Int64,
                           noise_mean :: Float64
        )
    fit_img_tiles(img, 64, 6, sigma_lb, sigma_ub, tau, final_loss_improvement,
                               min_weight,
                               max_iters,
                               max_cd_iters,
                               noise_mean
                )

end

function fit_img_tiles(img,
                       main_tile_width :: Int64,
                       tile_overlap :: Int64,
                       sigma_lb :: Float64,
                       sigma_ub :: Float64,
                       tau :: Float64,
                       final_loss_improvement :: Float64,
                       min_weight :: Float64,
                       max_iters :: Int64,
                       max_cd_iters :: Int64,
                       noise_mean :: Float64
        )

    #@assert size(img) == (2048, 2048)
    img_height, img_width = size(img)

    tile_width = main_tile_width#32
    overhang = tile_overlap

    #tiles_across = Int64(2048/tile_width)
    tiles_across = ceil(Int64, img_width/tile_width)
    bnds_start = tile_width*Array(0:(tiles_across-1)) .+ 1
    bnds_end = tile_width*Array(1:(tiles_across-1)) .+ overhang
    push!(bnds_end, img_width)
    # define overlapping tiles
    coords = [((bnds_start[i]),bnds_end[i], (bnds_start[j]),bnds_end[j]) for i in 1:tiles_across, j in 1:(tiles_across-1)]
    tiles = [img[cds[1]:cds[2],cds[3]:cds[4]] for cds in coords]

    fit_tile_inputs = [(tiles[i], sigma_lb, sigma_ub, noise_mean, tau, final_loss_improvement, min_weight, max_iters, max_cd_iters) for i in 1:length(tiles)]

    #fit tiles
    tile_fits = map(fit_tile, fit_tile_inputs)
    #tile_fits = map(fit_tile, fit_tile_inputs)
    #println("finished fitting tiles, putting together now...")

    #throw out points within 2 pixels of the edge of each tile.
    #trimmed_tile_fits = map(trim_tile_fit!, tile_fits)
    trimmed_tile_fits = [trim_tile_fit!(tile_fit, img_width + tile_overlap) for tile_fit in tile_fits]

    #concatenate points
    ps = []
    #ws = []
    for i = 1:length(trimmed_tile_fits)
        p = trimmed_tile_fits[i]#[1]
        #p .*= (tile_width + overhang)
        if length(p) > 0
            p[1,:] .+= coords[i][3] .- 1
            p[2,:] .+= coords[i][1] .- 1
            push!(ps, p)#strimmed_tile_fits[i][1])
            #push!(ws, trimmed_tile_fits[i][2])
        end
    end

    ps = hcat(ps...)
    #ws = cat(ws..., dims=1)

    #points_df = DataFrame()
    if length(ps) > 0
        points_df = DataFrame(ps', [:x, :y, :s, :w])
    else
        points_df = DataFrame(x=Float64[],y=Float64[],s=Float64[],w=Float64[])
    end
    #points_df[!, "x"] = ps[1,:]
    #points_df[!, "y"] = ps[2,:]
    #points_df[!, "s"] = ps[3,:]
    #points_df[!, "w"] = ws

    return points_df
end

function remove_duplicates(points :: DataFrame,
                           img,
                           sigma_lb :: Float64,
                           sigma_ub :: Float64,
                           tau :: Float64,
                           noise_mean :: Float64,
                           min_allowed_separation :: Float64,
                           dims :: Int64 = 2
                           )

    if nrow(points) == 0
        return points
    end
    ps = Matrix(hcat(points[!, "x"], points[!, "y"], points[!, "s"], points[!, "w"])')
    #ws = points[!, "w"]
    #ps, ws = remove_duplicates(ps, ws, img, sigma_lb, sigma_ub, tau, noise_mean, min_allowed_separation, dims)
    ps = remove_duplicates(ps, img, sigma_lb, sigma_ub, tau, noise_mean, min_allowed_separation, dims)
    points_df = DataFrame()
    points_df[!, "x"] = ps[1,:]
    points_df[!, "y"] = ps[2,:]
    points_df[!, "s"] = ps[3,:]
    points_df[!, "w"] = ps[4,:]#ws
    return points_df
end

function remove_duplicates(ps :: Matrix,
                           img,
                           sigma_lb :: Float64,
                           sigma_ub :: Float64,
                           tau :: Float64,
                           noise_mean :: Float64,
                           min_allowed_separation :: Float64,
                           dims :: Int64
                           )

    #run solveFiniteDimProblem to remove duplicate points
    #coords = ps[1:2, :]
    coords = ps[1:dims, :]
    #n_ps_init = length(ws)
    #gb_sim = GaussBlur2D(sigma_lb, sigma_ub, 2048, 4096)
    target = vec(img).-noise_mean
    target[target .< 0] .= 0
    tau = sum(target)
    bt = BallTree(coords)
    grps = inrange(bt, coords, min_allowed_separation, true)
    lone_point_grps = filter(g->length(g) == 1, grps)
    lone_point_inds = map(ia->ia[1], lone_point_grps)

    lone_points = ps[:, lone_point_inds]
    #lone_weights = ws[lone_point_inds]

    filter!(g-> length(g)>1, grps)
    grps = unique(grps)

    grp_brightest_inds = []
    #println("n groups: ", length(grps))
    for grp in grps
        #push!(grp_brightest_inds, grp[argmax(ws[grp])])
        push!(grp_brightest_inds, grp[argmax(ps[4,grp])])
    end
    sort!(grp_brightest_inds)
    bright_dup_ps = ps[:, grp_brightest_inds]
    #bright_dup_ws = ws[grp_brightest_inds]

    #nremoved = length(ws) - length(lone_weights) - length(bright_dup_ws)
    #if nremoved > 0
        #println("removed ", nremoved, " duplicates")
    #end

    #recalculate weights and prune
    #lone_point_model = phi(gb_sim, lone_points, lone_weights)
    #target2 = target .- lone_point_model
    #println("solveFDP")
    #re_ws = solveFiniteDimProblem(gb_sim, LSLoss(), bright_dup_ps, target2, tau)
    #bright_dup_ps = bright_dup_ps[:,re_ws.!= 0.0]
    #re_ws = re_ws[re_ws.!= 0.0]

    #ps = hcat(lone_points, bright_dup_ps)
    #ws = cat(lone_weights, re_ws, dims=1)

    ps = hcat(lone_points, bright_dup_ps)
    #ws = cat(lone_weights, bright_dup_ws, dims=1)
    return ps#, ws
end
